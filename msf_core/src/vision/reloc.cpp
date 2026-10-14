#include "ms_fusion/vision/reloc.h"
#include "ms_fusion/vision/yolo_obb.h"
#include "ms_fusion/camera/create_camera.h"
#include "ms_fusion/common/rotation.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/padded_image_geometry.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/factors/pose_mainfold.h"
#include "ms_fusion/factors/reloc_reprojection.h"
#include "ms_fusion/manager/ms_fusion_manager.h"
#include "ms_fusion/vision/drawer.h"
#include "ms_fusion/vision/upright_p2_solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace ms_fusion {

namespace {

struct RelocCameraGeometry {
    std::vector<double> intrinsics;
    cv::Mat camera_matrix;
    cv::Mat distortion;
    int content_width = 0;
    int content_height = 0;
    int pad_right = 0;
    int pad_bottom = 0;
};

RelocCameraGeometry makeRelocCameraGeometry(
    const CameraConfigParams& camera_config, size_t camera_index, int output_width,
    int output_height) {
    if (camera_index >= camera_config.cam_resolutions.size() ||
        camera_index >= camera_config.cam_intrinsics.size() ||
        camera_index >= camera_config.cam_distortions.size()) {
        throw std::runtime_error("Incomplete reloc camera calibration");
    }

    const auto& resolution = camera_config.cam_resolutions.at(camera_index);
    const auto& calibration_intrinsics = camera_config.cam_intrinsics.at(camera_index);
    const auto& calibration_distortion = camera_config.cam_distortions.at(camera_index);
    if (resolution.size() != 2 || calibration_intrinsics.size() != 4 ||
        calibration_distortion.empty() || resolution[0] <= 0 || resolution[1] <= 0 ||
        output_width <= 0 || output_height <= 0) {
        throw std::runtime_error("Invalid reloc camera calibration or output resolution");
    }

    const auto image_geometry = makeTopLeftPaddedImageGeometry(
        resolution[0], resolution[1], output_width, output_height);
    const auto scaled_intrinsics = scalePinholeIntrinsicsTopLeft(
        calibration_intrinsics[0], calibration_intrinsics[1],
        calibration_intrinsics[2], calibration_intrinsics[3], image_geometry);

    RelocCameraGeometry geometry;
    geometry.content_width = image_geometry.content_width;
    geometry.content_height = image_geometry.content_height;
    geometry.pad_right = image_geometry.pad_right;
    geometry.pad_bottom = image_geometry.pad_bottom;
    geometry.intrinsics.assign(scaled_intrinsics.begin(), scaled_intrinsics.end());
    geometry.camera_matrix =
        (cv::Mat_<double>(3, 3) << geometry.intrinsics[0], 0.0, geometry.intrinsics[2],
         0.0, geometry.intrinsics[1], geometry.intrinsics[3], 0.0, 0.0, 1.0);
    geometry.distortion = cv::Mat(1, static_cast<int>(calibration_distortion.size()), CV_64F);
    for (size_t i = 0; i < calibration_distortion.size(); ++i) {
        geometry.distortion.at<double>(0, static_cast<int>(i)) = calibration_distortion[i];
    }
    return geometry;
}

std::vector<double> intrinsicsFromCameraMatrix(const cv::Mat& camera_matrix) {
    return {
        camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1),
        camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 2),
    };
}

bool hasNonCollinearWorldPoints(const ReLocFeatInfo& observations) {
    Eigen::aligned_vector<Eigen::Vector3d> unique_points;
    unique_points.reserve(observations.size());
    for (const auto& feature : observations) {
        if (!feature.second.empty()) {
            unique_points.push_back(feature.second.front().second.block<3, 1>(2, 0));
        }
    }
    if (unique_points.size() < 3) {
        return false;
    }

    constexpr double kMinTriangleArea2 = 1e-6;
    for (size_t i = 0; i + 2 < unique_points.size(); ++i) {
        for (size_t j = i + 1; j + 1 < unique_points.size(); ++j) {
            for (size_t k = j + 1; k < unique_points.size(); ++k) {
                if (((unique_points[j] - unique_points[i])
                         .cross(unique_points[k] - unique_points[i]))
                        .norm() > kMinTriangleArea2) {
                    return true;
                }
            }
        }
    }
    return false;
}

// 统计同一个 feature_id 同时有 cam0 和 cam1 观测的地标数量（"成对地标"）。
// 只看左右是否都命中，不关心具体像素/世界坐标，用于 checkPose() 的运行期
// 数量门槛。
int countStereoPairLandmarks(const ReLocFeatInfo& observations) {
    int count = 0;
    for (const auto& feature : observations) {
        bool has_cam0 = false;
        bool has_cam1 = false;
        for (const auto& observation : feature.second) {
            if (observation.first == 0) {
                has_cam0 = true;
            } else if (observation.first == 1) {
                has_cam1 = true;
            }
        }
        if (has_cam0 && has_cam1) {
            ++count;
        }
    }
    return count;
}

// 诊断用（不参与 checkPose 判定）：inlier 集合里地面点（KEYPOINT_WORLD z==0，
// 场地标定点里占多数的那批）与非地面点（立柱等结构点，z!=0）各有多少个，以及
// 非地面点里左右目都命中的对数。用来现场核实"近共面配置导致 z 病态"这个猜测
// 站不站得住——每个 feature_id 的世界坐标对所有观测都一样，取第一条即可。
struct LandmarkPlaneStats {
    int ground_landmarks = 0;
    int off_plane_landmarks = 0;
    int off_plane_stereo_pairs = 0;
};

LandmarkPlaneStats classifyLandmarksByPlane(const ReLocFeatInfo& observations) {
    LandmarkPlaneStats stats;
    for (const auto& feature : observations) {
        if (feature.second.empty()) {
            continue;
        }
        const double world_z = feature.second.front().second(4);
        if (std::abs(world_z) > 1e-6) {
            ++stats.off_plane_landmarks;
            bool has_cam0 = false;
            bool has_cam1 = false;
            for (const auto& observation : feature.second) {
                if (observation.first == 0) {
                    has_cam0 = true;
                } else if (observation.first == 1) {
                    has_cam1 = true;
                }
            }
            if (has_cam0 && has_cam1) {
                ++stats.off_plane_stereo_pairs;
            }
        } else {
            ++stats.ground_landmarks;
        }
    }
    return stats;
}

// 双目三角化+重投影校验的结果。ok=false 时 reason 说明具体卡在哪一步；
// depth0/depth1/residual*_px 无论过没过都尽量填，方便诊断打印。
struct StereoConsistencyCheck {
    bool ok = false;
    double depth0 = std::numeric_limits<double>::quiet_NaN();
    double depth1 = std::numeric_limits<double>::quiet_NaN();
    double residual0_px = -1.0;
    double residual1_px = -1.0;
    const char* reason = "not_checked";
};

// 用跟 FeatureManager::triangulatePoint() 相同的单变量最小二乘公式（同一个
// 全局 camera_extrinsic_params.T_cam1_from_cam0），把 cam0/cam1 两条归一化
// 视线三角化成一个 3D 点，再校验：
//   1) 两个相机系下深度都必须为正（点在相机前方）；
//   2) 深度必须落在这个场地应用场景的合理范围内（min/max_depth_m）——纯极线
//      误差可能很小（两条视线恰好在空间上接近），但深度可以离谱到几十上百米，
//      光看极线误差不够，这一步专门补上；
//   3) 三角化出的 3D 点分别投影回两个相机，跟实际观测像素的残差都要在
//      max_reprojection_error_px 以内。
// pn0/pn1 是 (x, y, 1) 形式的归一化平面坐标（ReLocFeatInfo 里存的原始
// unproject 结果，不是单位球面向量）。
StereoConsistencyCheck checkStereoConsistency(
    const Eigen::Vector2d& pixel0, const Eigen::Vector3d& pn0, const Eigen::Vector2d& pixel1,
    const Eigen::Vector3d& pn1, const std::vector<std::shared_ptr<GenericCamera>>& cameras,
    double min_depth_m, double max_depth_m, double max_reprojection_error_px) {
    StereoConsistencyCheck result;
    if (!pn0.allFinite() || !pn1.allFinite()) {
        result.reason = "invalid_bearing";
        return result;
    }

    const Eigen::Matrix3d& R10 = camera_extrinsic_params.T_cam1_from_cam0.R;
    const Eigen::Vector3d& t10 = camera_extrinsic_params.T_cam1_from_cam0.t;

    const Eigen::Vector3d m = R10 * pn0;
    const Eigen::Vector2d A(m(0) - pn1(0) * m(2), m(1) - pn1(1) * m(2));
    const Eigen::Vector2d b(pn1(0) * t10(2) - t10(0), pn1(1) * t10(2) - t10(1));
    const double denom = A.squaredNorm();
    if (!std::isfinite(denom) || denom < 1e-12 || !b.allFinite()) {
        result.reason = "degenerate_rays";
        return result;
    }

    const double depth0 = A.dot(b) / denom;
    result.depth0 = depth0;
    if (!std::isfinite(depth0) || depth0 <= 0.0) {
        result.reason = "non_positive_depth0";
        return result;
    }

    const Eigen::Vector3d point_cam0 = depth0 * pn0;
    const Eigen::Vector3d point_cam1 = R10 * point_cam0 + t10;
    result.depth1 = point_cam1.z();
    if (!std::isfinite(result.depth1) || result.depth1 <= 0.0) {
        result.reason = "non_positive_depth1";
        return result;
    }

    if (depth0 < min_depth_m || depth0 > max_depth_m || result.depth1 < min_depth_m ||
        result.depth1 > max_depth_m) {
        result.reason = "depth_out_of_range";
        return result;
    }

    Eigen::Vector2d reproj0 = Eigen::Vector2d::Zero();
    Eigen::Vector2d reproj1 = Eigen::Vector2d::Zero();
    cameras.at(0)->project(point_cam0, reproj0);
    cameras.at(1)->project(point_cam1, reproj1);
    if (!reproj0.allFinite() || !reproj1.allFinite()) {
        result.reason = "reprojection_failed";
        return result;
    }
    result.residual0_px = (reproj0 - pixel0).norm();
    result.residual1_px = (reproj1 - pixel1).norm();
    if (result.residual0_px > max_reprojection_error_px ||
        result.residual1_px > max_reprojection_error_px) {
        result.reason = "reprojection_residual_too_large";
        return result;
    }

    result.ok = true;
    result.reason = "ok";
    return result;
}

// 假设/内点阶段的几何摘要（日志字段沿用 ransac={...}，离线脚本按它解析）。
// available=false 表示这一阶段还没跑到（比如候选本身就没通过共线性检查，
// 根本没机会求解），此时不应该拿 0 冒充"跑过但结果是 0"。
struct RansacGeometryStats {
    bool available = false;
    int inliers = -1;  // 最佳假设的支持数（主观测通过角度阈值的地标数）
    int inlier_landmarks = -1;
    int stereo_pairs = -1;
    int offplane_stereo_pairs = -1;
    int models = -1;   // 正深度假设总数
    std::string inlier_ids;  // 内点地标 ID 列表（"[a,b,...]"）
};

// 求出最佳假设之后各阶段共用的摘要：支持数/假设数取自本帧 timing，
// 内点几何按传入的内点集合统计。
RansacGeometryStats makeRansacGeometryStats(const ReLocFeatInfo& inlier_feat_info,
                                            const RelocProcessTiming& timing,
                                            const std::string& inlier_ids);

struct RelocPoseSummary {
    bool available = false;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
};

struct RelocMetricsSummary {
    bool available = false;
    double avg_reprojection_error_px = -1.0;
    int feat_num = -1;
};

// 本帧最后一次打印的 stage，供 ReLocMeasurement::main_stage。重定位只在单个
// 线程里串行处理，ReLoc 也只有一个实例，用文件内静态状态即可。
std::string g_reloc_last_stage;

// 每个候选一次结果只打一条（单行）日志：候选阶段（求解之前）的几何统计和
// 假设/内点阶段的几何统计分开成两个字段，不能互相覆盖——两者经常不同
// （评分会剔点），混在一起会看不出"同一批候选、选出错误假设"这种问题。
// pose/metrics 只在算出来的阶段才打印。
void printRelocResult(const char* verdict, const char* stage, double timestamp,
                      const std::string& candidate_summary, const RansacGeometryStats& ransac,
                      const RelocPoseSummary& pose, const RelocMetricsSummary& metrics,
                      const std::string& stereo_filter_summary = std::string()) {
    g_reloc_last_stage = stage;
    std::string out;
    char line[192];

    std::snprintf(line, sizeof(line), "[Reloc] %s stage=%s t=%.6f", verdict, stage, timestamp);
    out += line;

    std::snprintf(line, sizeof(line), " candidate=%s", candidate_summary.c_str());
    out += line;

    if (!stereo_filter_summary.empty()) {
        out += " stereo_filter=";
        out += stereo_filter_summary;
    }

    if (ransac.available) {
        std::snprintf(line, sizeof(line),
                      " ransac={inliers=%d,inlier_lm=%d,pairs=%d,offplane_pairs=%d,models=%d",
                      ransac.inliers, ransac.inlier_landmarks, ransac.stereo_pairs,
                      ransac.offplane_stereo_pairs, ransac.models);
        out += line;
        if (!ransac.inlier_ids.empty()) {
            out += ",ids=";
            out += ransac.inlier_ids;
        }
        out += "}";
    } else {
        out += " ransac={not_run}";
    }

    if (pose.available) {
        std::snprintf(line, sizeof(line),
                      " pose={x=%.3f,y=%.3f,z=%.3f,roll=%.2f,pitch=%.2f,yaw=%.2f}", pose.x,
                      pose.y, pose.z, pose.roll_deg, pose.pitch_deg, pose.yaw_deg);
        out += line;
    }

    if (metrics.available) {
        std::snprintf(line, sizeof(line), " metrics={avg_reprojection_error_px=%.2f,feat_num=%d}",
                      metrics.avg_reprojection_error_px, metrics.feat_num);
            out += line;
    }

    // 单帧结果不着色：通过单帧检查还要经多帧确认/门控才会被应用，颜色留给
    // "修正被应用 / 被门控拒绝"（见 MsFusionManager 与 Loop 侧的 reloc 日志）
    out += "\n";

    // 一次性写出，避免跟其他线程（比如 pipeline_timing 的周期打印）的输出交错。
    MSF_ILOG << out;
}

RansacGeometryStats makeRansacGeometryStats(const ReLocFeatInfo& inlier_feat_info,
                                            const RelocProcessTiming& timing,
                                            const std::string& inlier_ids) {
    RansacGeometryStats stats;
    stats.available = true;
    stats.inliers = timing.ransac_inliers;
    stats.models = timing.ransac_iterations;
    stats.inlier_ids = inlier_ids;
    stats.inlier_landmarks = static_cast<int>(inlier_feat_info.size());
    stats.stereo_pairs = countStereoPairLandmarks(inlier_feat_info);
    stats.offplane_stereo_pairs = classifyLandmarksByPlane(inlier_feat_info).off_plane_stereo_pairs;
    return stats;
}

}  // namespace

// 构造函数
ReLoc::ReLoc(const RelocConfigParams& reloc_config_params,
             const CameraConfigParams& camera_config_params)
    : reloc_image_width_(reloc_config_params.resolution_width),
      reloc_image_height_(reloc_config_params.resolution_height),
      undistort_enabled_(reloc_config_params.enable_undistort),
      publish_debug_image_(reloc_config_params.publish_debug_image),
      max_height_m_(reloc_config_params.max_height_m),
      max_reprojection_error_px_(reloc_config_params.max_reprojection_error_px),
      min_offplane_stereo_pairs_(reloc_config_params.min_offplane_stereo_pairs),
      inlier_pixel_threshold_px_(reloc_config_params.inlier_pixel_threshold_px),
      min_support_(reloc_config_params.min_support) {
    if (camera_config_params.cam_models.size() != 2) {
        throw std::runtime_error("Reloc requires exactly two cameras");
    }

    cameras_.reserve(2);
    const cv::Size output_size(reloc_image_width_, reloc_image_height_);
    for (size_t camera_index = 0; camera_index < 2; ++camera_index) {
        const auto geometry = makeRelocCameraGeometry(
            camera_config_params, camera_index, reloc_image_width_, reloc_image_height_);

        cv::Mat output_camera_matrix = geometry.camera_matrix;
        std::vector<double> output_distortion =
            camera_config_params.cam_distortions.at(camera_index);
        cv::Rect valid_roi(0, 0, geometry.content_width, geometry.content_height);

        if (undistort_enabled_) {
            // Knew is selected only from the calibrated content. Both input and output
            // content start at (0,0), so padding does not shift the principal point.
            const cv::Size content_size(geometry.content_width, geometry.content_height);
            output_camera_matrix = cv::getOptimalNewCameraMatrix(
                geometry.camera_matrix, geometry.distortion, content_size,
                reloc_config_params.undistort_alpha, content_size, &valid_roi, false);
            cv::initUndistortRectifyMap(
                geometry.camera_matrix, geometry.distortion, cv::Mat::eye(3, 3, CV_64F),
                output_camera_matrix, output_size, CV_16SC2, undistort_map1_[camera_index],
                undistort_map2_[camera_index]);
            output_distortion.assign(output_distortion.size(), 0.0);
        }
        valid_rois_[camera_index] = valid_roi;

        const auto output_intrinsics = intrinsicsFromCameraMatrix(output_camera_matrix);
        if (!std::isfinite(output_intrinsics[0]) || !std::isfinite(output_intrinsics[1]) ||
            !std::isfinite(output_intrinsics[2]) || !std::isfinite(output_intrinsics[3]) ||
            output_intrinsics[0] <= 0.0 || output_intrinsics[1] <= 0.0) {
            throw std::runtime_error("Invalid reloc output camera matrix");
        }
        cameras_.push_back(CreateCamera::createCamera(
            output_intrinsics, output_distortion,
            camera_config_params.cam_models.at(camera_index)));

        MSF_CLOG << "[Reloc] cam" << camera_index << " content="
                 << geometry.content_width << "x" << geometry.content_height
                 << " at (0,0) pad_right=" << geometry.pad_right << " pad_bottom="
                 << geometry.pad_bottom << " input_K=["
                 << geometry.intrinsics[0] << ", " << geometry.intrinsics[1] << ", "
                 << geometry.intrinsics[2] << ", " << geometry.intrinsics[3]
                 << "] undistort=" << (undistort_enabled_ ? "true" : "false")
                 << " output_K=[" << output_intrinsics[0] << ", " << output_intrinsics[1]
                 << ", " << output_intrinsics[2] << ", " << output_intrinsics[3]
                 << "] valid_roi=[" << valid_roi.x << ", " << valid_roi.y << ", "
                 << valid_roi.width << ", " << valid_roi.height << "]" << std::endl;
    }

    // 构造YOLO网络指针
    yolo_net_ = std::make_shared<YOLO>(reloc_config_params, cameras_);
}

// 重定位过程
bool ReLoc::process(double timestamp, const cv::Mat& image0, 
                    const cv::Mat& image1) {
    using Clock = MonotonicRawClock;
    last_process_timing_ = {};
    g_reloc_last_stage.clear();
    reloc_measurement_.main_stage.clear();

    // 赋值当前的时间戳
    timestamp_ = timestamp;

    if (image0.size() != cv::Size(reloc_image_width_, reloc_image_height_) ||
        image1.size() != cv::Size(reloc_image_width_, reloc_image_height_)) {
        MSF_CLOG << "[Reloc] unexpected input size cam0=" << image0.cols << "x" << image0.rows
                 << " cam1=" << image1.cols << "x" << image1.rows << ", expected "
                 << reloc_image_width_ << "x" << reloc_image_height_ << std::endl;
        reloc_measurement_.timestamp = timestamp;
        reloc_measurement_.is_reloc_success = false;
        reloc_measurement_.T_w_from_imu = Pose();
        return false;
    }

    cv::Mat undistorted_image0;
    cv::Mat undistorted_image1;
    const cv::Mat* yolo_image0 = &image0;
    const cv::Mat* yolo_image1 = &image1;
    if (undistort_enabled_) {
        const auto undistort_begin = Clock::now();
        cv::remap(image0, undistorted_image0, undistort_map1_[0], undistort_map2_[0],
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::remap(image1, undistorted_image1, undistort_map1_[1], undistort_map2_[1],
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        const auto mask_invalid_remap_pixels = [this](cv::Mat& image, size_t camera_index) {
            const cv::Rect image_bounds(0, 0, image.cols, image.rows);
            const cv::Rect valid_roi = valid_rois_[camera_index] & image_bounds;
            if (valid_roi.empty()) {
                image.setTo(cv::Scalar::all(0));
                return;
            }

            // getOptimalNewCameraMatrix() 给出的 ROI 是和 output_K 同一像素坐标系的
            // 唯一可靠区域。此前只清理输入 padding，遗留的 remap 无效边缘会呈现
            // 绿色/紫色条纹并可能被 YOLO 当成场地线。
            if (valid_roi.y > 0) {
                image.rowRange(0, valid_roi.y).setTo(cv::Scalar::all(0));
            }
            const int valid_bottom = valid_roi.y + valid_roi.height;
            if (valid_bottom < image.rows) {
                image.rowRange(valid_bottom, image.rows).setTo(cv::Scalar::all(0));
            }
            if (valid_roi.x > 0) {
                image.colRange(0, valid_roi.x).setTo(cv::Scalar::all(0));
            }
            const int valid_right = valid_roi.x + valid_roi.width;
            if (valid_right < image.cols) {
                image.colRange(valid_right, image.cols).setTo(cv::Scalar::all(0));
            }
        };
        mask_invalid_remap_pixels(undistorted_image0, 0);
        mask_invalid_remap_pixels(undistorted_image1, 1);
        last_process_timing_.undistort_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - undistort_begin).count();
        yolo_image0 = &undistorted_image0;
        yolo_image1 = &undistorted_image1;
    }

    // 通过网络，求重定位的信息。YOLO 输出与 cameras_ 始终处于同一像素域。
    yolo_net_->process(timestamp, *yolo_image0, *yolo_image1);
    last_process_timing_.yolo_ran = 1;

    if (publish_debug_image_) {
        const auto debug_begin = Clock::now();
        const auto& debug_info = yolo_net_->getDebugInfo();
        DebugImageMessage debug_image;
        debug_image.timestamp = timestamp;
        debug_image.image = Drawer::drawRelocStereoDebug(
            debug_info.image0, debug_info.image1, debug_info);
        if (!debug_image.image.empty()) {
            MsFusionManager::SnglPtr()->pubDebugImageToNode("reloc_debug", debug_image);
        }
        last_process_timing_.debug_publish_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - debug_begin).count();
    }

    // 拿到网络检测到的特征点的信息
    double curr_timestamp = 0.0;
    ReLocFeatInfo curr_reloc_feat_info;
    const auto feature_pack_begin = Clock::now();
    yolo_net_->info(curr_timestamp, curr_reloc_feat_info);
    last_process_timing_.feature_pack_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - feature_pack_begin).count();
    last_process_timing_.unique_landmarks = static_cast<int>(curr_reloc_feat_info.size());
    last_process_timing_.raw_observations = 0;
    for (const auto& feature : curr_reloc_feat_info) {
        last_process_timing_.raw_observations += static_cast<int>(feature.second.size());
    }

    // 双目一致性过滤：在 calcPose() 之前，把左右目三角化对不上的
    // feature_id 整个剔除，避免同一 class 多检测时左右目各自选中不同物理
    // 实例的观测混进候选集。过滤结果汇总进 stereo_filter_log_，拼进这一帧
    // 最终的 ACCEPTED/REJECTED 那一行（见 printRelocResult()）。
    filterStereoInconsistentLandmarks(curr_reloc_feat_info);

    const auto pose_begin = Clock::now();

    // 计算初始位姿
    Pose T_w_from_imu = Pose();
    ReLocFeatInfo inlier_feat_info;
    bool calc_success = calcPose(curr_reloc_feat_info, T_w_from_imu, inlier_feat_info);

    if (calc_success) {
        // calcPose 只给出最佳 2 点假设下的内点和最小解（未过门），重力先验 refine
        // 后整体过 checkPose；refine 失败或验收不过即整帧拒绝，最小解不作回退。
        last_process_timing_.refine_success = 0;
        const auto refine_begin = Clock::now();
        const bool refine_solved = optimizePose(inlier_feat_info, T_w_from_imu);
        last_process_timing_.refine_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - refine_begin).count();
        if (!refine_solved) {
            const RansacGeometryStats ransac_stats = makeRansacGeometryStats(
                inlier_feat_info, last_process_timing_, last_inlier_ids_log_);
            printRelocResult("REJECTED", "refine_failed", timestamp_, candidate_geometry_log_,
                             ransac_stats, RelocPoseSummary(), RelocMetricsSummary(),
                             stereo_filter_log_);
            calc_success = false;
        } else {
            const auto validate_begin = Clock::now();
            calc_success = checkPose(inlier_feat_info, T_w_from_imu);
            last_process_timing_.validate_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - validate_begin).count();
            last_process_timing_.refine_success = calc_success ? 1 : 0;
        }
    }

    last_process_timing_.pose_total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - pose_begin).count();
    last_process_timing_.pose_success = calc_success ? 1 : 0;

    // 精度诊断，供评估重定位效果用：反投影误差是唯一不依赖真值、能独立反映
    // 这一帧位姿准确度的量；T_w_from_imu 此时已经是最终采用的 refine 位姿，
    // inlier_feat_info 是这个结果对应的内点集合，跟 checkPose() 里用的是同一份数据。
    if (calc_success) {
        int feat_num = 0;
        last_process_timing_.avg_reprojection_error_px =
            computeAverageReprojectionError(inlier_feat_info, T_w_from_imu, &feat_num);
        // 只记录最终采用的结果，checkPose() 里不打印帧级 ACCEPTED。
        const Eigen::Vector3d euler_angle = Rotation::matrix2euler(T_w_from_imu.R);

        const RansacGeometryStats ransac_stats = makeRansacGeometryStats(
            inlier_feat_info, last_process_timing_, last_inlier_ids_log_);

        RelocPoseSummary pose_summary;
        pose_summary.available = true;
        pose_summary.x = T_w_from_imu.t.x();
        pose_summary.y = T_w_from_imu.t.y();
        pose_summary.z = T_w_from_imu.t.z();
        pose_summary.roll_deg = euler_angle[0] * R2D;
        pose_summary.pitch_deg = euler_angle[1] * R2D;
        pose_summary.yaw_deg = euler_angle[2] * R2D;

        RelocMetricsSummary metrics_summary;
        metrics_summary.available = true;
        metrics_summary.avg_reprojection_error_px = last_process_timing_.avg_reprojection_error_px;
        metrics_summary.feat_num = feat_num;

        printRelocResult("ACCEPTED", "accepted", timestamp_, candidate_geometry_log_, ransac_stats,
                         pose_summary, metrics_summary, stereo_filter_log_);

        last_process_timing_.pose_x = pose_summary.x;
        last_process_timing_.pose_y = pose_summary.y;
        last_process_timing_.pose_z = pose_summary.z;
        last_process_timing_.pose_yaw_deg = pose_summary.yaw_deg;
    }

    reloc_measurement_.main_stage =
        calc_success ? "accepted" : (g_reloc_last_stage.empty() ? "unknown" : g_reloc_last_stage);

    // 赋值
    reloc_measurement_.timestamp = timestamp;
    reloc_measurement_.is_reloc_success = calc_success;
    reloc_measurement_.T_w_from_imu = T_w_from_imu;
    return calc_success;
}

bool ReLoc::isValidObservation(
    const ReLocFeatInfo::mapped_type::value_type& feat_pair) const {
    const int cam_id = feat_pair.first;
    if (cam_id < 0 || static_cast<size_t>(cam_id) >= cameras_.size()) {
        return false;
    }
    const auto& observation = feat_pair.second;
    const Eigen::Vector2d pixel_point = observation.template block<2, 1>(0, 0);
    if (!pixel_point.allFinite() || pixel_point.x() < 0 ||
        pixel_point.x() >= reloc_image_width_ ||
        pixel_point.y() < 0 || pixel_point.y() >= reloc_image_height_) {
        return false;
    }
    const Eigen::Vector3d world_point = observation.template block<3, 1>(2, 0);
    const Eigen::Vector3d raw_bearing = observation.template block<3, 1>(5, 0);
    return world_point.allFinite() && raw_bearing.allFinite() && raw_bearing.norm() >= 1e-12;
}

/**
 * 返回信息
 */
void ReLoc::info(ReLocMeasurement& reloc_meassurement_info)  {
    // 返回信息
    reloc_meassurement_info = reloc_measurement_;

    return;
}

const RelocDebugInfo& ReLoc::getDebugInfo() const {
    return yolo_net_->getDebugInfo();
}

cv::Mat ReLoc::renderYoloDebugImage() const {
    const auto& debug_info = yolo_net_->getDebugInfo();
    return Drawer::drawRelocStereoDebug(debug_info.image0, debug_info.image1, debug_info);
}

const YoloFrameTiming& ReLoc::yoloFrameTiming() const {
    return yolo_net_->lastFrameTiming();
}

void ReLoc::filterStereoInconsistentLandmarks(ReLocFeatInfo& reloc_feat_info) {
    // 这个场地应用场景的合理深度范围：现场数据里正常工作距离在 9~12m 量级，
    // 这里留足余量（3~30m）——目的不是精确卡场地边界，是把明显错配挡在门外，
    // 纯极线误差挡不住这种情况（见诊断历史：某次错配极线误差只有 0.14px，
    // 深度却有 180m）。下限原来是 0.5m，现场数据证实过太松：KP20 一次错配
    // 三角化深度只有 1.17m，仍然落在 0.5~30m 范围内被放过——收紧到 3m，
    // 场景里从没出现过合法点近于这个距离。
    constexpr double kMinDepthM = 3.0;
    constexpr double kMaxDepthM = 30.0;

    int checked = 0;
    std::vector<int> feature_ids_to_erase;
    for (auto& feature : reloc_feat_info) {
        const int feature_id = feature.first;
        auto& observations = feature.second;
        if (observations.size() != 2) {
            // 只有单目观测，或者理论上不该出现的同相机重复观测，都不在这一步
            // 的处理范围内——双目一致性检查天然需要恰好一对 cam0+cam1。
            continue;
        }

        const bool obs0_is_cam0 = observations[0].first == 0;
        const auto& obs_cam0 = obs0_is_cam0 ? observations[0] : observations[1];
        const auto& obs_cam1 = obs0_is_cam0 ? observations[1] : observations[0];
        if (obs_cam0.first != 0 || obs_cam1.first != 1) {
            // 不是恰好一个 cam0 + 一个 cam1（比如同一相机出现两条观测），
            // 不是这一步要处理的情况，跳过。
            continue;
        }

        ++checked;
        const Eigen::Vector2d pixel0 = obs_cam0.second.block<2, 1>(0, 0);
        const Eigen::Vector2d pixel1 = obs_cam1.second.block<2, 1>(0, 0);
        const Eigen::Vector3d pn0 = obs_cam0.second.block<3, 1>(5, 0);
        const Eigen::Vector3d pn1 = obs_cam1.second.block<3, 1>(5, 0);

        const StereoConsistencyCheck check = checkStereoConsistency(
            pixel0, pn0, pixel1, pn1, cameras_, kMinDepthM, kMaxDepthM,
            max_reprojection_error_px_);

        if (!check.ok) {
            // 两侧观测都不可信——不知道该信哪一侧，两个都删比留一个猜的更
            // 稳妥（见头文件声明处注释）。
            feature_ids_to_erase.push_back(feature_id);
        }
    }

    for (const int feature_id : feature_ids_to_erase) {
        reloc_feat_info.erase(feature_id);
    }

    // 汇总成一小段文本，拼进这一帧最终的 ACCEPTED/REJECTED 那一行（见
    // printRelocResult() 的 stereo_filter 字段），不再逐点单独打一行——
    // 双目一致性问题的定位阶段已经过去，现在只需要知道"这一帧过滤掉了几个"。
    std::string log = "{checked=" + std::to_string(checked) + ",dropped=" +
                      std::to_string(feature_ids_to_erase.size());
    if (!feature_ids_to_erase.empty()) {
        log += ",kp=[";
        for (size_t i = 0; i < feature_ids_to_erase.size(); ++i) {
            if (i > 0) log += ",";
            log += std::to_string(feature_ids_to_erase[i]);
        }
        log += "]";
    }
    log += "}";
    stereo_filter_log_ = log;
}

/**
 * 根据网络提取的特征点信息，计算位姿
 */
bool ReLoc::calcPose(const ReLocFeatInfo& reloc_feat_info, Pose& T_w_from_imu,
                     ReLocFeatInfo& inlier_feat_info) {
    const auto correspondence_begin = MonotonicRawClock::now();
    inlier_feat_info.clear();
    last_inlier_ids_log_.clear();

    const auto cam_extrinsic = [](int cam_id) -> const Pose& {
        return cam_id == 0 ? camera_imu_extrinsic_params.T_imu_from_cam0
                           : camera_imu_extrinsic_params.T_imu_from_cam1;
    };

    // 每个物理地标的主观测：世界点、光心与单位视线（均在机体/IMU 系）
    struct Primary {
        int feature_id = -1;
        Eigen::Vector3d P = Eigen::Vector3d::Zero();
        Eigen::Vector3d c = Eigen::Vector3d::Zero();
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
    };
    std::vector<Primary> primaries;
    ReLocFeatInfo valid_feat_info;

    int valid_observation_count = 0;
    for (const auto& feature : reloc_feat_info) {
        const int feature_id = feature.first;

        // 假设生成与评分去重：每个物理地标只取一个合法观测（优先左目），双目同点
        // 不重复计票；其余合法观测留在 valid_feat_info，选出假设后再逐条校验
        // （见下方内点回取）。
        const ReLocFeatInfo::mapped_type::value_type* best_obs = nullptr;
        for (const auto& obs : feature.second) {
            if (!isValidObservation(obs)) {
                continue;
            }
            valid_feat_info[feature_id].push_back(obs);
            ++valid_observation_count;
            if (best_obs == nullptr || (obs.first == 0 && best_obs->first != 0)) {
                best_obs = &obs;
            }
        }
        if (best_obs == nullptr) {
            continue;
        }

        const Pose& T_imu_from_cam = cam_extrinsic(best_obs->first);
        Primary primary;
        primary.feature_id = feature_id;
        primary.P = best_obs->second.block<3, 1>(2, 0);
        primary.c = T_imu_from_cam.t;
        primary.d = (T_imu_from_cam.R * best_obs->second.block<3, 1>(5, 0).normalized()).normalized();
        primaries.push_back(primary);
    }
    last_process_timing_.valid_observations = valid_observation_count;
    last_process_timing_.correspondence_ms = std::chrono::duration<double, std::milli>(
                                                  MonotonicRawClock::now() -
                                                  correspondence_begin)
                                                  .count();

    // 仅做统计，不参与任何验收门限。以求解前的有效候选观测为口径，记录双目成对
    // 地标中地面/非地面点的构成，供后续评估是否值得在此处提前做几何预筛；不要用
    // 内点统计替代它，否则无法反映被剔除前的候选几何。
    const LandmarkPlaneStats candidate_plane_stats =
        classifyLandmarksByPlane(valid_feat_info);
    const int candidate_stereo_pair_landmarks = countStereoPairLandmarks(valid_feat_info);
    // 不在这里打印：先缓存候选阶段（求解之前）的几何摘要，等这个候选算出
    // 最终 ACCEPTED/REJECTED 结果后，跟假设/内点阶段的统计一起打印
    // （见 printRelocResult() 的调用点），两个阶段的字段分开、不覆盖。
    char candidate_geometry_buf[128];
    std::snprintf(candidate_geometry_buf, sizeof(candidate_geometry_buf),
                  "{obs=%d,lm=%zu,pairs=%d,offplane_pairs=%d}", valid_observation_count,
                  valid_feat_info.size(), candidate_stereo_pair_landmarks,
                  candidate_plane_stats.off_plane_stereo_pairs);
    candidate_geometry_log_ = candidate_geometry_buf;

    // 至少四个非共线地标（按地标计数，双目同点只算一个）。
    if (primaries.size() < 4 || !hasNonCollinearWorldPoints(valid_feat_info)) {
        // 还没求解、位姿也没算，三个都用默认（不可用）值——命名成局部
        // 变量而不是就地构造临时对象，跟下面几处 REJECTED 保持同一种写法。
        const RansacGeometryStats ransac_stats;
        const RelocPoseSummary pose_summary;
        const RelocMetricsSummary metrics_summary;
        printRelocResult("REJECTED", "insufficient_or_collinear_candidates", timestamp_,
                         candidate_geometry_log_, ransac_stats, pose_summary, metrics_summary,
                         stereo_filter_log_);
        return false;
    }

    const auto hypothesis_begin = MonotonicRawClock::now();

    // 内点判定角度阈值：1 - cos(观测视线, 模型预测视线) < threshold
    const double threshold =
        1.0 - std::cos(std::atan(std::sqrt(2.0) * inlier_pixel_threshold_px_ /
                                 cameras_.at(0)->getFocalLength()));
    const auto ray_error = [](const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                              const Eigen::Vector3d& P, const Eigen::Vector3d& c,
                              const Eigen::Vector3d& d) {
        const Eigen::Vector3d v = R.transpose() * (P - t) - c;
        const double norm = v.norm();
        return norm < 1e-12 ? 2.0 : 1.0 - d.dot(v / norm);
    };

    // 地标数有限（实测最多二十余个），两两组合全部枚举，不随机采样：
    // 结果可复现，耗时实测 p95 < 1ms。
    struct Hypothesis {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
        int support = 0;
        double cost = 0.0;  // Σ min(err, thr)/thr，同支持数时越小越好
    };
    Hypothesis best;
    int model_count = 0;
    const UprightP2Options options;
    std::vector<UprightP2Solution> solutions;
    for (size_t i = 0; i < primaries.size(); ++i) {
        for (size_t j = i + 1; j < primaries.size(); ++j) {
            const Primary& a = primaries[i];
            const Primary& b = primaries[j];
            if (solveUprightP2(a.P, b.P, a.c, a.d, b.c, b.d, options, solutions) !=
                UprightP2Status::kOk) {
                continue;
            }
            for (const auto& solution : solutions) {
                ++model_count;
                Hypothesis h;
                h.R = Eigen::AngleAxisd(solution.yaw_rad, Eigen::Vector3d::UnitZ())
                          .toRotationMatrix();
                h.t = solution.t;
                for (const Primary& p : primaries) {
                    const double err = ray_error(h.R, h.t, p.P, p.c, p.d);
                    if (err < threshold) {
                        ++h.support;
                    }
                    h.cost += std::min(err, threshold) / threshold;
                }
                if (h.support > best.support ||
                    (h.support == best.support && h.cost < best.cost)) {
                    best = h;
                }
            }
        }
    }
    last_process_timing_.ransac_ms =
        std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - hypothesis_begin)
            .count();
    last_process_timing_.ransac_inliers = best.support;
    last_process_timing_.ransac_iterations = model_count;

    // 最低支持数：回放实测支持数 ≤5 的结果错解率很高（4~5 时 7/8 为错解），
    // ≥6 时约 2%。
    if (model_count == 0 || best.support < min_support_) {
        RansacGeometryStats ransac_stats;
        ransac_stats.available = true;
        ransac_stats.inliers = model_count == 0 ? 0 : best.support;
        ransac_stats.models = model_count;
        const RelocPoseSummary pose_summary;
        const RelocMetricsSummary metrics_summary;
        const std::string stage = "low_support(<" + std::to_string(min_support_) + ")";
        printRelocResult("REJECTED", stage.c_str(), timestamp_, candidate_geometry_log_,
                         ransac_stats, pose_summary, metrics_summary, stereo_filter_log_);
        return false;
    }

    // 评分只用了每个地标的主观测；内点地标的全部合法观测（含另一目）再用最佳
    // 假设按同一角度阈值逐条校验，通过的才进内点集合。
    for (const Primary& p : primaries) {
        if (ray_error(best.R, best.t, p.P, p.c, p.d) >= threshold) {
            continue;
        }
        for (const auto& feat_pair : valid_feat_info.at(p.feature_id)) {
            const Pose& T_imu_from_cam = cam_extrinsic(feat_pair.first);
            const Eigen::Vector3d d =
                (T_imu_from_cam.R * feat_pair.second.block<3, 1>(5, 0).normalized()).normalized();
            if (ray_error(best.R, best.t, feat_pair.second.block<3, 1>(2, 0), T_imu_from_cam.t,
                          d) < threshold) {
                inlier_feat_info[p.feature_id].push_back(feat_pair);
            }
        }
    }
    last_process_timing_.inlier_landmarks = static_cast<int>(inlier_feat_info.size());

    // 诊断：内点地标 ID，拼进后续 ACCEPTED/REJECTED 行。
    last_inlier_ids_log_ = "[";
    for (const auto& feature : inlier_feat_info) {
        if (last_inlier_ids_log_.size() > 1) last_inlier_ids_log_ += ",";
        last_inlier_ids_log_ += std::to_string(feature.first);
    }
    last_inlier_ids_log_ += "]";

    if (inlier_feat_info.size() < 3 || !hasNonCollinearWorldPoints(inlier_feat_info)) {
        const RansacGeometryStats ransac_stats = makeRansacGeometryStats(
            inlier_feat_info, last_process_timing_, last_inlier_ids_log_);
        // 位姿还没算出来，没有 pose 可打——命名成局部变量而不是就地构造临时
        // 对象，跟其他几处 REJECTED 保持同一种写法。
        const RelocPoseSummary pose_summary;
        const RelocMetricsSummary metrics_summary;
        printRelocResult("REJECTED", "inlier_non_collinear", timestamp_, candidate_geometry_log_,
                         ransac_stats, pose_summary, metrics_summary, stereo_filter_log_);
        inlier_feat_info.clear();
        return false;
    }

    // 最小解只由两个地标决定，远处地标下位置误差可达米级；内点交给调用方先
    // refine 再整体验收，不在这里对最小解过门。
    T_w_from_imu.R = best.R;
    T_w_from_imu.t = best.t;
    MSF_VLOG << "[Reloc] upright P2 landmarks=" << primaries.size() << " models=" << model_count
             << " support=" << best.support
             << " unique_inlier_landmarks=" << inlier_feat_info.size() << std::endl;
    return true;
}

/**
 * ceres进行优化
 */
bool ReLoc::optimizePose(const ReLocFeatInfo& reloc_feat_info, Pose& T_w_from_imu) {
    if (reloc_feat_info.empty() || !T_w_from_imu.R.allFinite() ||
        !T_w_from_imu.t.allFinite()) {
        return false;
    }

    // 旋转矩阵转成四元数
    Eigen::Quaterniond q_w_from_imu = Rotation::matrix2quaternion(T_w_from_imu.R);
    if (!q_w_from_imu.coeffs().allFinite() || q_w_from_imu.norm() < 1e-12) {
        return false;
    }
    q_w_from_imu.normalize();
    // 拿到位移
    Eigen::Vector3d& t_w_from_imu = T_w_from_imu.t;

    // 定义数组，保存状态量
    double pose_params[7] = {0};
    // 把状态量数据拷贝到数组中(前3维保存位移)
    memcpy(pose_params, t_w_from_imu.data(), 3 * sizeof(double));
    // 把四元数数据拷贝到数组中(后4维保存旋转)
    memcpy(pose_params + 3, q_w_from_imu.coeffs().data(), 4 * sizeof(double));

    // 定义优化问题
    ceres::Problem problem;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = 8;
    options.max_num_iterations = 50;

    // 添加优化变量
    ceres::Manifold* pose_manifold = new PoseManifold();
    problem.AddParameterBlock(pose_params, 7, pose_manifold);

    // 定义核函数
    ceres::HuberLoss* huber_loss = new ceres::HuberLoss(1.0);

    // 重力先验：约束 IMU z 轴与世界系 z 轴对齐（Roll/Pitch≈0），只约束倾角，
    // 不约束高度——高度没有独立于本帧求解的外部信息，仍由 checkPose()
    // 的 max_height_m 门限把关。残差为 R_w_from_imu·e_z 的 x/y 分量（≈倾角，
    // 弧度），与右扰动 PoseManifold 对应的雅可比为 -R[e_z]x。
    // 权重 kGravityPriorWeight=100 ⇒ 倾角 σ≈0.01rad≈0.57°；重投影残差按
    // 10px 归一化（见下方 point_std），二者同为无量纲单位。
    constexpr double kGravityPriorWeight = 100.0;
    class RelocGravityPriorFactor : public ceres::SizedCostFunction<2, 7> {
    public:
        explicit RelocGravityPriorFactor(double weight_rp) : weight_rp_(weight_rp) {}
        bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override {
            Eigen::Quaterniond q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
            Eigen::Vector3d z_axis = q * Eigen::Vector3d::UnitZ();
            residuals[0] = z_axis.x() * weight_rp_;
            residuals[1] = z_axis.y() * weight_rp_;
            if (jacobians && jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                Eigen::Matrix3d R = q.toRotationMatrix();
                Eigen::Matrix3d J_R = -R * Rotation::skewSymmetric(Eigen::Vector3d::UnitZ());
                J.block<2, 3>(0, 3) = J_R.topRows<2>() * weight_rp_;
            }
            return true;
        }
    private:
        double weight_rp_;
    };
    problem.AddResidualBlock(new RelocGravityPriorFactor(kGravityPriorWeight), nullptr, pose_params);

    size_t residual_count = 0;
    // 一个地标的左右目观测分别作为单目残差；只优化 calcPose() 选出的内点。
    for (const auto& feature : reloc_feat_info) {
        for (const auto& feat_pair : feature.second) {
            const int cam_id = feat_pair.first;
            if (cam_id < 0 || static_cast<size_t>(cam_id) >= cameras_.size()) {
                continue;
            }

            const auto& feat_info = feat_pair.second;
            const Eigen::Vector2d pixel_point = feat_info.block<2, 1>(0, 0);
            const Eigen::Vector3d world_point = feat_info.block<3, 1>(2, 0);
            const Eigen::Vector3d norm_point = feat_info.block<3, 1>(5, 0);
            if (!pixel_point.allFinite() || !world_point.allFinite() ||
                !norm_point.allFinite() || pixel_point.x() < 0 ||
                pixel_point.x() >= reloc_image_width_ || pixel_point.y() < 0 ||
                pixel_point.y() >= reloc_image_height_) {
                continue;
            }

            const Pose& T_cam_from_imu =
                cam_id == 0 ? camera_imu_extrinsic_params.T_cam0_from_imu
                            : camera_imu_extrinsic_params.T_cam1_from_imu;
            const double point_std = 10.0 / cameras_.at(cam_id)->getFocalLength();
            ceres::CostFunction* cost_function = new ReLocMonoReprojectionFactor(
                norm_point, world_point, T_cam_from_imu.R, T_cam_from_imu.t,
                Eigen::Vector2d(point_std, point_std));
            problem.AddResidualBlock(cost_function, huber_loss, pose_params);
            ++residual_count;
        }
    }

    if (residual_count < 4) {
        last_process_timing_.refine_residuals = static_cast<int>(residual_count);
        return false;
    }
    last_process_timing_.refine_residuals = static_cast<int>(residual_count);

    // 求解优化问题
    ceres::Solve(options, &problem, &summary);
    last_process_timing_.refine_iterations = static_cast<int>(summary.iterations.size());
    // 打印优化结果
    MSF_VLOG << "summary: " << summary.BriefReport() << std::endl;

    if (!summary.IsSolutionUsable()) {
        return false;
    }

    // 拿到优化后的位姿
    Eigen::Quaterniond optimized_q(pose_params[6], pose_params[3], pose_params[4], pose_params[5]);
    const Eigen::Vector3d optimized_t(pose_params[0], pose_params[1], pose_params[2]);
    if (!optimized_q.coeffs().allFinite() || optimized_q.norm() < 1e-12 ||
        !optimized_t.allFinite()) {
        return false;
    }
    optimized_q.normalize();
    T_w_from_imu.R = Rotation::quaternion2matrix(optimized_q);
    T_w_from_imu.t = optimized_t;

    return true;
}

/**
 * 检查位姿
 */
bool ReLoc::checkPose(const ReLocFeatInfo& reloc_feat_info, const Pose& T_w_from_imu) {
    // 第一步，检查有限性以及旋转矩阵是否合法。
    if (!T_w_from_imu.R.allFinite() || !T_w_from_imu.t.allFinite() ||
        std::abs(T_w_from_imu.R.determinant() - 1.0) > 1e-2 ||
        (T_w_from_imu.R.transpose() * T_w_from_imu.R - Eigen::Matrix3d::Identity()).norm() >
            1e-2) {
        MSF_VLOG << "[Reloc] invalid pose matrix" << std::endl;
        return false;
    }

    // 诊断统计，不参与判定：inlier 集合的地标构成/成对观测数，后面几步不管走
    // 哪个分支都要用，提前算一次。
    const LandmarkPlaneStats plane_stats = classifyLandmarksByPlane(reloc_feat_info);
    const int stereo_pair_landmarks = countStereoPairLandmarks(reloc_feat_info);

    RansacGeometryStats ransac_stats;
    ransac_stats.available = true;
    ransac_stats.models = last_process_timing_.ransac_iterations;
    ransac_stats.inlier_ids = last_inlier_ids_log_;
    ransac_stats.inliers = last_process_timing_.ransac_inliers;
    ransac_stats.inlier_landmarks = static_cast<int>(reloc_feat_info.size());
    ransac_stats.stereo_pairs = stereo_pair_landmarks;
    ransac_stats.offplane_stereo_pairs = plane_stats.off_plane_stereo_pairs;

    const Eigen::Vector3d euler_angle = Rotation::matrix2euler(T_w_from_imu.R);
    RelocPoseSummary pose_summary;
    pose_summary.available = true;
    pose_summary.x = T_w_from_imu.t.x();
    pose_summary.y = T_w_from_imu.t.y();
    pose_summary.z = T_w_from_imu.t.z();
    pose_summary.roll_deg = euler_angle[0] * R2D;
    pose_summary.pitch_deg = euler_angle[1] * R2D;
    pose_summary.yaw_deg = euler_angle[2] * R2D;

    // 声明一次、后面按阶段填充——height 门槛这一步误差还没算出来，就用它的
    // 默认（不可用）值打印，跟其他几处 REJECTED 保持同一种"命名局部变量"
    // 的写法，不再就地构造匿名临时对象。
    RelocMetricsSummary metrics_summary;

    // 第二步，根据高度判断
    if (std::abs(T_w_from_imu.t.z()) > max_height_m_) {
        char stage[48];
        std::snprintf(stage, sizeof(stage), "height(>%.2f)", max_height_m_);
        printRelocResult("REJECTED", stage, timestamp_, candidate_geometry_log_, ransac_stats,
                         pose_summary, metrics_summary, stereo_filter_log_);
        return false;
    }

    // IMU z 轴必须朝上半球（虚拟 IMU 与世界系都是前右下，正常姿态下 R·e_z≈e_z）。
    // 2 点直立解本身不会给出倒置解；保留作为 refine 后的姿态兜底——重力先验只约束
    // R·e_z 的 x/y 分量，对 z 轴整体反向同样是零残差。
    // 用点积判定，不受欧拉角 ±180° 环绕影响。
    if (T_w_from_imu.R.col(2).dot(Eigen::Vector3d::UnitZ()) <= 0.0) {
        printRelocResult("REJECTED", "imu_up_axis_inverted", timestamp_, candidate_geometry_log_,
                         ransac_stats, pose_summary, metrics_summary, stereo_filter_log_);
        return false;
    }

    // 第三步，用反投影误差判定这个结果是否正确
    int feat_num = 0;
    const double average_reprojection_error =
        computeAverageReprojectionError(reloc_feat_info, T_w_from_imu, &feat_num);
    if (average_reprojection_error < 0.0) {
        return false;
    }
    metrics_summary.available = true;
    metrics_summary.avg_reprojection_error_px = average_reprojection_error;
    metrics_summary.feat_num = feat_num;

    // 非共面（球网高度点）成对观测数门槛：总成对观测数只保证数量，不保证结构——
    // 现场数据显示全部球门附近共面点也能凑够总数，但那样解出的高度方向仍然
    // 病态；这一步专门卡"是否有足够非共面结构约束高度自由度"。<=0 表示不启用
    // （兼容还没配置这一项的现场设备）。
    if (min_offplane_stereo_pairs_ > 0 &&
        plane_stats.off_plane_stereo_pairs < min_offplane_stereo_pairs_) {
        char stage[64];
        std::snprintf(stage, sizeof(stage), "too_few_offplane_stereo_pairs(<%d)",
                      min_offplane_stereo_pairs_);
        printRelocResult("REJECTED", stage, timestamp_, candidate_geometry_log_, ransac_stats,
                         pose_summary, metrics_summary, stereo_filter_log_);
        return false;
    }

    // 初始化与运行期统一要求平均重投影误差严格小于配置门限。
    if (average_reprojection_error >= max_reprojection_error_px_) {
        char stage[64];
        std::snprintf(stage, sizeof(stage), "reprojection_error(>=%.2f)", max_reprojection_error_px_);
        printRelocResult("REJECTED", stage, timestamp_, candidate_geometry_log_, ransac_stats,
                         pose_summary, metrics_summary, stereo_filter_log_);
        return false;
    }

    return true;
}

/**
 * 计算给定位姿下这批观测的平均反投影误差（像素）。checkPose() 拿它做阈值判定；
 * process() 结束时再用同一份内点集合、最终采用的位姿算一次，供评估重定位精度。
 * 观测数不足（<4）或位姿/中间量出现非法值时返回 -1，feat_num_out 仍会写入实际
 * 统计到的观测数（非法值触发提前退出时保持初值 0，跟 checkPose() 原来遇到任何
 * 一条非法观测就整体判失败的行为一致）。
 */
double ReLoc::computeAverageReprojectionError(const ReLocFeatInfo& reloc_feat_info,
                                              const Pose& T_w_from_imu, int* feat_num_out) const {
    int feat_num = 0;
    double reprojection_error_sum = 0.0;
    bool malformed = !T_w_from_imu.R.allFinite() || !T_w_from_imu.t.allFinite();

    if (!malformed) {
        const Eigen::Matrix3d R_imu_from_w = T_w_from_imu.R.transpose();
        const Eigen::Vector3d t_imu_from_w = -R_imu_from_w * T_w_from_imu.t;

        for (const auto& feat : reloc_feat_info) {
            for (const auto& feat_pair : feat.second) {
                const int cam_id = feat_pair.first;
                if (cam_id < 0 || static_cast<size_t>(cam_id) >= cameras_.size()) {
                    malformed = true;
                    break;
                }

                const auto& feat_info = feat_pair.second;
                const Eigen::Vector2d pixel_point = feat_info.block<2, 1>(0, 0);
                if (!pixel_point.allFinite() || pixel_point.x() < 0 ||
                    pixel_point.x() >= reloc_image_width_ ||
                    pixel_point.y() < 0 || pixel_point.y() >= reloc_image_height_) {
                    continue;
                }

                const Eigen::Vector3d world_point = feat_info.block<3, 1>(2, 0);
                if (!world_point.allFinite()) {
                    malformed = true;
                    break;
                }

                const Eigen::Vector3d imu_point = R_imu_from_w * world_point + t_imu_from_w;
                const Pose& T_cam_from_imu =
                    cam_id == 0 ? camera_imu_extrinsic_params.T_cam0_from_imu
                                : camera_imu_extrinsic_params.T_cam1_from_imu;
                const Eigen::Vector3d cam_point = T_cam_from_imu.R * imu_point + T_cam_from_imu.t;
                if (!cam_point.allFinite() || cam_point.z() <= 1e-6) {
                    malformed = true;
                    break;
                }

                Eigen::Vector2d predicted_pixel_point = Eigen::Vector2d::Zero();
                cameras_.at(cam_id)->project(cam_point, predicted_pixel_point);
                if (!predicted_pixel_point.allFinite()) {
                    malformed = true;
                    break;
                }

                const double reprojection_error_norm =
                    (pixel_point - predicted_pixel_point).norm();
                if (!std::isfinite(reprojection_error_norm)) {
                    malformed = true;
                    break;
                }

                reprojection_error_sum += reprojection_error_norm;
                ++feat_num;
            }
            if (malformed) {
                break;
            }
        }
    }

    if (feat_num_out != nullptr) {
        *feat_num_out = feat_num;
    }
    if (malformed || feat_num < 4) {
        return -1.0;
    }
    return reprojection_error_sum / feat_num;
}

} // namespace ms_fusion
