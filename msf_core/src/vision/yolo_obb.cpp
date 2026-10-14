#include "ms_fusion/vision/yolo_obb.h"
#include "ms_fusion/camera/create_camera.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/manager/ms_fusion_manager.h"

#include <chrono>
#include <cstdio>
#include <opencv2/imgproc.hpp>

#include <sstream>
#include <tbb/tbb.h>

namespace ms_fusion {

// 构造函数
YOLO::YOLO(const RelocConfigParams& reloc_config_params, 
           const std::vector<std::shared_ptr<GenericCamera>>& cameras)
    : cameras_(std::move(cameras)),
      publish_debug_image_(reloc_config_params.publish_debug_image),
      image_width_(reloc_config_params.resolution_width),
      image_height_(reloc_config_params.resolution_height) {
    // 初始化网络的配置参数
    yolo_config_.conf_thresh = reloc_config_params.conf_threshold;
    yolo_config_.nms_thresh = reloc_config_params.nms_threshold;
    yolo_config_.num_classes = reloc_config_params.num_classes;
    yolo_config_.max_det = reloc_config_params.max_det;

    // 打印配置参数
    MSF_CLOG << "YOLO config parameters: " << std::endl;
    MSF_CLOG << "conf_threshold: " << yolo_config_.conf_thresh << std::endl;
    MSF_CLOG << "nms_threshold: " << yolo_config_.nms_thresh << std::endl;
    MSF_CLOG << "num_classes: " << yolo_config_.num_classes << std::endl;
    MSF_CLOG << "max_det: " << yolo_config_.max_det << std::endl;

    // 拿到左右相机的焦距
    for (const auto& camera : cameras_) {
        focal_lengths_.push_back(camera->getFocalLength());

        MSF_CLOG << "focal_length: " << focal_lengths_.back() << std::endl;
    }

    // 拿到模型路径
    const std::string model_path = reloc_config_params.model_path;

    // NPU 亲和性必须在 init_model() 之前设置，init 时才会真正调用
    // AX_ENGINE_SetAffinity 落到左/右目各自的 handle 上
    yolo_infer_.set_npu_affinity(
        static_cast<AX_ENGINE_NPU_SET_T>(reloc_config_params.left_npu_affinity),
        static_cast<AX_ENGINE_NPU_SET_T>(reloc_config_params.right_npu_affinity));

    // 进行网络模型的初始化,如果初始化失败，抛出异常
    auto init_modle_res = yolo_infer_.init_model(model_path);
    if (init_modle_res != 0) {
        MSF_ELOG << "Error: failed to initialize model from " << model_path
                 << " (ret=" << init_modle_res << "). "
                 << "Check that the .axmodel exists under share/ms_fusion/models/ "
                 << "and matches reloc.resolution in the config." << std::endl;
        throw std::runtime_error("Failed to initialize model: " + model_path);
    }

    // 模型初始化成功，获取网络要求输入图片的尺寸和通道数
    const auto input_info = yolo_infer_.get_input_info();
    if (input_info.empty()) {
        MSF_ELOG << "Error: failed to get input info from model " << model_path << std::endl;
        throw std::runtime_error("Failed to get input info");

        yolo_infer_.deinit_model();
    }

    // 根据输入信息，解析出对应的图片参数
    const auto& first_input_shape = input_info.begin()->second;
    if (!parseHWFromShape(first_input_shape, input_h_, input_w_, input_c_)) {
        MSF_ELOG << "Error: failed to parse input shape from model " << model_path << std::endl;
        throw std::runtime_error("Failed to parse input shape");

        yolo_infer_.deinit_model();
        throw std::runtime_error("Failed to parse input shape");
    }

    MSF_CLOG << "input_h_: " << input_h_ << std::endl;
    MSF_CLOG << "input_w_: " << input_w_ << std::endl;
    MSF_CLOG << "input_c_: " << input_c_ << std::endl;

    if (image_width_ != input_w_ || image_height_ != input_h_) {
        std::ostringstream oss;
        oss << "reloc.resolution [" << image_width_ << ", " << image_height_
            << "] does not match model input [" << input_w_ << ", " << input_h_
            << "]. Set reloc.resolution to match the .axmodel input size.";
        yolo_infer_.deinit_model();
        throw std::runtime_error(oss.str());
    }

    // 计算输入图片所占字节
    input_bytes_ = static_cast<size_t>(input_h_) * 
                   static_cast<size_t>(input_w_) * 
                   static_cast<size_t>(input_c_);
}

// 重定位接口
void YOLO::process(const double& timestamp, const cv::Mat& image0, const cv::Mat& image1) { 
    using Clock = MonotonicRawClock;
    const auto total_begin = Clock::now();
    last_frame_timing_ = {};
    last_line_counts_ = {{0, 0}};

    // 赋值时间戳
    timestamp_ = timestamp;

    // 初始化阶段无条件捕获可视化数据（不受 publish_debug_image 配置影响）：
    // 初始化超时时要存一张画了线段/关键点的图供离线分析，纯 undistort 后的
    // 原图看不出 YOLO-OBB 到底检测/匹配成了什么样，必须能拿到这份数据——
    // 初始化阶段本来就短、reloc 频率也不高，多这一次 clone 的开销可以接受。
    const bool capture_debug_info =
        publish_debug_image_ || MsFusionManager::SnglPtr()->isInitializing();
    if (capture_debug_info) {
        const auto debug_copy_begin = Clock::now();
        debug_info_.image0 = image0.clone();
        debug_info_.image1 = image1.clone();
        last_frame_timing_.debug_copy_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - debug_copy_begin).count();
    } else {
        debug_info_ = {};
    }

    // 如果没有设置配置文件，就要设置配置文件
    if (!restore_cfg_inited_) {
        // 赋值原始图片的高和宽
        yolo_config_.restore_width = image0.cols;
        yolo_config_.restore_height = image0.rows;

        // 给模型设置配置参数
        yolo_infer_.set_config(yolo_config_);

        // 更新标志位
        restore_cfg_inited_ = true;
    }

    // 定义resize后的图片
    AX_ENGINE_IO_BUFFER_T ext_input[2]{};

    const auto preprocess_begin = Clock::now();

    // 用tbb多线程对图片进行resize并写入 NPU 输入（要求 BGR 三通道）
    // 直接传图像引用，而不是传 k 再在函数体内反查 image0/image1；k 仅用于
    // 定位 ext_input[k] 这个输出槽位和日志里标出是哪一路。
    auto resize_func = [&](const cv::Mat& src, int k) {
        cv::Mat resized_image;
        cv::resize(src, resized_image, cv::Size(input_w_, input_h_));

        cv::Mat input_bgr;
        if (resized_image.channels() == 1) {
            cv::cvtColor(resized_image, input_bgr, cv::COLOR_GRAY2BGR);
        } else if (resized_image.channels() == 3) {
            input_bgr = resized_image.isContinuous() ? resized_image : resized_image.clone();
        } else {
            MSF_ELOG << "Error: YOLO input image " << k << " has unsupported channels="
                     << resized_image.channels() << std::endl;
            throw std::runtime_error("Unsupported YOLO input channels");
        }

        int ret = AX_SYS_MemAllocCached(
            reinterpret_cast<AX_U64*>(&ext_input[k].phyAddr),
            &ext_input[k].pVirAddr,
            input_bytes_,
            AX_CMM_ALIGN_SIZE,
            reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));

        if (ret != 0 || ext_input[k].pVirAddr == nullptr) {
            MSF_ELOG << "Error: failed to allocate memory for image " << k << std::endl;
            throw std::runtime_error("Failed to allocate memory");
        }

        ext_input[k].nSize = input_bytes_;
        std::memcpy(ext_input[k].pVirAddr, input_bgr.data, input_bytes_);
        AX_SYS_MflushCache(ext_input[k].phyAddr, ext_input[k].pVirAddr, ext_input[k].nSize);
    };

    // 执行：只有左右两路，parallel_invoke 比按 range(0,2) 切分的 parallel_for 语义更直接
    tbb::parallel_invoke([&] { resize_func(image0, 0); }, [&] { resize_func(image1, 1); });

    const auto preprocess_end = Clock::now();
    last_frame_timing_.preprocess_ms =
        std::chrono::duration<double, std::milli>(preprocess_end - preprocess_begin).count();

    // 绑定输入数据，检查绑定是否成功
    const auto bind_begin = Clock::now();
    yolo_infer_.bind_external_inputs(ext_input, 2);
    last_frame_timing_.bind_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - bind_begin).count();

    const auto infer_begin = Clock::now();
    // 运行网络模型
    yolo_infer_.run_model();
    const auto infer_end = Clock::now();
    last_frame_timing_.infer_ms =
        std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();

    // 获取推理结果
    const auto output_begin = Clock::now();
    const auto& pair_dets = yolo_infer_.get_last_pair_detections();
    last_frame_timing_.output_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - output_begin).count();
    last_frame_timing_.detections_left = static_cast<int>(pair_dets.first.size());
    last_frame_timing_.detections_right = static_cast<int>(pair_dets.second.size());

    const auto post_begin = Clock::now();
    // 进行后处理
    postProcess(pair_dets);
    const auto post_end = Clock::now();
    last_frame_timing_.post_ms =
        std::chrono::duration<double, std::milli>(post_end - post_begin).count();
    last_frame_timing_.lines_left = last_line_counts_[0];
    last_frame_timing_.lines_right = last_line_counts_[1];
    last_frame_timing_.keypoints_left = static_cast<int>(feature_ids_cam0_.size());
    last_frame_timing_.keypoints_right = static_cast<int>(feature_ids_cam1_.size());

    const auto release_begin = Clock::now();
    for (int k = 0; k < 2; ++k) {
        if (ext_input[k].pVirAddr != nullptr) {
            AX_SYS_MemFree(ext_input[k].phyAddr, ext_input[k].pVirAddr);
            ext_input[k] = {};
        }
    }
    last_frame_timing_.release_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - release_begin).count();

    last_frame_timing_.total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - total_begin).count();
}

/**
 * 获取重定位特征点的信息
 * @param timestamp 时间戳
 * @param reloc_feat_info 重定位特征点的信息
 */
void YOLO::info(double& timestamp, ReLocFeatInfo& reloc_feat_info) {
    // 赋值时间戳
    timestamp = timestamp_;

    // 赋值左右相机的ID
    int cam0_id = 0, cam1_id = 1;

    // 遍历左相机的特征点
    for (size_t i = 0; i < feature_ids_cam0_.size(); ++i) {
        // 拿到特征点的ID
        int feature_id = feature_ids_cam0_[i];

        // 拿到特征点对应的世界坐标
        const auto& world_point = KEYPOINT_WORLD[feature_id];

        // 拿到特征点的像素坐标
        const auto& pixel_point = feature_pts_cam0_[i];

        if (pixel_point.x < 0 || pixel_point.x >= image_width_ || pixel_point.y < 0 ||
            pixel_point.y >= image_height_) {
            continue;
        }

        // 把像素坐标转到归一化
        Eigen::Vector3d norm_point = Eigen::Vector3d::Zero();
        cameras_[cam0_id]->unproject(Eigen::Vector2d(pixel_point.x, pixel_point.y), norm_point);

        // 定义特征点信息，并赋值
        Eigen::Matrix<double, 8, 1> feat_info = Eigen::Matrix<double, 8, 1>::Zero();
        feat_info << pixel_point.x, pixel_point.y,                       // 像素坐标
                     world_point.x, world_point.y, world_point.z,        // 世界坐标
                     norm_point.x(), norm_point.y(), norm_point.z();     // 归一化坐标

        reloc_feat_info[feature_id].push_back(std::make_pair(cam0_id, feat_info));
    }

    // 遍历右相机的特征点
    for (size_t i = 0; i < feature_ids_cam1_.size(); ++i) {
        // 拿到特征点的ID
        int feature_id = feature_ids_cam1_[i];

        // 拿到特征点对应的世界坐标
        const auto& world_point = KEYPOINT_WORLD[feature_id];

        // 拿到特征点的像素坐标
        const auto& pixel_point = feature_pts_cam1_[i];

        if (pixel_point.x < 0 || pixel_point.x >= image_width_ || pixel_point.y < 0 ||
            pixel_point.y >= image_height_) {
            continue;
        }

        // 把像素坐标转到归一化坐标
        Eigen::Vector3d norm_point = Eigen::Vector3d::Zero();
        cameras_[cam1_id]->unproject(Eigen::Vector2d(pixel_point.x, pixel_point.y), norm_point);

        // 定义特征点信息，并赋值
        Eigen::Matrix<double, 8, 1> feat_info = Eigen::Matrix<double, 8, 1>::Zero();
        feat_info << pixel_point.x, pixel_point.y,                                         // 像素坐标
                     world_point.x, world_point.y, world_point.z,                          // 世界坐标
                     norm_point.x(), norm_point.y(), norm_point.z();                       // 归一化坐标

        // 保存特征点信息
        reloc_feat_info[feature_id].push_back(std::make_pair(cam1_id, feat_info));
    }

    // 清理容器
    feature_ids_cam0_.clear();
    feature_ids_cam1_.clear();
    feature_pts_cam0_.clear();
    feature_pts_cam1_.clear();
}

void YOLO::postProcess(const std::pair<std::vector<ax_yolo_obb_infer_node::obb_detection>,
                       std::vector<ax_yolo_obb_infer_node::obb_detection>>& pair_dets) {
    feature_ids_cam0_.clear();
    feature_ids_cam1_.clear();
    feature_pts_cam0_.clear();
    feature_pts_cam1_.clear();

    // 跟 process() 开头那份判定同一条件（这里是独立成员函数，拿不到 process()
    // 里的局部变量，只能重算一次，开销可忽略）。
    const bool capture_debug_info =
        publish_debug_image_ || MsFusionManager::SnglPtr()->isInitializing();

    // 用TBB多线程对pair_dets进行后处理
    // 直接传该路的检测结果引用，而不是传 k 再在函数体内反查 pair_dets.first/second；
    // k 仅用于往 last_line_counts_/feature_ids_camX_ 这些按左右分开的输出容器里写。
    auto post_process_func = [&](const std::vector<ax_yolo_obb_infer_node::obb_detection>& dets,
                                  int k) {

        // 现在把有向包围盒转成线
        std::vector<std::vector<ObbLine>> obb_to_lines = std::vector<std::vector<ObbLine>>(NUM_LINES_CLASS);
        obbToLine(dets, obb_to_lines);
        int line_count = 0;
            for (const auto& class_lines : obb_to_lines) {
                line_count += static_cast<int>(class_lines.size());
        }
        last_line_counts_[static_cast<size_t>(k)] = line_count;

        // 求出了线的起始坐标，寻找NET点和POST点
        std::vector<int> feature_ids;
        std::vector<cv::Point2f> feature_pts;
        findNetAndPostPoints(obb_to_lines, feature_ids, feature_pts);

        // 保存这些点
        if (k == 0) {
            // 这是左图(保存特征点坐标和ID)
            feature_ids_cam0_.insert(feature_ids_cam0_.end(), feature_ids.begin(), feature_ids.end());
            feature_pts_cam0_.insert(feature_pts_cam0_.end(), feature_pts.begin(), feature_pts.end());
        } else if (k == 1) {
            // 这是右图
            feature_ids_cam1_.insert(feature_ids_cam1_.end(), feature_ids.begin(), feature_ids.end());
            feature_pts_cam1_.insert(feature_pts_cam1_.end(), feature_pts.begin(), feature_pts.end());
        }

        // 通过线的点，求线的一般表达时
        std::vector<Line> lines = std::vector<Line>(NUM_LINES_CLASS);
        segmentLinesToLines(obb_to_lines, lines);

        // 开始计算线之间的交点
        std::vector<int> intersection_feature_ids = std::vector<int>();
        std::vector<cv::Point2f> intersection_feature_pts = std::vector<cv::Point2f>();
        linesIntersections(lines, intersection_feature_ids, intersection_feature_pts);

        // 保存这些点
        if (k == 0) {
            feature_ids_cam0_.insert(feature_ids_cam0_.end(), intersection_feature_ids.begin(),
                                     intersection_feature_ids.end());
            feature_pts_cam0_.insert(feature_pts_cam0_.end(), intersection_feature_pts.begin(),
                                     intersection_feature_pts.end());
        } else if (k == 1) {
            feature_ids_cam1_.insert(feature_ids_cam1_.end(), intersection_feature_ids.begin(),
                                     intersection_feature_ids.end());
            feature_pts_cam1_.insert(feature_pts_cam1_.end(), intersection_feature_pts.begin(),
                                     intersection_feature_pts.end());
        }

        if (capture_debug_info && k == 0) {
            debug_info_.obb_lines_cam0 = obb_to_lines;
            debug_info_.feature_ids_cam0 = feature_ids_cam0_;
            debug_info_.feature_pts_cam0 = feature_pts_cam0_;
        } else if (capture_debug_info && k == 1) {
            debug_info_.obb_lines_cam1 = obb_to_lines;
            debug_info_.feature_ids_cam1 = feature_ids_cam1_;
            debug_info_.feature_pts_cam1 = feature_pts_cam1_;
        }
    };

    // 执行函数：左右两路固定并行，parallel_invoke 语义比 range(0,2) 的 parallel_for 更直接
    tbb::parallel_invoke([&] { post_process_func(pair_dets.first, 0); },
                         [&] { post_process_func(pair_dets.second, 1); });
}

/**
 * obb框转到线
 * @param obb框
 * @param 线
 */
void YOLO::obbToLine(const std::vector<ax_yolo_obb_infer_node::obb_detection>& dets,
                     std::vector<std::vector<ObbLine>>& obb_to_lines) {
    // 遍历所有的检测框
    for (const auto& det : dets) {
        // 拿到有向包围盒的中心点坐标
        float cx = det.cx;
        float cy = det.cy;

        // 拿到有向包围盒的宽度和高度
        float w = det.w;
        float h = det.h;
        
        // 拿到有向包围盒的旋转角度
        float angle = det.angle;

        // 计算三角函数的值
        float sin_theta = std::sin(angle);
        float cos_theta = std::cos(angle);

        float wx = w / 2.f * cos_theta;
        float wy = w / 2.f * sin_theta;
        float hx = h / 2.f * sin_theta * (-1.0);
        float hy = h / 2.f * cos_theta;

        // 计算四个顶点的坐标
        cv::Point2f p0{cx - wx - hx, cy - wy - hy};  // 左上
        cv::Point2f p1{cx + wx - hx, cy + wy - hy};  // 右上
        cv::Point2f p2{cx + wx + hx, cy + wy + hy};  // 右下
        cv::Point2f p3{cx - wx + hx, cy - wy + hy};  // 左下

        // 定义线段的两个端点的坐标
        cv::Point2f segment_point0 = cv::Point2f(), segment_point1 = cv::Point2f();

        //网上的线和地面的线分开处理（class 10~14 都是网线，需与 lineIdToNetFeatureId() 的范围一致）
        if (det.class_id == 0 || det.class_id == 2 || det.class_id >= 10) {
            ////// 这个是网上的线
            // 通过01和23可以求出两个中点坐标
            cv::Point2f point_01 = cv::Point2f((p0.x + p1.x) / 2.0f, (p0.y + p1.y) / 2.0f);
            cv::Point2f point_23 = cv::Point2f((p2.x + p3.x) / 2.0f, (p2.y + p3.y) / 2.0f);

            // 通过12和30可以求出两个中点坐标
            cv::Point2f point_12 = cv::Point2f((p1.x + p2.x) / 2.0f, (p1.y + p2.y) / 2.0f);
            cv::Point2f point_30 = cv::Point2f((p3.x + p0.x) / 2.0f, (p3.y + p0.y) / 2.0f);

            // y变化大的点为直线的起始坐标和终点坐标
            if (std::abs(point_01.y - point_23.y) > std::abs(point_12.y - point_30.y)) {
                segment_point0 = point_01;
                segment_point1 = point_23;
            } else {
                segment_point0 = point_12;
                segment_point1 = point_30;
            }
        } else {
            ////// 这个是地面的线
            // 对比高和框，确定哪个是短轴，哪个是长轴
            float long_edge = std::max(h, w);
            float short_edge = std::min(h, w);

            // 这里说明框的宽和高比较相近，且是七号线
            if (short_edge / long_edge > 0.5 && det.class_id == 7) { 
                // 通过01和23可以求出两个中点坐标
                cv::Point2f point_01 = cv::Point2f((p0.x + p1.x) / 2.0f, (p0.y + p1.y) / 2.0f);
                cv::Point2f point_23 = cv::Point2f((p2.x + p3.x) / 2.0f, (p2.y + p3.y) / 2.0f);

                // 通过12和30可以求出两个中点坐标
                cv::Point2f point_12 = cv::Point2f((p1.x + p2.x) / 2.0f, (p1.y + p2.y) / 2.0f);
                cv::Point2f point_30 = cv::Point2f((p3.x + p0.x) / 2.0f, (p3.y + p0.y) / 2.0f);

                // y变化大的点为直线的起始坐标和终点坐标
                if (std::abs(point_01.y - point_23.y) > std::abs(point_12.y - point_30.y)) {
                    segment_point0 = point_01;
                    segment_point1 = point_23;
                } else {
                    segment_point0 = point_12;
                    segment_point1 = point_30;
                }
            } else {
                // 计算四条边的长度
                float edge0 = std::sqrt((p1.x - p0.x) * (p1.x - p0.x) + (p1.y - p0.y) * (p1.y - p0.y));
                float edge1 = std::sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
                float edge2 = std::sqrt((p3.x - p2.x) * (p3.x - p2.x) + (p3.y - p2.y) * (p3.y - p2.y));
                float edge3 = std::sqrt((p0.x - p3.x) * (p0.x - p3.x) + (p0.y - p3.y) * (p0.y - p3.y));

                if (edge0 + edge2 >= edge1 + edge3) {
                    segment_point0 = cv::Point2f((p0.x + p3.x) / 2.0f, (p0.y + p3.y) / 2.0f);
                    segment_point1 = cv::Point2f((p2.x + p1.x) / 2.0f, (p2.y + p1.y) / 2.0f);
                } else {
                    segment_point0 = cv::Point2f((p1.x + p0.x) / 2.0f, (p1.y + p0.y) / 2.0f);
                    segment_point1 = cv::Point2f((p3.x + p2.x) / 2.0f, (p3.y + p2.y) / 2.0f);
                }
            }
        }

        // 求这条线的长度
        float line_length = std::sqrt((segment_point1.x - segment_point0.x) * (segment_point1.x - segment_point0.x) + 
                                      (segment_point1.y - segment_point0.y) * (segment_point1.y - segment_point0.y));

        // 线太短，丢弃这条线
        if (line_length < 1.0) {
            continue;
        }

        // 定义线段的起始坐标和结束坐标
        cv::Point2f start_point = cv::Point2f(), end_point = cv::Point2f();

        //对于线段0,2或这大于等于5的线，y小的坐标为起始坐标，对于其他线，x小的坐标为起始坐标
        if (det.class_id == 0 || det.class_id == 2 || det.class_id >= 5) {
            if (segment_point0.y < segment_point1.y) {
                start_point = segment_point0;
                end_point = segment_point1;
            } else {
                start_point = segment_point1;
                end_point = segment_point0;
            }
        } else {
            if (segment_point0.x < segment_point1.x) {
                start_point = segment_point0;
                end_point = segment_point1;
            } else {
                start_point = segment_point1;
                end_point = segment_point0;
            }
        }

        // 这条线满足要求,构造线的结构体
        ObbLine line;
        line.score = det.score;
        line.class_id = det.class_id;
        line.start_point = start_point;
        line.end_point = end_point;

        // 将线添加到obb_to_lines中
        obb_to_lines[det.class_id].push_back(line);
    }
}

/**
 * 寻找NET点和POST点
 */
void YOLO::findNetAndPostPoints(const std::vector<std::vector<ObbLine>>& obb_to_lines,
                                std::vector<int>& feature_ids,
                                std::vector<cv::Point2f>& feature_pixels) {
    // 遍历每一个类线段
    for (const auto& class_lines : obb_to_lines) {
        // 如果没有这一类的线，不进行处理
        if (class_lines.empty()) {
            continue;
        }

        // 拿到这条特征点的ID
        const auto& line_id = class_lines[0].class_id;

        // 第一步判断是否是NET线
        auto feature_id = lineIdToNetFeatureId(line_id);
        if (feature_id != -1) {
            // 表明这条线是NET线，把特征点的ID和像素坐标添加到容器中
            feature_ids.push_back(feature_id);

            //保存特征点的像素坐标
            feature_pixels.push_back(class_lines[0].start_point);

            continue;
        }

        // 第二步判断是否是POST线
        auto post_feature_ids = lineIdToPostFeatureIds(line_id);
        if (!post_feature_ids.empty()) {
            // 表明这条线是POST线，POST线对应两个坐标点
            feature_ids.insert(feature_ids.end(), post_feature_ids.begin(), post_feature_ids.end());

            //保存特征点的像素坐标
            feature_pixels.push_back(class_lines[0].end_point);
            feature_pixels.push_back(class_lines[0].start_point);

            continue;
        }
    }
}

/**
 * 通过坐标求线的一般表达式
 */
void YOLO::segmentLinesToLines(const std::vector<std::vector<ObbLine>>& obb_to_lines, 
                               std::vector<Line>& lines) {
    // 按照线的种类，遍历线
    for (int i = 0; i < NUM_LINES_CLASS; i++) {
        // 拿到这一类的线
        const auto& class_lines = obb_to_lines.at(i);

        // 如果没有这一类的线，不进行处理
        if (class_lines.empty()) {
            continue;
        } else if (class_lines.size() == 1) {
            // 拿到这条线
            const auto& line = class_lines.at(0);

            // 拿到线段的起始坐标和结束坐标
            const auto& p1 = line.start_point;
            const auto& p2 = line.end_point;

            // 求线的一般表达式
            cv::Vec3d line_params = line2dFromSegment(p1, p2);

            // 保存线的一般表达式
            lines.at(i).a = line_params[0];
            lines.at(i).b = line_params[1];
            lines.at(i).c = line_params[2];

            // 赋值线的ID号
            lines.at(i).class_id = i;

            // std::cout << "i - class_id: " << i << " - " << class_lines.at(0).class_id << std::endl;

        } else {
            // 打印有多条线参与拟合
            // std::cout << "有多条线参与拟合, i = " << i << ", class_lines.size() = " << class_lines.size() << std::endl;
            // std::cout << "class id: " << class_lines.at(0).class_id << std::endl;

            // 有多条线段，用拟合的方法求线的一般表达式
            // 拿到这条线的所有点的坐标
            std::vector<cv::Point2f> points;
            for (const auto& line : class_lines) {
                points.push_back(line.start_point);
                points.push_back(line.end_point);
            }

            // 定义容器保存点的均值
            cv::Point2f mean(0.0f, 0.0f);
            for(const auto& pt : points) {
                mean.x += pt.x;
                mean.y += pt.y;
            }

            mean.x /= static_cast<float>(points.size());
            mean.y /= static_cast<float>(points.size());

            // 中心化点
            cv::Mat C(points.size(), 2, CV_32F);
            for(size_t j = 0; j < points.size(); j++) {
                C.at<float>(j, 0) = points[j].x - mean.x;
                C.at<float>(j, 1) = points[j].y - mean.y;
            }

            // 做svd分解
            cv::SVD svd(C, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
            cv::Mat Vt = svd.vt; 

            // 获取最后一行
            float a = Vt.at<float>(Vt.rows - 1, 0);
            float b = Vt.at<float>(Vt.rows - 1, 1);
            
            float n = std::sqrt(a * a + b * b);
            if(n < 1e-12f) {
                continue;
            }

            // 返回归一化结果
            a /= n;    
            b /= n;

            // 求c
            float c = -(a * mean.x + b * mean.y);

            float norm_ab = std::sqrt(a * a + b * b);
            if(norm_ab < 1e-12f) {
                continue;
            }

            // 参数归一化
            a /= norm_ab;
            b /= norm_ab;    
            c /= norm_ab;

            // 保存线的一般表达式
            lines.at(i).a = a;
            lines.at(i).b = b;
            lines.at(i).c = c;

            // 赋值线的ID号
            lines.at(i).class_id = i;
        }
    }
}

/**
 * 通过线的起始坐标和结束坐标，求线的一般表达式
 */
cv::Vec3d YOLO::line2dFromSegment(const cv::Point2f &p1, const cv::Point2f &p2) {
    // 求三个参数
    double a = static_cast<double>(p1.y - p2.y);
    double b = static_cast<double>(p2.x - p1.x);
    double c = static_cast<double>(p1.x * p2.y - p2.x * p1.y);

    // 归一化前两个分量
    double norm = std::sqrt(a * a + b * b);

    // 如果norm太小
    if(norm < 1e-10) {
        return cv::Vec3d(a, b, c);
    }

    // 返回归一化结果
    return cv::Vec3d(a / norm, b / norm, c / norm);
}

/**
 * 根据所有线的一般表达式求所有线的交点
 */
void YOLO::linesIntersections(const std::vector<Line>& lines,  std::vector<int>& feature_ids, 
                              std::vector<cv::Point2f>& feature_pts) {
    // 遍历每一个交点，判断是否有对应的线
    for (int i = 0; i < NUM_INTERSECTION_MAPPINGS; i++) {
        // 拿到i对应的交点的信息
        const auto& mapping = INTERSECTION_LINES2KEYPOINTS[i];

        // 拿到第一条线的ID
        int cat_a = mapping.cat_a;
        // 拿到第二条线的ID
        int cat_b = mapping.cat_b;
        // 拿到特征点的索引
        int kp_idx = mapping.kp_idx;

        // 打印这三个信息
        // std::cout << "cat_a - cat_b - kp_idx: " << cat_a << " - " << cat_b << " - " << kp_idx << std::endl;

        // 检查两条线是否存在
        float norm_a = std::sqrt(lines.at(cat_a).a * lines.at(cat_a).a + lines.at(cat_a).b * lines.at(cat_a).b);
        float norm_b = std::sqrt(lines.at(cat_b).a * lines.at(cat_b).a + lines.at(cat_b).b * lines.at(cat_b).b);

        bool line_a_valid = (norm_a > 0.1f);  // 阈值检查，确保直线被拟合
        bool line_b_valid = (norm_b > 0.1f);

        if(!line_a_valid || !line_b_valid) {
            // 打印没有这两条线
            // std::cout << "没有这两条线, cat_a = " << cat_a << ", cat_b = " << cat_b << std::endl;
            continue;
        }

        // 计算交点
        cv::Point2f uv = lineIntersection(lines.at(cat_a), lines.at(cat_b));

        // 检查交点是否有效
        if(std::isnan(uv.x) || std::isnan(uv.y) || !std::isfinite(uv.x) || !std::isfinite(uv.y)) {
            continue;
        }

        // 保存像素坐标和特做点对应的ID
        feature_pts.push_back(uv);
        feature_ids.push_back(kp_idx);
    }
}

/**
 * 给两条线的表达，求线的交点
 */
cv::Point2f YOLO::lineIntersection(const Line& line1, const Line& line2) {
    // 拿到线段1的参数
    const auto& a1 = line1.a;
    const auto& b1 = line1.b;
    const auto& c1 = line1.c;
    
    // 拿到线段2的参数
    const auto& a2 = line2.a;
    const auto& b2 = line2.b;
    const auto& c2 = line2.c;

    // 求两条直线的叉乘向量
    float p_x = b1 * c2 - b2 * c1;
    float p_y = a2 * c1 - a1 * c2;
    float p_z = a1 * b2 - a2 * b1;

    // 判断直线是否平行
    if(std::abs(p_z) < 1e-12f) {
        // 打印交点不合法
        MSF_VLOG << "交点不合法, p_z = " << p_z << std::endl;

        return cv::Point2f(
            std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    }

    // 转换坐标
    return cv::Point2f(p_x / p_z, p_y / p_z);
}

/**
 * 线的ID与NET特征点的ID映射
 */
int YOLO::lineIdToNetFeatureId(const int& line_id) {
    switch (line_id) {
        case 10:
            return 15;
        case 11:
            return 16;
        case 12:
            return 17;
        case 13:
            return 18;
        case 14:
            return 19;
        default:
            return -1;
    }
}

/**
 * 线的ID映射到POST特征点的ID
 */
std::vector<int> YOLO::lineIdToPostFeatureIds(const int& line_id) {
    switch (line_id) {
        case 0:
            return {12, 14};
        case 2:
            return {13, 20};
        default:
            return {};
    }
}

/**
 * 功能函数
 */
bool YOLO::parseHWFromShape(const std::vector<int>& shape, int& h, int& w, int& c) {
    if (shape.size() < 3) {
        return false;
    }

    if (shape.size() == 4) {
        // 优先判断 NCHW
        if (shape[1] == 1 || shape[1] == 3) {
            c = shape[1];
            h = shape[2];
            w = shape[3];
            return true;
        }
        // 否则按 NHWC
        h = shape[1];
        w = shape[2];
        c = shape[3];
        MSF_VLOG << "NCHW shape: " << h << " " << w << " " << c << std::endl;

        return true;
    }
    

    h = shape[0];
    w = shape[1];
    c = shape[2];
    MSF_VLOG << "HWC shape: " << h << " " << w << " " << c << std::endl;
    
    return true;
}

} // naespace ms_fusion
