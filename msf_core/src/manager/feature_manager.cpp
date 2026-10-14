#include "ms_fusion/manager/feature_manager.h"
#include "ms_fusion/common/parameters.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ms_fusion {

namespace {
// 特征历史观测次数达到这个值才算"长期跟踪"（仅用于诊断统计，不参与判定）
constexpr int kLongTrackObservationCount = 4;
}  // namespace

FeatureManager::FeatureManager(double keyframe_min_interval_s, double keyframe_min_parallax_px,
                               int keyframe_min_parallax_track_num,
                               double triangulation_max_reprojection_px,
                               double triangulation_min_angle_deg,
                               double triangulation_min_depth_baseline_ratio)
    : triangulation_max_reprojection_px_(triangulation_max_reprojection_px),
      triangulation_min_angle_deg_(triangulation_min_angle_deg),
      stereo_baseline_m_(camera_extrinsic_params.T_cam1_from_cam0.t.norm()),
      min_triangulation_depth_m_(triangulation_min_depth_baseline_ratio * stereo_baseline_m_),
      keyframe_min_interval_s_(keyframe_min_interval_s),
      keyframe_min_parallax_px_(keyframe_min_parallax_px),
      keyframe_min_parallax_track_num_(keyframe_min_parallax_track_num) {
    if (!std::isfinite(triangulation_min_depth_baseline_ratio) ||
        triangulation_min_depth_baseline_ratio <= 0.0) {
        throw std::runtime_error("triangulation_min_depth_baseline_ratio must be finite and positive");
    }
    if (!std::isfinite(stereo_baseline_m_) || stereo_baseline_m_ <= 0.0) {
        throw std::runtime_error("Triangulation requires a finite positive calibrated stereo baseline");
    }
    if (!std::isfinite(min_triangulation_depth_m_) || min_triangulation_depth_m_ <= 0.0 ||
        min_triangulation_depth_m_ >= MAX_TRIANGULATION_DISTANCE ||
        !std::isfinite(1.0 / min_triangulation_depth_m_)) {
        throw std::runtime_error("Baseline-scaled minimum depth must be in (0, 100)m with finite inverse depth");
    }
}

double FeatureManager::computeAverageParallaxPx(const TrackFeatInfo& feature_info,
                                                int* parallax_num_out) const {
    double parallax_sum = 0.0;
    int parallax_num = 0;

    for (const auto& id_pts : feature_info) {
        const auto it = landmarks_.find(id_pts.first);
        if (it == landmarks_.end()) {
            continue;  // 新特征，不参与视差统计
        }

        const auto obs_it = it->second.obs_.find(time_frame_pre_);
        if (obs_it == it->second.obs_.end()) {
            continue;  // 上一关键帧没观测到该点
        }

        const Eigen::Vector3d& pn_prev = obs_it->second.normal_point_;
        const Eigen::Vector3d& pn_curr = id_pts.second[0].second.segment<3>(2);
        const double du = pn_curr.x() - pn_prev.x();
        const double dv = pn_curr.y() - pn_prev.y();
        parallax_sum += std::sqrt(du * du + dv * dv);
        ++parallax_num;
    }

    if (parallax_num_out != nullptr) {
        *parallax_num_out = parallax_num;
    }

    if (parallax_num < keyframe_min_parallax_track_num_) {
        // 共同观测点数太少（含 0），视差估计不可靠，不视为达标——不提前接受，
        // 交给正常的时间门控兜底
        return 0.0;
    }

    return (parallax_sum / parallax_num) * focal_length;
}

bool FeatureManager::tryAddKeyframe(TimeFrameID frame_id,
                                    const TrackFeatInfo& feature_info,
                                    double td,
                                    FeatureCheckStats* stats,
                                    double matching_reference_timestamp,
                                    size_t matching_reference_count,
                                    const Eigen::Matrix3d* R_prev_from_curr) {
    // 定义标志位，表示这一帧是否是关键帧
    bool insert_image_frame = false;

    // 诊断统计：跟踪质量相关计数，不参与下面的判定，只用于对外暴露给 timing 日志
    // 观察实际分布（跟踪到的老点数/长期跟踪点数/新点数，无论走哪条判定分支都统计）
    if (stats != nullptr) {
        for (const auto& id_pts : feature_info) {
            const auto it = landmarks_.find(id_pts.first);
            if (it == landmarks_.end()) {
                ++stats->new_feature_num;
            } else {
                ++stats->last_track_num;
                if (static_cast<int>(it->second.obs_.size()) >= kLongTrackObservationCount) {
                    ++stats->long_track_num;
                }
            }
        }
    }

    // Count final frontend associations to the actual reference snapshot, not
    // supplemented points or every ID that merely remains in the landmark map.
    int valid_matches = 0;
    int valid_triangulated_matches = 0;
    if (matching_reference_count > 0 &&
        local_active_frames_.count(matching_reference_timestamp) != 0) {
        for (const auto& id_pts : feature_info) {
            const auto landmark = landmarks_.find(id_pts.first);
            if (landmark == landmarks_.end() || !landmark->second.is_valid_ ||
                id_pts.second.empty() || !id_pts.second.front().second.allFinite()) {
                continue;
            }
            const auto reference = landmark->second.obs_.find(matching_reference_timestamp);
            if (reference == landmark->second.obs_.end() ||
                !reference->second.hasValidLeftObservation()) {
                continue;
            }
            const Eigen::Vector3d pn_curr = id_pts.second.front().second.segment<3>(2);
            if (std::abs(pn_curr.z() - 1.0) > 1e-6) {
                continue;
            }
            // Frontend-validated 2D continuity survives a backend geometry reset.
            ++valid_matches;

            // 统计有效三维路标连接数：
            // 1. 路标已三角化，深度在统一的运行时深度区间内
            if (reference->second.is_outlier_ || !landmark->second.is_triangulated_ ||
                !std::isfinite(landmark->second.depth_) ||
                landmark->second.depth_ <= minTriangulationDepth() ||
                landmark->second.depth_ > MAX_TRIANGULATION_DISTANCE) {
                continue;
            }
            // 2. 锚帧仍在滑窗内，锚观测可用且未被判外点，归一化射线有效
            const TimeFrameID anchor_time = landmark->second.kf_id_;
            if (local_active_frames_.count(anchor_time) == 0) {
                continue;
            }
            const auto anchor_obs = landmark->second.obs_.find(anchor_time);
            if (anchor_obs == landmark->second.obs_.end() || anchor_obs->second.is_outlier_ ||
                !anchor_obs->second.normal_point_.allFinite() ||
                std::abs(anchor_obs->second.normal_point_.z() - 1.0) > 1e-6) {
                continue;
            }
            ++valid_triangulated_matches;
        }
    }
    const double match_ratio = matching_reference_count > 0
        ? static_cast<double>(valid_matches) / matching_reference_count : -1.0;
    if (stats != nullptr) {
        stats->valid_match_num = valid_matches;
        stats->valid_triangulated_match_num = valid_triangulated_matches;
        stats->match_ratio = match_ratio;
    }
    // In-flight frames matched to an older KF must not immediately replace a
    // newer accepted reference. Its next actual tracking result decides support.
    const bool current_reference = matching_reference_count > 0 &&
        matching_reference_timestamp == time_frame_pre_;
    const bool low_matches = current_reference &&
        valid_matches >= keyframe_min_connected_matches_ &&
        (valid_matches < keyframe_min_valid_matches_ && match_ratio < keyframe_min_match_ratio_);
    const bool low_3d_support = current_reference &&
        valid_matches >= keyframe_min_connected_matches_ &&
        valid_triangulated_matches < keyframe_min_triangulated_matches_;

    KeyframeTrigger trigger = KeyframeTrigger::kNone;
    if (local_active_frames_.empty()) {
        insert_image_frame = true;
        trigger = KeyframeTrigger::kFirst;
    } else {
        const double dt = frame_id - time_frame_pre_;
        // 视差通道的硬下限：不单独开配置项，直接取 keyframe_min_interval_s_ 的
        // 固定比例——快速运动时视差几乎每帧都能达标，如果不设一个视差绕不过去
        // 的下限，关键帧速率会被顶到接近原始帧率、压垮 backend（见
        // vision_measurement_deque_ 积压事件）。不到这个下限直接不接受，视差
        // 再大也没用，也不用算视差——省下这次计算，快速运动时这个分支会被
        // 命中很多次。
        const double hard_min_interval_s = 0.5 * keyframe_min_interval_s_;
        if (dt < hard_min_interval_s) {
            // insert_image_frame 保持 false，trigger 保持 kNone
        } else if (dt >= keyframe_min_interval_s_) {
            // 时间间隔达标，直接接受
            insert_image_frame = true;
            trigger = KeyframeTrigger::kInterval;
        } else if (low_matches && dt >= keyframe_match_min_interval_s_) {
            // Still connected: renew the reference before matches disappear.
            // This branch also obeys the half-interval gate above and the
            // compute-budget floor supplied by the optimizer.
            insert_image_frame = true;
            trigger = KeyframeTrigger::kLowMatches;
        } else if (low_3d_support && dt >= keyframe_match_min_interval_s_) {
            // 3D 有效连接数过低：在三维约束断流前提前更新参考帧
            insert_image_frame = true;
            trigger = KeyframeTrigger::kLow3DSupport;
        } else {
            // 处于硬下限和时间间隔之间：只允许旋转补偿后的剩余视差提前接受。
            // 原始平均视差仍保留为诊断，不能参与 kParallax 判定。
            int raw_parallax_num = 0;
            const double raw_parallax_px =
                computeAverageParallaxPx(feature_info, &raw_parallax_num);
            int rot_parallax_num = 0;
            double rot_parallax_px = -1.0;
            if (R_prev_from_curr != nullptr && R_prev_from_curr->allFinite()) {
                double parallax_sum = 0.0;
                for (const auto& id_pts : feature_info) {
                    const auto landmark = landmarks_.find(id_pts.first);
                    if (landmark == landmarks_.end() || id_pts.second.empty()) {
                        continue;
                    }
                    const auto previous = landmark->second.obs_.find(time_frame_pre_);
                    if (previous == landmark->second.obs_.end()) {
                        continue;
                    }
                    const Eigen::Vector3d& pn_prev = previous->second.normal_point_;
                    const Eigen::Vector3d pn_curr = id_pts.second.front().second.segment<3>(2);
                    const Eigen::Vector3d predicted = (*R_prev_from_curr) * pn_curr;
                    if (!pn_prev.allFinite() || !predicted.allFinite() ||
                        std::abs(predicted.z()) < 1e-9) {
                        continue;
                    }
                    const double du = predicted.x() / predicted.z() - pn_prev.x();
                    const double dv = predicted.y() / predicted.z() - pn_prev.y();
                    parallax_sum += std::sqrt(du * du + dv * dv);
                    ++rot_parallax_num;
                }
                if (rot_parallax_num >= keyframe_min_parallax_track_num_) {
                    rot_parallax_px =
                        parallax_sum / static_cast<double>(rot_parallax_num) * focal_length;
                } else {
                    rot_parallax_px = 0.0;
                }
            }
            if (stats != nullptr) {
                stats->parallax_track_num = raw_parallax_num;
                stats->avg_parallax_px = raw_parallax_px;
                stats->rot_comp_parallax_track_num = rot_parallax_num;
                stats->rot_comp_parallax_px = rot_parallax_px;
            }
            insert_image_frame = rot_parallax_px >= keyframe_min_parallax_px_;
            if (insert_image_frame) {
                trigger = KeyframeTrigger::kParallax;
            }
        }
    }
    if (stats != nullptr) {
        stats->trigger = trigger;
    }

    if (insert_image_frame) {
        local_active_frames_.insert(frame_id);

        for (auto& id_pts : feature_info) {
            FeatPerFrame feat_per_frame(id_pts.second[0].second, td);

            if (id_pts.second.size() == 2) {
                feat_per_frame.addRightObservation(id_pts.second[1].second);
            }

            FeatID feat_id = id_pts.first;

            if (landmarks_.find(feat_id) == landmarks_.end()) {
                // 表明这是一个新的特征点

                SFMFeat sfm_feat(feat_id, frame_id);

                sfm_feat.obs_[frame_id] = feat_per_frame;

                landmarks_[feat_id] = sfm_feat;
            } else {
                landmarks_[feat_id].obs_[frame_id] = feat_per_frame;
            }
        }

        time_frame_pre_ = frame_id;

        return true;
    }

    return false;
}

namespace {
// nth_element 就够了，不需要全排序；空时返回 -1
double medianInPlace(std::vector<double>& values) {
    if (values.empty()) {
        return -1.0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values.at(mid);
}
}  // namespace

void FeatureManager::triangulate(const TimeFrameID& frame_id, TriangulateStats* stats) {
    if (stats != nullptr) *stats = TriangulateStats{};
    // 新三角化点的深度/视差/视差角，只用来算中位数
    std::vector<double> new_depths;
    std::vector<double> new_disparities;
    std::vector<double> new_tri_angles_deg;

    for (auto& id_pts : landmarks_) {
        if (id_pts.second.is_triangulated_) {
            continue;
        }

        if (!id_pts.second.is_valid_) {
            continue;   
        }

        if (id_pts.second.obs_.find(frame_id) == id_pts.second.obs_.end()) {
            continue;
        }

        FeatPerFrame& observation = id_pts.second.obs_[frame_id];
        if (!observation.has_right_observation_) {
            continue;
        }

        if (observation.is_outlier_ || !observation.hasValidLeftObservation()) {
            continue;
        }
        if (stats != nullptr) ++stats->candidates;
        // 拿到特征点左右图的归一化坐标
        const Eigen::Vector3d& pn_left = observation.normal_point_;
        const Eigen::Vector3d& pn_right = observation.normal_point_right_;

        // 计算，得到特征点的深度
        double depth = triangulatePoint(pn_left, pn_right);

        if (std::isfinite(depth) && depth > minTriangulationDepth() &&
            depth <= MAX_TRIANGULATION_DISTANCE) {
            if (!passesTriangulationQuality(pn_left, pn_right, depth)) {
                if (stats != nullptr) ++stats->quality_rejected;
                // 当前帧双目几何质量不达标：仅标记当前帧右目观测不可用，
                // 不赋深度、保持 is_triangulated_ = false，绝不杀死左目时序轨迹（保持 is_valid_ = true）
                observation.has_right_observation_ = false;
                continue;
            }

            // 当前帧的新双目观测提供首次或重新初始化的深度。
            // 由于系统采用前向视觉因子（index_obs > index_ref），将锚帧定于提供深度的当前帧；
            // 截断当前帧之前的旧观测，避免构造逆向残差导致优化问题秩亏。
            if (id_pts.second.kf_id_ != frame_id) {
                id_pts.second.obs_.erase(id_pts.second.obs_.begin(),
                                         id_pts.second.obs_.lower_bound(frame_id));
                id_pts.second.kf_id_ = frame_id;
                if (stats != nullptr) ++stats->delayed_reanchored;
            }
            // A new depth must never reuse the world point from old geometry.
            id_pts.second.p3d_.setZero();
            id_pts.second.depth_ = depth;
            id_pts.second.stereo_initialization_frame_id_ = frame_id;

            // 赋值标志位：成功获得高置信度物理逆深度参数
            if (stats != nullptr && id_pts.second.has_been_triangulated_) ++stats->reinitialized;
            id_pts.second.has_been_triangulated_ = true;
            id_pts.second.is_triangulated_ = true;

            if (stats != nullptr) {
                new_depths.push_back(depth);
                new_disparities.push_back(
                    static_cast<double>(observation.uv_.x() - observation.uv_right_.x()));

                // 视差角：右目视线转到左目相机系后与左目视线的夹角
                const Eigen::Matrix3d& R_cam1_from_cam0 =
                    camera_extrinsic_params.T_cam1_from_cam0.R;
                const Eigen::Vector3d ray_right_in_cam0 =
                    (R_cam1_from_cam0.transpose() * pn_right).normalized();
                const Eigen::Vector3d ray_left = pn_left.normalized();
                const double cos_angle =
                    std::clamp(ray_left.dot(ray_right_in_cam0), -1.0, 1.0);
                new_tri_angles_deg.push_back(std::acos(cos_angle) * 180.0 / M_PI);
            }
        } else {
            // 双目初始化失败：仅标记该帧右目不可用，保持 is_triangulated_ = false，绝不修改 is_valid_，保住左目时序长轨迹
            observation.has_right_observation_ = false;
            continue;
        }
    }

    if (stats != nullptr) {
        stats->new_triangulated = static_cast<int>(new_depths.size());
        stats->depth_median = medianInPlace(new_depths);
        stats->disparity_median = medianInPlace(new_disparities);
        stats->triangulation_angle_median = medianInPlace(new_tri_angles_deg);
    }
}

void FeatureManager::computeRotCompensatedParallaxStats(
    TimeFrameID prev_frame_id, TimeFrameID frame_id, const Eigen::Matrix3d& R_prev_from_curr,
    RotCompensatedParallaxStats* stats) const {
    if (stats == nullptr) {
        return;
    }
    *stats = RotCompensatedParallaxStats{};

    std::vector<double> parallax_px;
    for (const auto& id_landmark : landmarks_) {
        const auto& feat = id_landmark.second;
        const auto obs_prev = feat.obs_.find(prev_frame_id);
        if (obs_prev == feat.obs_.end()) {
            continue;
        }
        const auto obs_curr = feat.obs_.find(frame_id);
        if (obs_curr == feat.obs_.end()) {
            continue;
        }

        const Eigen::Vector3d& pn_prev = obs_prev->second.normal_point_;
        const Eigen::Vector3d predicted = R_prev_from_curr * obs_curr->second.normal_point_;
        if (std::abs(predicted.z()) < 1e-9) {
            continue;
        }
        const double du = predicted.x() / predicted.z() - pn_prev.x();
        const double dv = predicted.y() / predicted.z() - pn_prev.y();
        parallax_px.push_back(std::sqrt(du * du + dv * dv) * focal_length);
    }

    stats->track_num = static_cast<int>(parallax_px.size());
    if (parallax_px.empty()) {
        return;
    }
    stats->median_px = medianInPlace(parallax_px);
}

void FeatureManager::collectKeyframePoints(TimeFrameID frame_id,
                                           const Eigen::Matrix3d& R_w_i,
                                           const Eigen::Vector3d& t_w_i,
                                           const Pose& T_imu_from_cam0,
                                           std::vector<cv::Point3f>& points_3d,
                                           std::vector<cv::Point2f>& points_2d_uv,
                                           std::vector<cv::Point2f>& points_2d_norm,
                                           std::vector<double>& point_ids) const {
    points_3d.clear();
    points_2d_uv.clear();
    points_2d_norm.clear();
    point_ids.clear();

    for (const auto& id_pts : landmarks_) {
        const SFMFeat& feat = id_pts.second;
        if (!feat.is_valid_ || !feat.is_triangulated_) {
            continue;
        }
        const auto it = feat.obs_.find(frame_id);
        if (it == feat.obs_.end()) {
            continue;
        }

        Eigen::Vector3d pw = feat.p3d_;
        if (!pw.allFinite()) {
            continue;
        }
        if (pw.norm() < 1e-6) {
            // Prefer stereo depth at the current keyframe so every KF can publish
            // long-lived landmarks (not only the birth frame before p3d_ is set).
            double depth = -1.0;
            const Eigen::Vector3d& pn = it->second.normal_point_;
            if (it->second.has_right_observation_) {
                depth = triangulatePoint(pn, it->second.normal_point_right_);
            } else if (std::abs(frame_id - feat.kf_id_) <= 1e-9 && feat.depth_ > 0.0) {
                depth = feat.depth_;
            } else {
                continue;
            }
            if (!std::isfinite(depth) || depth <= minTriangulationDepth() ||
                depth > MAX_TRIANGULATION_DISTANCE ||
                (it->second.has_right_observation_ &&
                 !passesTriangulationQuality(pn, it->second.normal_point_right_, depth))) {
                continue;
            }
            const Eigen::Vector3d pc = pn * depth;
            const Eigen::Vector3d pi = T_imu_from_cam0.R * pc + T_imu_from_cam0.t;
            pw = R_w_i * pi + t_w_i;
        }
        if (!pw.allFinite()) {
            continue;
        }

        points_3d.emplace_back(static_cast<float>(pw.x()), static_cast<float>(pw.y()),
                               static_cast<float>(pw.z()));
        points_2d_uv.emplace_back(static_cast<float>(it->second.uv_.x()),
                                  static_cast<float>(it->second.uv_.y()));
        points_2d_norm.emplace_back(static_cast<float>(it->second.normal_point_.x()),
                                    static_cast<float>(it->second.normal_point_.y()));
        point_ids.push_back(static_cast<double>(feat.feat_id_));
    }
}

bool FeatureManager::triangulateStereoObservation(const Eigen::Vector3d& left,
                                                 const Eigen::Vector3d& right,
                                                 Eigen::Vector3d& point) const {
    if (!left.allFinite() || !right.allFinite() ||
        std::abs(left.z() - 1.0) > 1e-6 || std::abs(right.z() - 1.0) > 1e-6) {
        return false;
    }
    const double depth = triangulatePoint(left, right);
    if (!passesTriangulationQuality(left, right, depth)) {
        return false;
    }
    point = left * depth;
    return true;
}

bool FeatureManager::passesTriangulationQuality(const Eigen::Vector3d& left,
                                                 const Eigen::Vector3d& right,
                                                 double depth) const {
    if (!left.allFinite() || !right.allFinite() || !std::isfinite(depth) ||
        depth <= minTriangulationDepth() || depth > MAX_TRIANGULATION_DISTANCE ||
        left.z() <= 0.0 || right.z() <= 0.0 || !std::isfinite(focal_length) || focal_length <= 0.0) {
        return false;
    }
    const auto& extrinsic = camera_extrinsic_params.T_cam1_from_cam0;
    const Eigen::Vector3d p_left = left * depth;
    const Eigen::Vector3d p_right = extrinsic.R * p_left + extrinsic.t;
    if (!p_left.allFinite() || !p_right.allFinite() || p_left.z() <= 1e-9 || p_right.z() <= 1e-9) {
        return false;
    }
    // Left projection is exact by construction (the point is on its measured
    // ray). Check the independent right measurement, in equivalent pixels.
    const double reprojection_px = focal_length *
        (p_right.head<2>() / p_right.z() - right.head<2>() / right.z()).norm();
    if (!std::isfinite(reprojection_px) || reprojection_px > triangulation_max_reprojection_px_) {
        return false;
    }
    const Eigen::Vector3d right_in_left = extrinsic.R.transpose() * right;
    const double cosine = std::clamp(left.normalized().dot(right_in_left.normalized()), -1.0, 1.0);
    const double angle_deg = std::acos(cosine) * 180.0 / std::acos(-1.0);
    // Noisy measurements can exaggerate parallax. Also require the actual
    // candidate geometry to have sufficient baseline angle (especially at infinity).
    const Eigen::Vector3d candidate_right_in_left = extrinsic.R.transpose() * p_right;
    const double candidate_cosine = std::clamp(
        p_left.normalized().dot(candidate_right_in_left.normalized()), -1.0, 1.0);
    const double candidate_angle_deg = std::acos(candidate_cosine) * 180.0 / std::acos(-1.0);
    return std::isfinite(angle_deg) && angle_deg >= triangulation_min_angle_deg_ &&
           std::isfinite(candidate_angle_deg) && candidate_angle_deg >= triangulation_min_angle_deg_;
}

double FeatureManager::triangulatePoint(const Eigen::Vector3d& pn0,
                                        const Eigen::Vector3d& pn1) const {
    if (!pn0.allFinite() || !pn1.allFinite()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // 拿到相机0到相机1的变换矩阵
    const Eigen::Matrix3d& R_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.R;
    const Eigen::Vector3d& t_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.t;

    // 把相机1的归一化坐标转换到相机0的归一化坐标
    const Eigen::Vector3d m = R_cam1_from_cam0 * pn0;

    // 构造方程的系数矩阵A
    Eigen::Vector2d A(0.0, 0.0);
    A(0) = m(0) - pn1(0) * m(2);
    A(1) = m(1) - pn1(1) * m(2);

    // 构造b
    Eigen::Vector2d b(0.0, 0.0);
    b(0) = pn1(0) * t_cam1_from_cam0(2) - t_cam1_from_cam0(0);
    b(1) = pn1(1) * t_cam1_from_cam0(2) - t_cam1_from_cam0(1);

    // Keep the measured left ray, as required by the backend's inverse-depth
    // parameterization. Minimize actual normalized right reprojection error:
    //   E(d) = ||A*d - b||^2 / (m.z()*d + t.z())^2.
    // Algebraic least squares drops this denominator and biases noisy pairs
    // toward short depth. The stationary point of E has the closed form below.
    // It is the projection onto the right epipolar line, not a free 3D midpoint.
    const Eigen::Vector2d direction = t_cam1_from_cam0.z() * A + m.z() * b;
    const double scale = direction.norm() * A.norm();
    const double denominator = direction.dot(A);
    if (!A.allFinite() || !b.allFinite() || !direction.allFinite() ||
        !std::isfinite(scale) || scale <= 1e-24 ||
        !std::isfinite(denominator) || std::abs(denominator) <= 1e-12 * scale) {
        // No finite, observable stationary depth (including an optimum at infinity).
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double depth = direction.dot(b) / denominator;
    // This is only a candidate. Both initialization and publishing must still
    // apply passesTriangulationQuality(), including actual candidate parallax.
    return std::isfinite(depth) ? depth : std::numeric_limits<double>::quiet_NaN();
}

} // namespace ms_fusion
