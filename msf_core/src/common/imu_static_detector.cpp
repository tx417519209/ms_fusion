#include "ms_fusion/common/imu_static_detector.h"

namespace ms_fusion {

bool ImuStaticDetector::evaluateWindow(
    const Eigen::aligned_vector<ImuMeasurement>& imu_measurement_buff,
    std::vector<double>& average, ImuStaticWindowStats* stats) const {
    average.assign(6, 0.0);

    if (imu_measurement_buff.size() < MIN_SAMPLE_COUNT) {
        return false;
    }

    const double size_invert = 1.0 / static_cast<double>(imu_measurement_buff.size());
    for (const auto& imu_measurement : imu_measurement_buff) {
        average[0] += imu_measurement.wm.x();
        average[1] += imu_measurement.wm.y();
        average[2] += imu_measurement.wm.z();
        average[3] += imu_measurement.am.x();
        average[4] += imu_measurement.am.y();
        average[5] += imu_measurement.am.z();
    }
    for (double& value : average) {
        value *= size_invert;
    }

    return evaluateStdAndGravity(imu_measurement_buff, average, stats);
}

void ImuStaticDetector::pushSamples(const Eigen::aligned_vector<ImuMeasurement>& new_samples) {
    if (new_samples.empty()) {
        return;
    }

    for (const auto& sample : new_samples) {
        window_.push_back(sample);
        running_sum_[0] += sample.wm.x();
        running_sum_[1] += sample.wm.y();
        running_sum_[2] += sample.wm.z();
        running_sum_[3] += sample.am.x();
        running_sum_[4] += sample.am.y();
        running_sum_[5] += sample.am.z();
    }

    if (window_duration_s_ <= 0.0) {
        return;
    }

    // 新进旧出：裁掉时间戳早于"最新采样时间 - window_duration_s_"的旧数据，
    // 裁掉的同时从累加和里减掉对应贡献，保持 running_sum_ 和 window_ 的内容
    // 始终一致
    const double newest_timestamp = window_.back().timestamp;
    while (window_.size() > 1 &&
          newest_timestamp - window_.front().timestamp > window_duration_s_) {
        const auto& expired = window_.front();
        running_sum_[0] -= expired.wm.x();
        running_sum_[1] -= expired.wm.y();
        running_sum_[2] -= expired.wm.z();
        running_sum_[3] -= expired.am.x();
        running_sum_[4] -= expired.am.y();
        running_sum_[5] -= expired.am.z();
        window_.pop_front();
    }
}

bool ImuStaticDetector::evaluateCurrentWindow(std::vector<double>& average,
                                              ImuStaticWindowStats* stats) const {
    average.assign(6, 0.0);

    if (window_.size() < MIN_SAMPLE_COUNT) {
        return false;
    }

    const double size_invert = 1.0 / static_cast<double>(window_.size());
    for (int a = 0; a < 6; ++a) {
        average[a] = running_sum_[a] * size_invert;
    }

    return evaluateStdAndGravity(window_, average, stats);
}

bool ImuStaticDetector::update(bool window_pass) {
    if (window_pass) {
        if (consecutive_pass_count_ < thresholds_.confirm_windows) {
            ++consecutive_pass_count_;
        }
    } else {
        consecutive_pass_count_ = 0;
    }

    is_static_ = consecutive_pass_count_ >= thresholds_.confirm_windows;
    return is_static_;
}

void ImuStaticDetector::reset() {
    consecutive_pass_count_ = 0;
    is_static_ = false;
}

void ImuStaticDetector::clearWindow() {
    window_.clear();
    for (double& value : running_sum_) {
        value = 0.0;
    }
}

} // namespace ms_fusion
