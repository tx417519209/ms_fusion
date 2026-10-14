#ifndef MS_FUSION_COMMON_IMU_STATIC_DETECTOR_H
#define MS_FUSION_COMMON_IMU_STATIC_DETECTOR_H

#include "ms_fusion/common/measurement.h"

#include <algorithm>
#include <cmath>

namespace ms_fusion {

// IMU 静止判据的阈值和防抖窗口数。gyro 三轴共用一个阈值；acc 的 x/y（侧向）和
// z（重力轴）分开，因为 z 轴叠着重力，噪声/残余振动天然比水平轴大。
struct ImuStaticThresholds {
    double gyr_std_threshold = 0.01;      // rad/s，陀螺三轴标准差上限
    double acc_xy_std_threshold = 0.1;    // m/s^2，加表 x/y 标准差上限
    double acc_z_std_threshold = 0.2;     // m/s^2，加表 z 标准差上限
    double gravity_tolerance = 0.1;       // m/s^2，比力模长与当地重力的允许偏差
    // 陀螺三轴均值（角速度）模长上限，rad/s。跟 gyr_std_threshold 是两个独立判据：
    // std 只看窗口内抖动，对匀速/平滑旋转不敏感（见 evaluateStdAndGravity 里的
    // 说明），这条直接卡角速度均值模长，兜住"转得很稳、没有明显抖动"的场景。
    // 注意这里没有减掉陀螺零偏——零偏本身也会贡献到均值里，阈值定得比典型零偏
    // 量级还小会导致零偏偏大的 IMU 即使真正静止也永远通不过，所以默认值留了
    // 明显高于典型零偏的余量，只用来卡肉眼可见的旋转，抓不住比零偏大不了多少
    // 的极慢旋转。
    double gyr_mean_norm_threshold = 0.03;  // rad/s，约 1.7 deg/s
    int confirm_windows = 5;              // 连续通过多少个窗口才确认静止（3~5）

    // 标准差计算前先做非重叠滑动平均（块平均）的块大小，单位是采样点数，1 表示
    // 不平滑（等价于原始逐样本标准差）。板子上如果有高频振动/电磁噪声源（比如
    // IMU 旁边的芯片），逐样本标准差会被这类高频成分主导、怎么放阈值都通不过；
    // 块平均能压掉这类高频噪声，同时基本不影响低频/持续性的真实运动信号。
    int std_smoothing_window = 1;
};

// 单次窗口判据的中间量，供调用方打印诊断信息用。
struct ImuStaticWindowStats {
    double std_gyr[3] = {0.0, 0.0, 0.0};
    double std_acc[3] = {0.0, 0.0, 0.0};
    double specific_force_norm = 0.0;
    double gyr_mean_norm = 0.0;  // 陀螺三轴均值（未减零偏）的模长，rad/s
};

// 静止检测：用一段 IMU 窗口的均值/标准差判定"这一窗口是否静止"，再要求连续
// confirm_windows 个窗口都通过才把 isStatic() 置真，避免单次窗口蹭到噪声低谷就
// 误判。提供两种用法：
//   - evaluateWindow()：无状态，调用方自己攒好一段数据传进来算一次。Initializer
//     用这个（它自己按视觉帧边界攒窗口，不需要跨帧维护）。
//   - pushSamples() + evaluateCurrentWindow()：有状态，内部维护一个按时长（不是
//     按个数）裁剪的滑动窗口，新数据从后面进、超出窗口时长的旧数据从前面出，
//     均值/累加和随 push 增量维护。MsFusionManager 运行时持续判定"当前是否
//     静止"用这个——单个关键帧间隔（~0.15~0.2s）太短，磨不平板子上的高频振动/
//     电磁噪声，需要攒够足够时长（构造时传入 window_duration_s）才有意义。
class ImuStaticDetector {
public:
    ImuStaticDetector() = default;

    // window_duration_s: 内部滑窗保留的时长（s），只影响 pushSamples/
    // evaluateCurrentWindow 这条有状态的路径；<=0 表示不使用内部滑窗（只用
    // evaluateWindow 这条无状态路径，比如 Initializer）。
    ImuStaticDetector(const ImuStaticThresholds& thresholds, double gravity,
                      double window_duration_s = 0.0)
        : thresholds_(thresholds), gravity_(gravity), window_duration_s_(window_duration_s) {}

    // 无状态：给一批 IMU 数据，直接算这批数据的均值/标准差，不碰内部滑窗。
    // average 无论静止与否都会填好（陀螺均值只有在真正静止时才等于零偏）。
    // stats 非空时回填标准差/比力模长，供调用方打印诊断日志。
    bool evaluateWindow(const Eigen::aligned_vector<ImuMeasurement>& imu_measurement_buff,
                        std::vector<double>& average,
                        ImuStaticWindowStats* stats = nullptr) const;

    // 有状态：往内部滑窗追加新采样，自动裁掉时间戳早于
    // "最新采样时间 - window_duration_s" 的旧数据；均值的累加和随 push/裁剪
    // 增量维护（O(1) 摊还），不用每次全量重算。
    void pushSamples(const Eigen::aligned_vector<ImuMeasurement>& new_samples);

    // 对当前内部滑窗算标准差/比力模长、判定这一窗口是否静止（不含防抖）。均值
    // 是 pushSamples 增量维护好的，这里直接读；标准差每次照整窗重算——块平均后
    // 只剩几十个块，这一步开销可以忽略，不值得为了省掉它把块方差也做成增量、
    // 徒增复杂度。
    bool evaluateCurrentWindow(std::vector<double>& average,
                               ImuStaticWindowStats* stats = nullptr) const;

    // 用最新一次窗口判定结果推进防抖计数器：通过则计数 +1（封顶在
    // confirm_windows），失败立刻清零。返回防抖后的静止状态。
    bool update(bool window_pass);

    bool isStatic() const { return is_static_; }
    int consecutivePassCount() const { return consecutive_pass_count_; }
    const ImuStaticThresholds& thresholds() const { return thresholds_; }

    // 内部滑窗当前的采样数/时间跨度（s），供调用方打印诊断日志
    size_t windowSampleCount() const { return window_.size(); }
    double windowSpanSeconds() const {
        return window_.size() < 2 ? 0.0 : window_.back().timestamp - window_.front().timestamp;
    }

    // 重置防抖计数（比如重定位/大幅运动后想强制重新累计）
    void reset();

    // 清空内部滑窗（比如系统重启/重新初始化时）
    void clearWindow();

private:
    static constexpr size_t MIN_SAMPLE_COUNT = 5;

    // evaluateWindow/evaluateCurrentWindow 共用的核心计算：average 已经算好，
    // 这里只负责块平均标准差 + 比力模长判定 + 回填 stats。buff 只要求支持
    // size()/operator[]，vector 和 deque 都满足。
    template <typename Container>
    bool evaluateStdAndGravity(const Container& buff, const std::vector<double>& average,
                               ImuStaticWindowStats* stats) const {
        const size_t block_size =
            std::max<size_t>(1, static_cast<size_t>(thresholds_.std_smoothing_window));
        // 块平均至少要凑够 2 个完整块才有意义：只有 1 个块时，那个块的均值就
        // 等于整体均值，标准差会退化成 0，变成不管数据实际抖不抖都能"通过"的
        // 漏洞。凑不够就当这一窗口没通过，而不是让它白捡一次。
        if (buff.size() < 2 * block_size) {
            return false;
        }

        const size_t sample_count = buff.size();
        const size_t block_count = std::max<size_t>(1, sample_count / block_size);

        double std_dev[6] = {0, 0, 0, 0, 0, 0};
        for (size_t b = 0; b < block_count; ++b) {
            const size_t begin = b * block_size;
            const size_t block_end = (b + 1 == block_count) ? sample_count : begin + block_size;

            double block_sum[6] = {0, 0, 0, 0, 0, 0};
            for (size_t i = begin; i < block_end; ++i) {
                const auto& m = buff[i];
                block_sum[0] += m.wm.x();
                block_sum[1] += m.wm.y();
                block_sum[2] += m.wm.z();
                block_sum[3] += m.am.x();
                block_sum[4] += m.am.y();
                block_sum[5] += m.am.z();
            }
            const double block_invert = 1.0 / static_cast<double>(block_end - begin);
            for (int a = 0; a < 6; ++a) {
                const double block_avg = block_sum[a] * block_invert;
                std_dev[a] += (block_avg - average[a]) * (block_avg - average[a]);
            }
        }
        const double block_count_invert = 1.0 / static_cast<double>(block_count);
        for (double& value : std_dev) {
            value = std::sqrt(value * block_count_invert);
        }

        const bool std_ok = std_dev[0] <= thresholds_.gyr_std_threshold &&
                            std_dev[1] <= thresholds_.gyr_std_threshold &&
                            std_dev[2] <= thresholds_.gyr_std_threshold &&
                            std_dev[3] <= thresholds_.acc_xy_std_threshold &&
                            std_dev[4] <= thresholds_.acc_xy_std_threshold &&
                            std_dev[5] <= thresholds_.acc_z_std_threshold;

        // 陀螺均值模长判据：std_ok 只看窗口内抖动，匀速/平滑旋转的 std 依然很小，
        // 会被误判成静止；这里直接卡角速度均值本身，兜住这类场景。
        const double gyr_mean_norm =
            Eigen::Vector3d(average[0], average[1], average[2]).norm();
        const bool gyr_mean_ok = gyr_mean_norm <= thresholds_.gyr_mean_norm_threshold;

        // 比力均值的模长必须等于当地重力。标准差判据只看窗口内抖动，匀速运动
        // 能蒙混过关；模长偏差则直接说明加表除了重力还测到了别的加速度。
        const double specific_force_norm =
            Eigen::Vector3d(average[3], average[4], average[5]).norm();
        const bool gravity_ok =
            std::fabs(specific_force_norm - gravity_) <= thresholds_.gravity_tolerance;

        if (stats != nullptr) {
            stats->std_gyr[0] = std_dev[0];
            stats->std_gyr[1] = std_dev[1];
            stats->std_gyr[2] = std_dev[2];
            stats->std_acc[0] = std_dev[3];
            stats->std_acc[1] = std_dev[4];
            stats->std_acc[2] = std_dev[5];
            stats->specific_force_norm = specific_force_norm;
            stats->gyr_mean_norm = gyr_mean_norm;
        }

        return std_ok && gyr_mean_ok && gravity_ok;
    }

    ImuStaticThresholds thresholds_;
    double gravity_ = 9.81;
    double window_duration_s_ = 0.0;

    // 内部滑窗（有状态路径专用）+ 均值的累加和，随 pushSamples 增量维护
    Eigen::aligned_deque<ImuMeasurement> window_;
    double running_sum_[6] = {0, 0, 0, 0, 0, 0};

    int consecutive_pass_count_ = 0;
    bool is_static_ = false;
};

} // namespace ms_fusion

#endif // MS_FUSION_COMMON_IMU_STATIC_DETECTOR_H
