#ifndef MEASUREMENT_MANAGER_H
#define MEASUREMENT_MANAGER_H

#include "eigen_utils.hpp"
#include "config_params.h"
#include "ms_fusion_message.h"
#include "ms_fusion/common/macros.h"
#include "ms_fusion/common/measurement.h"

#include <Eigen/Geometry>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <queue>
#include <thread>
#include <utility>

namespace ms_fusion {

// 声明类
class Tracking;
class GenericCamera;

class MeasurementManager {
    // 单例模式
    SINGLETON_DECLARE(MeasurementManager)

    // 定义智能锁
    using Lock = std::unique_lock<std::mutex>;

    // EIGEN数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    // 初始化函数
    void initMeasurementManager(const MsFusionConfigParams& config_params);

    // 停止并 join 本类的工作线程。进程退出前、ROS 层发布回调绑定的对象析构前调用。
    void stop();

    // 从外界接收IMU数据的接口
    void addNewImuMessage(const ImuMessage& imu_message);

    // 从外界接收ODOM数据的接口
    void addNewOdomMessage(const OdomMessage& odom_message);

    // 从外界接收相机数据的接口
    void addNewImageMessage(const ImageMessage& image_message);

    // 从外界接收GNSS数据的接口
    void addNewGnssMessage(const GnssMessage& gnss_message);

    // 定义函数，拿到相机模型指针
    std::vector<std::shared_ptr<GenericCamera>> getCameras() const;

private:
    // 线程开始执行
    void startThread();

    // 构造需要的类对象指针
    void constructPtrs(const MsFusionConfigParams& config_params);

    // IMU线程执行函数
    void runImuThread();

    // ODOM线程执行函数
    void runOdomThread();

    // 视觉线程执行函数
    void runVisionThread();

    // 图片数据处理接口
    void processImageMessage(const ImageMessage& image_message,
                             size_t image_queue_size_after = 0,
                             size_t image_queue_dropped = 0);

    // 记录一帧 IMU 间隔并更新名义采样间隔（最近若干帧间隔的中位数）
    void updateNominalImuDt(double dt);

    // IMU数据插值
    void imuInterpolation(const ImuMeasurement& imu_measurement_pre,
                          const ImuMeasurement& imu_measurement_cur,
                          Eigen::aligned_vector<ImuMeasurement>& imu_measurement_interpolated);

    // 查询 [from_ts, to_ts] 间纯陀螺积分（virtual IMU 系，不扣零偏）的相对旋转
    // R_to_from_from；超出缓存范围返回 false，不外推。不影响 backend 状态估计
    bool queryGyroRotation(double from_ts, double to_ts, Eigen::Quaterniond* R_to_from_from) const;

private:
    // IMU测量数据的队列和互斥锁
    std::mutex imu_message_queue_mutex_;
    std::queue<ImuMessage> imu_message_queue_;

    // odom测量数据的队列和互斥锁
    std::mutex odom_message_queue_mutex_;
    std::queue<OdomMessage> odom_message_queue_;

    // 相机测量数据的队列和互斥锁
    std::mutex image_message_queue_mutex_;
    std::queue<ImageMessage> image_message_queue_;
    // Counts realtime frames discarded by addNewImageMessage() before they can
    // enter the consumer queue.  The vision thread snapshots it into timing.
    std::atomic<std::uint64_t> producer_dropped_count_{0};

    // IMU线程（线程锁，线程对象，条件变量）
    std::thread imu_thread_;
    std::condition_variable imu_thread_condition_;

    // 里程计线程
    std::thread odom_thread_;
    std::condition_variable odom_thread_condition_;

    // 视觉线程（线程锁，线程对象，条件变量）
    std::thread vision_thread_;
    std::condition_variable vision_thread_condition_;

    // 线程退出标志（stop() 置位，各线程无锁读取）
    std::atomic<bool> is_finished_ = false;

    // IMU 名义采样间隔：从实际数据估计（最近帧间隔的中位数），不再读配置；
    // 样本不足前为 0，此时不做断档插值。间隔超过名义值 1.5 倍视为断档。
    double imu_data_dt_ = 0.0;
    std::deque<double> recent_imu_dt_;

    // 相机的指针对象
    std::vector<std::shared_ptr<GenericCamera>> cameras_;

    // 特征点跟踪的指针对象
    std::shared_ptr<Tracking> tracking_ptr_ = nullptr;

    // 当前类用的数据（是否实时处理图片，imu是否使用中值做积分，imu和相机的时间差, IMU需要插值的阈值）
    bool vision_realtime_force_ = true;
    bool imu_use_median_ = false;
    double td_cam_to_imu_ = 0.0;

    // 帧间旋转先验的陀螺积分历史 (timestamp, 累积旋转)，保留最近 2s。
    // IMU 线程写、视觉线程读，用独立锁避免与 IMU 队列争抢
    mutable std::mutex gyro_rotation_mutex_;
    std::deque<std::pair<double, Eigen::Quaterniond>> gyro_rotation_history_;

    // 标志位，是否设置了GNSS的原点
    bool is_set_gnss_origin_ = false;

    // GNSS原点的坐标
    Eigen::Vector3d gnss_origin_lla_ = Eigen::Vector3d::Zero();

    // 配置参数
    MsFusionConfigParams config_params_;
};

} // namespace ms_fusion

#endif // MEASUREMENT_MANAGER_H
