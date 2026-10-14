#ifndef MS_FUSION_COMMON_INS_MECH_H
#define MS_FUSION_COMMON_INS_MECH_H

#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/measurement.h"
#include "ms_fusion/preintegration/integration_state.h"

namespace ms_fusion {

class InsMech {
public:
    // 给一个时间戳找到窗口中对应的索引
    static size_t getInsWindowIndex(
        const Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& window, 
        double time);

    // 给一个时间戳找到里程计队列中对应的索引号
    static size_t getDequeIndex(const Eigen::aligned_deque<OdomMeasurement>& odom_deque, double time);

    // IMU进行单次的机械编排
    static void insMechanization(const IntegrationConfiguration& config, 
                                 const ImuMeasurement& imu_pre, 
                                 const ImuMeasurement& imu_cur,
                                 IntegrationState& state);

    // 优化以后进行多次机械编排
    static void redoInsMechanization(const IntegrationConfiguration& config, 
                                     const IntegrationState& updated_state,
                                     size_t reserved_ins_num,
                                     Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& ins_windows);

    // 判断IMU是否需要插值
    static int isNeedInterpolation(const ImuMeasurement& imu0, const ImuMeasurement& imu1,
                                   double mid);

    // 判断ODOM是否需要插值
    static int odomIsNeedInterpolation(const OdomMeasurement& odom0, const OdomMeasurement& odom1, double mid);
    
    // IMU进行插值
    static void imuInterpolation(const ImuMeasurement& imu01, ImuMeasurement& imu00,
                                 ImuMeasurement& imu11, double mid);

    // ODOM进行插值
    static void odomInterpolation(const OdomMeasurement& odom0, const OdomMeasurement& odom1, 
                                  OdomMeasurement& odom01, double mid);

    // 获取IMU观测值的系列
    static void getImuMeasurementSeries(
        const Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& ins_window,
        double start, double end,
        Eigen::aligned_vector<ImuMeasurement>& imu_measurement_series);

    // 获取里程计观测值
    static void getOdomMeasurementSeries(Eigen::aligned_deque<OdomMeasurement>& odom_deque,
                                         double start, double end,
                                         Eigen::aligned_vector<OdomMeasurement>& odom_measurement_series);

    // 给一个时间戳，找到对应的索引
    static size_t getStateDataIndex(const std::deque<double>& timelist, double time, double interval);

    // 给两个时间戳是不是统一个时间戳
    static bool isTheSameTimeNode(double time0, double time1, double interval);

    // 给一个状态量，求相机系到世界系的变换矩阵
    static Pose stateToPoseWithExtrinsic(const IntegrationState& state, const Pose& pose_ext);

public:
    // 允许的时间对齐误差
    static constexpr double MINIMUM_TIME_INTERVAL = 0.0001;
};

} // namespace ms_fusion

#endif // MS_FUSION_COMMON_INS_MECH_H