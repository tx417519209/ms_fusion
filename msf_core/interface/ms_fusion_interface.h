#ifndef MS_FUSION_INTERFACE_H
#define MS_FUSION_INTERFACE_H

#include "config_params.h"
#include "ms_fusion_message.h"

namespace ms_fusion {

// 定义回调函数
using pubOdomInfoFunc = std::function<void(const OdomInfo& odom_info)>;
using pubDebugImageFunc = std::function<void(const DebugImageMessage& debug_image)>;
using pubKeyframeLoopFunc = std::function<void(const KeyframeLoopMessage& keyframe)>;
using pubRelocCorrectionFunc = std::function<void(const RelocPoseResult& result)>;

// 系统初始化接口
void initMsFusion(const MsFusionConfigParams& config_params);

// 系统停止接口：停止并 join 全部工作线程。进程退出前、setPub*Callback 注册的回调所绑定的
// 对象析构前调用（否则线程退出过程中仍可能调用已析构对象的发布回调）
void stopMsFusion();

// imu数据处理接口
void sendImuMessageToManager(const ImuMessage& imu_message);

// odom数据处理接口
void sendOdomMessageToManager(const OdomMessage& odom_message);

// 相机数据处理接口
void sendImageMessageToManager(const ImageMessage& image_message);

// GNSS数据处理接口
void sendGnssMessageToManager(const GnssMessage& gnss_message);

// 设置发布里程计的回调函数
void setPubOdomInfoCallback(const std::string& name, const pubOdomInfoFunc& func);

// 设置发布调试图像的回调函数
void setPubDebugImageCallback(const std::string& name, const pubDebugImageFunc& func);

// 设置关键帧回环数据发布回调
void setPubKeyframeLoopCallback(const pubKeyframeLoopFunc& func);

// 设置 reloc 绝对修正发布回调（周期重定位触发、或 /reloc 手动触发，Loop
// 开启时都经这个回调把 raw/target base_link 位姿交给 ROS 层发布）
void setPubRelocCorrectionCallback(const pubRelocCorrectionFunc& func);

// Loop 对 reloc 修正的 ACK（ROS 层收到后转进来）。可在任意线程调用，不阻塞。
void sendRelocCorrectionAckToManager(const RelocCorrectionAck& ack);

// /rtk_calibrate：直接将融合后端位姿设为球场目标 (-12, 0, yaw=0)
bool rtkCalibrate();

// /reloc：触发一次视觉重定位。Loop 开启时返回绝对修正观测（同时经上面的
// 回调发布），关闭时直接对齐后端（默认超时 5s）
bool requestReloc(double timeout_sec = 5.0, RelocPoseResult* result = nullptr);

} // namespace ms_fusion

#endif // MS_FUSION_INTERFACE_H
