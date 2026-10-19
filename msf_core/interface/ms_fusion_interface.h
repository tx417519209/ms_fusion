#ifndef MS_FUSION_INTERFACE_H
#define MS_FUSION_INTERFACE_H

#include "config_params.h"
#include "ms_fusion_message.h"

namespace ms_fusion {

// 定义回调函数
using pubOdomInfoFunc = std::function<void(const OdomInfo& odom_info)>;
using pubDebugImageFunc = std::function<void(const DebugImageMessage& debug_image)>;
using pubKeyframeLoopFunc = std::function<void(const KeyframeLoopMessage& keyframe)>;

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

// /rtk_calibrate：直接将融合后端位姿设为球场目标 (-12, 0, yaw=0)
bool rtkCalibrate();

} // namespace ms_fusion

#endif // MS_FUSION_INTERFACE_H
