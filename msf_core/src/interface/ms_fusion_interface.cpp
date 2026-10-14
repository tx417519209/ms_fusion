#include "ms_fusion_interface.h"
#include "ms_fusion/manager/ms_fusion_manager.h"
#include "ms_fusion/manager/measurement_manager.h"

#include <opencv2/opencv.hpp>

namespace ms_fusion {

// 系统初始化接口
void initMsFusion(const MsFusionConfigParams& config_params) {
    // 初始化系统
    MsFusionManager::SnglPtr()->initMsFusionManager(config_params);
}

void stopMsFusion() {
    // 先停后端：fusion 线程等 IMU 时一直持有视觉队列锁，视觉线程可能正卡在
    // pushVisionMeasurement() 上，后端先退出放锁，视觉线程才能回到循环头部退出
    MsFusionManager::SnglPtr()->stop();
    MeasurementManager::SnglPtr()->stop();
}

// imu数据处理接口
void sendImuMessageToManager(const ImuMessage& imu_message) {
    // 发送imu数据到系统
    MeasurementManager::SnglPtr()->addNewImuMessage(imu_message);
}

// odom数据处理接口
void sendOdomMessageToManager(const OdomMessage& odom_message) {
    MeasurementManager::SnglPtr()->addNewOdomMessage(odom_message);
}

// 相机数据处理接口
void sendImageMessageToManager(const ImageMessage& image_message) {
    // 发送相机数据到系统
    MeasurementManager::SnglPtr()->addNewImageMessage(image_message);
}

// GNSS数据处理接口
void sendGnssMessageToManager(const GnssMessage& gnss_message) {
    // 发送GNSS数据到系统
    MeasurementManager::SnglPtr()->addNewGnssMessage(gnss_message);
}

// 设置发布里程计的回调函数
void setPubOdomInfoCallback(const std::string& name, const pubOdomInfoFunc& func) {
    // 设置发布里程计的回调函数
    MsFusionManager::SnglPtr()->pushOdomPubCallback(name, func);
}

void setPubDebugImageCallback(const std::string& name, const pubDebugImageFunc& func) {
    MsFusionManager::SnglPtr()->pushDebugImagePubCallback(name, func);
}

void setPubKeyframeLoopCallback(const pubKeyframeLoopFunc& func) {
    MsFusionManager::SnglPtr()->pushKeyframeLoopPubCallback(func);
}

void setPubRelocCorrectionCallback(const pubRelocCorrectionFunc& func) {
    MsFusionManager::SnglPtr()->pushRelocCorrectionPubCallback(func);
}

void sendRelocCorrectionAckToManager(const RelocCorrectionAck& ack) {
    MsFusionManager::SnglPtr()->onRelocCorrectionAck(ack);
}

bool rtkCalibrate() {
    return MsFusionManager::SnglPtr()->rtkCalibrate();
}

bool requestReloc(double timeout_sec, RelocPoseResult* result) {
    return MsFusionManager::SnglPtr()->requestReloc(timeout_sec, result);
}

} // namespace ms_fusion