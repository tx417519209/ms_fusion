#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/padded_image_geometry.h"
#include "ms_fusion/common/runtime_log.h"

#include <iomanip>
#include <iostream>

namespace ms_fusion {

// 图片的高和宽
int image_width = 0;
int image_height = 0;
// 相机的焦距
double focal_length = 0.0;
// 真实IMU坐标系到虚拟IMU坐标系（前-右-下）的变换矩阵
Pose T_virtual_imu_from_real_imu;
// 更新
std::mutex camera_imu_extrinsic_params_mutex;
// 左相机和右相机的外参
CameraExtrinsicParams camera_extrinsic_params;
// 左相机和右相机到IMU的外参
CameraImuExtrinsicParams camera_imu_extrinsic_params;

// 设置全局参数
void ParametersManager::readParameters(const MsFusionConfigParams& config_params) {
    // // 赋值真实IMU坐标系到虚拟IMU坐标系（前-右-下）的变换矩阵
    T_virtual_imu_from_real_imu = 
        Pose(config_params.imu_config_params_.T_virtual_imu_from_real_imu.block<3, 3>(0, 0), 
             config_params.imu_config_params_.T_virtual_imu_from_real_imu.block<3, 1>(0, 3));

    // 拿到相机到IMU的变换矩阵（这个变换矩阵还是用的真实IMU系）
    auto T_imu_from_cams = config_params.camera_config_params_.T_imu_from_cams;

    // 拿到左相机到IMU的变换矩阵(这里是真实的IMU变换矩阵)
    const auto R_imu_from_cam0 = T_imu_from_cams[0].block<3, 3>(0, 0);
    const auto t_imu_from_cam0 = T_imu_from_cams[0].block<3, 1>(0, 3);

    // 拿到左相机到虚拟IMU的变换矩阵（这里转到虚拟的前右下IMU坐标系了）
    Eigen::Matrix3d R_virtual_imu_from_cam0 = T_virtual_imu_from_real_imu.R * R_imu_from_cam0;
    Eigen::Vector3d t_virtual_imu_from_cam0 = T_virtual_imu_from_real_imu.R * t_imu_from_cam0 + 
                                              T_virtual_imu_from_real_imu.t;


    // 拿到右相机到IMU的变换矩阵(现在是真实的IMU坐标系)
    const auto R_imu_from_cam1 = T_imu_from_cams[1].block<3, 3>(0, 0);
    const auto t_imu_from_cam1 = T_imu_from_cams[1].block<3, 1>(0, 3);

    // 拿到右相机到虚拟IMU坐标系的变换矩阵
    Eigen::Matrix3d R_virtual_imu_from_cam1 = T_virtual_imu_from_real_imu.R * R_imu_from_cam1;
    Eigen::Vector3d t_virtual_imu_from_cam1 = T_virtual_imu_from_real_imu.R * t_imu_from_cam1 + 
                                              T_virtual_imu_from_real_imu.t;

    // 赋值相机和虚拟IMU的外参（这里变成了虚拟的IMU坐标系）
    camera_imu_extrinsic_params.T_imu_from_cam0 = Pose(R_virtual_imu_from_cam0, t_virtual_imu_from_cam0);
    camera_imu_extrinsic_params.T_imu_from_cam1 = Pose(R_virtual_imu_from_cam1, t_virtual_imu_from_cam1);
    camera_imu_extrinsic_params.T_cam0_from_imu = 
        camera_imu_extrinsic_params.T_imu_from_cam0.inverse();
    camera_imu_extrinsic_params.T_cam1_from_imu = 
        camera_imu_extrinsic_params.T_imu_from_cam1.inverse();

    // 赋值相机外参
    camera_extrinsic_params.T_cam1_from_cam0 = camera_imu_extrinsic_params.T_cam1_from_imu * 
                                               camera_imu_extrinsic_params.T_imu_from_cam0;
    camera_extrinsic_params.T_cam0_from_cam1 = camera_imu_extrinsic_params.T_cam0_from_imu * 
                                               camera_imu_extrinsic_params.T_imu_from_cam1;

    const auto& T_cam0_to_cam1_pose = camera_extrinsic_params.T_cam1_from_cam0;
    Eigen::Matrix4d T_cam0_to_cam1 = Eigen::Matrix4d::Identity();
    T_cam0_to_cam1.block<3, 3>(0, 0) = T_cam0_to_cam1_pose.R;
    T_cam0_to_cam1.block<3, 1>(0, 3) = T_cam0_to_cam1_pose.t;
    MSF_CLOG << std::fixed << std::setprecision(8);
    MSF_CLOG << "T_cam0_to_cam1 (virtual IMU frame, p_cam1 = R * p_cam0 + t):\n"
             << T_cam0_to_cam1 << std::endl;
    MSF_CLOG << "  R_cam0_to_cam1:\n" << T_cam0_to_cam1_pose.R << std::endl;
    MSF_CLOG << "  t_cam0_to_cam1: [" << T_cam0_to_cam1_pose.t.transpose() << "] m" << std::endl;
    MSF_CLOG << "  baseline |t|: " << T_cam0_to_cam1_pose.t.norm() << " m" << std::endl;

    // 赋值图片的高和宽
    image_width = config_params.camera_config_params_.cam_resolutions.at(0).at(0);
    image_height = config_params.camera_config_params_.cam_resolutions.at(0).at(1);

    // 赋值相机焦距
    focal_length = (config_params.camera_config_params_.cam_intrinsics.at(0).at(0) + 
                    config_params.camera_config_params_.cam_intrinsics.at(0).at(1) +
                    config_params.camera_config_params_.cam_intrinsics.at(1).at(0) +
                    config_params.camera_config_params_.cam_intrinsics.at(1).at(1)) * 0.25;
    
    // 如果相机降采样
    if (config_params.tracking_config_params_.downsize_camera) {
        int tracking_w = 0;
        int tracking_h = 0;
        ConfigUtils::computeTrackingImageSize(config_params, tracking_w, tracking_h);
        image_width = tracking_w;
        image_height = tracking_h;

        double scaled_focal_sum = 0.0;
        size_t scaled_focal_count = 0;
        const auto& camera_params = config_params.camera_config_params_;
        for (size_t camera_index = 0; camera_index < camera_params.cam_intrinsics.size();
             ++camera_index) {
            const auto& resolution = camera_params.cam_resolutions.at(camera_index);
            const auto& intrinsics = camera_params.cam_intrinsics.at(camera_index);
            const auto geometry = makeTopLeftPaddedImageGeometry(
                resolution.at(0), resolution.at(1), tracking_w, tracking_h);
            scaled_focal_sum += intrinsics.at(0) * geometry.scale_x;
            scaled_focal_sum += intrinsics.at(1) * geometry.scale_y;
            scaled_focal_count += 2;
        }
        focal_length = scaled_focal_sum / static_cast<double>(scaled_focal_count);
    }
}

// 更新相机和IMU的外参
void ParametersManager::updateCameraImuExtrinsicParams(const Pose&) {
    std::lock_guard<std::mutex> lock(camera_imu_extrinsic_params_mutex);
}

} // namespace ms_fusion
