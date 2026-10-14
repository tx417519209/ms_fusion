#ifndef CREATE_CAMERA_H
#define CREATE_CAMERA_H

#include "ms_fusion/camera/pinhole_radtan5_camera.h"
#include "ms_fusion/camera/pinhole_radtan4_camera.h"

#include <iostream>
#include "ms_fusion/common/runtime_log.h"

namespace ms_fusion {

// 相机模型构造类
class CreateCamera {
public:
    static std::shared_ptr<GenericCamera> createCamera(const std::vector<double>& camera_intrinsic, 
                                                       const std::vector<double>& camera_distortion,
                                                       const std::string& camera_model) {
        // 根据相机模型创建对应的相机
        if (camera_model == "pinhole-radtan5") {
            // 拿到相机内参
            auto intrinsic_params = Eigen::Vector4d(camera_intrinsic.data());
            // 拿到相机畸变参数
            auto distortion_params = Eigen::Matrix<double, 5, 1>(camera_distortion.data());

            // 打印相机模型和内参
            MSF_ILOG << "Camera model: " << camera_model << std::endl;
            MSF_ILOG << "Camera intrinsic: " << intrinsic_params.transpose() << std::endl;
            MSF_ILOG << "Camera distortion: " << distortion_params.transpose() << std::endl;
            
            // 创建相机
            return std::make_shared<PinholeRadtan5Camera>(intrinsic_params, distortion_params);
                
        } else if (camera_model == "pinhole-radtan4") {
            // 拿到相机内参
            auto intrinsic_params = Eigen::Vector4d(camera_intrinsic.data());
            // 拿到相机畸变参数
            auto distortion_params = Eigen::Vector4d(camera_distortion.data());

            // 打印相机模型和内参
            MSF_ILOG << "Camera model: " << camera_model << std::endl;
            MSF_ILOG << "Camera intrinsic: " << intrinsic_params.transpose() << std::endl;
            MSF_ILOG << "Camera distortion: " << distortion_params.transpose() << std::endl;
                
            // 创建相机
            return std::make_shared<PinholeRadtan4Camera>(intrinsic_params, distortion_params);
                
        } else {
            throw std::runtime_error("Invalid camera model: " + camera_model);
        }
    }
};


} // namespace ms_fusion

#endif // CREATE_CAMERA_H
