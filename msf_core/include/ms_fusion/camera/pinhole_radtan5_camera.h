#ifndef PINHOLE_RADTAN5_CAMERA_H
#define PINHOLE_RADTAN5_CAMERA_H

#include "ms_fusion/camera/generic_camera.h"

namespace ms_fusion {

class PinholeRadtan5Camera : public GenericCamera {
    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    PinholeRadtan5Camera(const Eigen::Vector4d& intrinsic_params, 
                         const Eigen::Matrix<double, 5, 1>& distortion_params);
    ~PinholeRadtan5Camera() = default;

    // 由像素坐标求归一化坐标
    void unproject(const Eigen::Vector2d& proj, Eigen::Vector3d& p3d) override;

    // 由相机坐标或者归一化坐标，转到像素坐标
    void project(const Eigen::Vector3d& p3d, Eigen::Vector2d& proj) override;

    // 输入一组像素坐标，转到归一化坐标
    void unprojects(const std::vector<cv::Point2f>& projs, 
                    Eigen::aligned_vector<Eigen::Vector3d>& p3ds) override;

    // 获取相机的焦距
    double getFocalLength() override;

private:
    // 不带畸变的归一化坐标到带畸变的归一化坐标的变化量
    void distortion(const Eigen::Vector2d& undist, Eigen::Vector2d& d_u);

private:
    // 相机内参
    Eigen::Vector4d intrinsic_params_;

    // 相机畸变系数
    Eigen::Matrix<double, 5, 1> distortion_params_;

    // 求内参的倒数
    Eigen::Vector4d inv_intrinsic_params_;
};

} // namespace ms_fusion

#endif // PINHOLE_RADTAN5_CAMERA_H
