#ifndef GENERIC_CAMERA_H
#define GENERIC_CAMERA_H

#include "eigen_utils.hpp"

#include <opencv2/core/core.hpp>

#include <memory>
#include <string>

namespace ms_fusion {

class GenericCamera {
public:
    // 由像素坐标转到归一化坐标
    virtual void unproject(const Eigen::Vector2d& proj, Eigen::Vector3d& p3d) = 0;

    // 由相机坐标或者归一化坐标，转到像素坐标
    virtual void project(const Eigen::Vector3d& p3d, Eigen::Vector2d& proj) = 0;

    // 输入一组像素坐标，转到归一化坐标
    virtual void unprojects(const std::vector<cv::Point2f>& projs, 
                            Eigen::aligned_vector<Eigen::Vector3d>& p3ds) = 0;

    // 获取相机的焦距
    virtual double getFocalLength() = 0;
};

} // namespace ms_fusion

#endif // GENERIC_CAMERA_H
