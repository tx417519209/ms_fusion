#ifndef MS_FUSION_COMMON_ROTATION_H
#define MS_FUSION_COMMON_ROTATION_H

#include "eigen_utils.hpp"

namespace ms_fusion {

const double D2R = (M_PI / 180.0);
const double R2D = (180.0 / M_PI);

class Rotation {
public:
    // 旋转矩阵转四元数
    static Eigen::Quaterniond matrix2quaternion(const Eigen::Matrix3d& matrix);

    // 四元数转旋转矩阵
    static Eigen::Matrix3d quaternion2matrix(const Eigen::Quaterniond& quaternion);

    // 旋转矢量转四元数
    static Eigen::Quaterniond rotvec2quaternion(const Eigen::Vector3d& rotvec);

    // 四元数转旋转矢量
    static Eigen::Vector3d quaternion2rotvec(const Eigen::Quaterniond& quaternion);

    // 旋转矩阵转欧拉角
    static Eigen::Vector3d matrix2euler(const Eigen::Matrix3d& matrix);
    // 欧拉角转旋转矩阵
    static Eigen::Matrix3d euler2matrix(const Eigen::Vector3d& euler);

    // 欧拉角转四元数
    static Eigen::Quaterniond euler2quaternion(const Eigen::Vector3d& euler);
    // 四元数转欧拉角
    static Eigen::Vector3d quaternion2euler(const Eigen::Quaterniond& quaternion);

    // 四元数的左乘矩阵和右乘矩阵
    static Eigen::Matrix4d quaternionleft(const Eigen::Quaterniond& q);
    static Eigen::Matrix4d quaternionright(const Eigen::Quaterniond& p);

    // 求反对称矩阵
    static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& vector);

    // 使旋转矩阵满足约束
    static Eigen::Matrix3d normalizeRotation(const Eigen::Matrix3d& R);

    // 求右雅可比矩阵的逆
    static Eigen::Matrix3d inverseRightJacobianSO3(const Eigen::Vector3d& v);
};

// 给角速度，求旋转增量和右雅可比矩阵
class IntegratedRotation {
public:
    IntegratedRotation() {}

    // 构造函数
    IntegratedRotation(const Eigen::Vector3d& wm, const double& dt);

public:
    // 与上一帧的时间差
    double delta_time_ = 0.0;

    // 计算出来的旋转增量
    Eigen::Matrix3d delta_R_ = Eigen::Matrix3d::Identity();
    
    // 计算右雅可比矩阵
    Eigen::Matrix3d right_J_ = Eigen::Matrix3d::Identity();

    // Eigen数据类型内存对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace ms_fusion

#endif // MS_FUSION_COMMON_ROTATION_H