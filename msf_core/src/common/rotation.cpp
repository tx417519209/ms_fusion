#include "ms_fusion/common/rotation.h"

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <cmath>

namespace ms_fusion {

const double eps = 1e-4;

Eigen::Quaterniond Rotation::matrix2quaternion(const Eigen::Matrix3d& matrix) {
    return Eigen::Quaterniond(matrix);
}

/**
 * 四元数转旋转矩阵
 * @param quaternion 四元数
 * @return 旋转矩阵
 */
Eigen::Matrix3d Rotation::quaternion2matrix(const Eigen::Quaterniond& quaternion) {
    return quaternion.toRotationMatrix();
}

/**
 * 旋转矢量转四元数
 * @param rotvec 旋转矢量
 * @return 四元数
 */
Eigen::Quaterniond Rotation::rotvec2quaternion(const Eigen::Vector3d& rotvec) {
    // 计算旋转角度
    double theta = rotvec.norm();
    // Eigen::AngleAxisd requires a valid axis.  A zero rotation has no
    // defined axis, and normalized() on a zero vector produces NaNs.
    if (!std::isfinite(theta) || theta < 1e-12) {
        return Eigen::Quaterniond::Identity();
    }
    // 计算旋转轴
    Eigen::Vector3d axis = rotvec.normalized();

    // 返回四元数
    return Eigen::Quaterniond(Eigen::AngleAxisd(theta, axis)).normalized();
}

/**
 * 四元数转旋转矢量
 * @param quaternion 四元数
 * @return 旋转矢量
 */
Eigen::Vector3d Rotation::quaternion2rotvec(const Eigen::Quaterniond& quaternion) {
    // 四元数直接转轴角
    Eigen::AngleAxisd angle_axisd(quaternion);
    return angle_axisd.angle() * angle_axisd.axis();
}

/**
 * 四元数转旋转矩阵
 */

/**
 * 四元数转欧拉角
 */
Eigen::Vector3d Rotation::quaternion2euler(const Eigen::Quaterniond& quaternion) {
    return matrix2euler(quaternion.toRotationMatrix());
}

/**
 * 旋转矩阵转欧拉角
 */
Eigen::Vector3d Rotation::matrix2euler(const Eigen::Matrix3d& dcm) {
    Eigen::Vector3d euler;

    euler[1] = atan(-dcm(2, 0) / sqrt(dcm(2, 1) * dcm(2, 1) + dcm(2, 2) * dcm(2, 2)));

    if (dcm(2, 0) <= -0.999) {
        euler[0] = atan2(dcm(2, 1), dcm(2, 2));
        euler[2] = atan2((dcm(1, 2) - dcm(0, 1)), (dcm(0, 2) + dcm(1, 1)));
    } else if (dcm(2, 0) >= 0.999) {
        euler[0] = atan2(dcm(2, 1), dcm(2, 2));
        euler[2] = M_PI + atan2((dcm(1, 2) + dcm(0, 1)), (dcm(0, 2) - dcm(1, 1)));
    } else {
        euler[0] = atan2(dcm(2, 1), dcm(2, 2));
        euler[2] = atan2(dcm(1, 0), dcm(0, 0));
    }

    // heading 0~2PI
    if (euler[2] < 0) {
        euler[2] = M_PI * 2 + euler[2];
    }

    return euler;
}

/**
 * 欧拉角转旋转矩阵
 */
Eigen::Matrix3d Rotation::euler2matrix(const Eigen::Vector3d& euler) {
    return Eigen::Matrix3d(Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()) * 
                           Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) * 
                           Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()));
}

/**
 * 欧拉角转四元数
 */
Eigen::Quaterniond Rotation::euler2quaternion(const Eigen::Vector3d& euler) {
    return Eigen::Quaterniond(Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()) * 
                              Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) * 
                              Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()));
}

Eigen::Matrix4d Rotation::quaternionleft(const Eigen::Quaterniond& q) {
    Eigen::Matrix4d ans;
    ans(0, 0) = q.w();
    ans.block<1, 3>(0, 1) = -q.vec().transpose();
    ans.block<3, 1>(1, 0) = q.vec();
    ans.block<3, 3>(1, 1) = q.w() * Eigen::Matrix3d::Identity() + skewSymmetric(q.vec());
    return ans;
}

Eigen::Matrix4d Rotation::quaternionright(const Eigen::Quaterniond& p) {
    Eigen::Matrix4d ans;
    ans(0, 0) = p.w();
    ans.block<1, 3>(0, 1) = -p.vec().transpose();
    ans.block<3, 1>(1, 0) = p.vec();
    ans.block<3, 3>(1, 1) = p.w() * Eigen::Matrix3d::Identity() - skewSymmetric(p.vec());
    return ans;
}

// 求反对称矩阵
Eigen::Matrix3d Rotation::skewSymmetric(const Eigen::Vector3d& vector) {
    Eigen::Matrix3d mat;
    mat << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1), vector(0), 0;
    return mat;
}

// 使旋转矩阵满足约束
Eigen::Matrix3d Rotation::normalizeRotation(const Eigen::Matrix3d& R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

// 求右雅可比矩阵的逆
Eigen::Matrix3d Rotation::inverseRightJacobianSO3(const Eigen::Vector3d& v) {
    // 拿到向量的三个元素
    const double& x = v[0];
    const double& y = v[1];
    const double& z = v[2];

    const double d2 = x*x + y*y + z*z;
    const double d = sqrt(d2);

    // 定义向量对应的反对称矩阵
    Eigen::Matrix3d v_hat;
    v_hat << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;

    // 判断向量是否为小量
    if (d < 1e-5) {
        // 这个是小量
        return Eigen::Matrix3d::Identity();
    } else {
        // 这个不为小量
        return Eigen::Matrix3d::Identity() + v_hat / 2.0 + v_hat * v_hat * (1.0 / d2 - (1.0 + cos(d))/(2.0 * d * sin(d)));
    }
}

// 积分旋转增量的构造函数
IntegratedRotation::IntegratedRotation(const Eigen::Vector3d& wm, const double& dt) {
    // 计算等效旋转矢量
    const double x = wm(0) * dt;
    const double y = wm(1) * dt;
    const double z = wm(2) * dt;

    // 计算等效旋转矢量的模值
    const double d2 = x * x + y * y + z * z;
    const double d = sqrt(d2);

    // 赋值等效旋转矢量
    Eigen::Vector3d rotvec(x, y, z);

    // 计算等效旋转矢量对应的hat
    Eigen::Matrix3d rotvec_hat = Sophus::SO3d::hat(rotvec);

    // 判断等效旋转矢量是否为小量
    if (d < eps) {
        // 如果是小量
        delta_R_ = Eigen::Matrix3d::Identity() + rotvec_hat;
        // 右雅可比矩阵为单位阵矩阵
        right_J_ = Eigen::Matrix3d::Identity();
    } else {
        // 如果不是小量
        delta_R_ = Eigen::Matrix3d::Identity() + rotvec_hat * sin(d) / d + 
                   rotvec_hat * rotvec_hat * (1.0f - cos(d)) / d2;
        right_J_ = Eigen::Matrix3d::Identity() - rotvec_hat * (1.0f - cos(d)) / d2 + 
                   rotvec_hat * rotvec_hat * (d - sin(d)) / (d2 * d);
    }
}


} // namespace ms_fusion
