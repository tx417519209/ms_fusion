#include "ms_fusion/camera/pinhole_radtan5_camera.h"

namespace ms_fusion {

PinholeRadtan5Camera::PinholeRadtan5Camera(const Eigen::Vector4d& intrinsic_params, 
                                           const Eigen::Matrix<double, 5, 1>& distortion_params)
    : intrinsic_params_(intrinsic_params), 
      distortion_params_(distortion_params) {
    // 求内参的倒数
    inv_intrinsic_params_[0] = 1.0 / intrinsic_params_[0];
    inv_intrinsic_params_[1] = 1.0 / intrinsic_params_[1];
    inv_intrinsic_params_[2] = -intrinsic_params_[2] / intrinsic_params_[0];
    inv_intrinsic_params_[3] = -intrinsic_params_[3] / intrinsic_params_[1];
}

// 由像素坐标转到归一化坐标
void PinholeRadtan5Camera::unproject(const Eigen::Vector2d& proj, Eigen::Vector3d& p3d) {
    // 拿到内参的倒数
    const double& inv_k11 = inv_intrinsic_params_[0];
    const double& inv_k22 = inv_intrinsic_params_[1];
    const double& inv_k13 = inv_intrinsic_params_[2];
    const double& inv_k23 = inv_intrinsic_params_[3];

    // 定义中间变量（带畸变的归一化坐标，不带畸变的归一化坐标）
    double mx_d, my_d, mx_u, my_u;

    // 像素坐标转到带畸变归一化坐标
    mx_d = proj[0] * inv_k11 + inv_k13;
    my_d = proj[1] * inv_k22 + inv_k23;

    // 进行迭代，求不带畸变的归一化坐标
    {
        // 定义迭代的次数
        int iter_num = 10;

        // 定义一个变量, 不带畸变归一化坐标到带畸变归一化坐标的变化量
        Eigen::Vector2d d_u;

        distortion(Eigen::Vector2d(mx_d, my_d), d_u);

        // 更新不带畸变的归一化坐标
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);

        // 进行迭代求解
        for (int i = 0; i < iter_num; i++) {
            // 计算不带畸变的归一化坐标到带畸变归一化坐标的变化量
            distortion(Eigen::Vector2d(mx_u, my_u), d_u);

            // 更新不带畸变的归一化坐标
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);
        }
    }

    p3d << mx_u, my_u, 1.0;
}

// 由相机坐标或者归一化坐标，转到像素坐标
void PinholeRadtan5Camera::project(const Eigen::Vector3d& p3d, Eigen::Vector2d& proj) {
    // 拿到对应的相机内参
    const double& fx = intrinsic_params_[0];
    const double& fy = intrinsic_params_[1];
    const double& cx = intrinsic_params_[2];
    const double& cy = intrinsic_params_[3];

    // 定义带畸变和不带畸变的归一化坐标
    Eigen::Vector2d p_d, p_u;

    // 求不带畸变的归一化坐标
    p_u << p3d(0) / p3d(2), p3d(1) / p3d(2);

    // 求不带畸变归一化坐标到带畸变归一化坐标的变化量
    Eigen::Vector2d d_u;
    distortion(p_u, d_u);

    // 计算带畸变的归一化坐标
    p_d = p_u + d_u;

    // 计算像素坐标
    proj << fx * p_d(0) + cx, fy * p_d(1) + cy;
}

// 输入一组像素坐标，转到归一化坐标
void PinholeRadtan5Camera::unprojects(const std::vector<cv::Point2f>& projs, 
                                      Eigen::aligned_vector<Eigen::Vector3d>& p3ds) {
    // 遍历一组像素坐标，转到归一化坐标
    for (const auto& proj : projs) {
        Eigen::Vector3d p3d;
        unproject(Eigen::Vector2d(proj.x, proj.y), p3d);
        p3ds.push_back(p3d);
    }
}

// 获取相机的焦距
double PinholeRadtan5Camera::getFocalLength() {
    return (intrinsic_params_(0) + intrinsic_params_(1)) / 2.0;
}

// 不带畸变的归一化坐标到带畸变的归一化坐标的变化量
void PinholeRadtan5Camera::distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) {
    // 拿到对应的畸变系数
    const double& k1 = distortion_params_[0];
    const double& k2 = distortion_params_[1];
    const double& p1 = distortion_params_[2];
    const double& p2 = distortion_params_[3];
    const double& k3 = distortion_params_[4];

    // 定义中间变量
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    // 这里求出来中间变量
    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);

    // 这里求出来rho2_u
    rho2_u = mx2_u + my2_u;
    // 这里求出来rad_dist_u
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u + k3 * rho2_u * rho2_u * rho2_u;

    // 这里求出来d_u
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

} // namespace ms_fusion
