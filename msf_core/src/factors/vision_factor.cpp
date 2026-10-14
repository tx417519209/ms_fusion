#include "ms_fusion/factors/vision_factor.h"
#include "ms_fusion/common/rotation.h"

#include <cmath>

namespace ms_fusion {

namespace {
// cam_z 接近 0 时残差和雅可比都按 1/cam_z 发散，Huber 压不住。cam_z 小于
// kMinProjectionDepth（含落到相机后方）时把分母钳制为该常数，对 z 的偏导为 0；
// 正常深度下与原始透视投影完全一致。
constexpr double kMinProjectionDepth = 0.05;

// 相机系点 p 的归一化投影，jacobian 非空时写入 2x3 雅可比
inline Eigen::Vector2d safeProject(const Eigen::Vector3d& p,
                                    Eigen::Matrix<double, 2, 3>* jacobian) {
    const double raw_d = p.z();
    const bool protected_region = raw_d < kMinProjectionDepth;
    const double d = protected_region ? kMinProjectionDepth : raw_d;
    const double inv_d = 1.0 / d;

    if (jacobian) {
        if (protected_region) {
            // d 是常数，不再随 p.z() 变化，对 z 的偏导精确为 0
            *jacobian << inv_d, 0.0, 0.0, 0.0, inv_d, 0.0;
        } else {
            *jacobian << inv_d, 0.0, -p.x() * inv_d * inv_d, 0.0, inv_d,
                -p.y() * inv_d * inv_d;
        }
    }
    return Eigen::Vector2d(p.x() * inv_d, p.y() * inv_d);
}
}  // namespace

StereoAnchorFactor::StereoAnchorFactor(const Eigen::Vector3d& left,
                                       const Eigen::Vector3d& right,
                                       const Eigen::Matrix3d& R_cam1_from_cam0,
                                       const Eigen::Vector3d& t_cam1_from_cam0, double std)
    : rotated_left_ray_(R_cam1_from_cam0 * left), right_ray_(right),
      t_cam1_from_cam0_(t_cam1_from_cam0), inv_std_(1.0 / std) {}

bool StereoAnchorFactor::Evaluate(const double* const* parameters, double* residuals,
                                  double** jacobians) const {
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual.setZero();
    if (jacobians && jacobians[0]) {
        Eigen::Map<Eigen::Vector2d> jacobian_depth(jacobians[0]);
        jacobian_depth.setZero();
    }
    const double inverse_depth = parameters[0][0];
    if (!std::isfinite(inverse_depth) || inverse_depth <= 0.0 ||
        !std::isfinite(inv_std_) || inv_std_ <= 0.0) {
        return false;
    }
    const Eigen::Vector3d point_right =
        rotated_left_ray_ / inverse_depth + t_cam1_from_cam0_;
    if (!point_right.allFinite() || !right_ray_.allFinite()) {
        return false;
    }
    Eigen::Matrix<double, 2, 3> projection_jacobian;
    const bool need_jacobian = jacobians && jacobians[0];
    residual = inv_std_ *
        (safeProject(point_right, need_jacobian ? &projection_jacobian : nullptr) -
         right_ray_.head<2>());
    if (need_jacobian) {
        Eigen::Map<Eigen::Vector2d> jacobian_depth(jacobians[0]);
        jacobian_depth = -inv_std_ * projection_jacobian * rotated_left_ray_ /
                         (inverse_depth * inverse_depth);
        if (!jacobian_depth.allFinite()) {
            return false;
        }
    }
    return residual.allFinite();
}

// 构造函数
MonoVisionFactor::MonoVisionFactor(const Eigen::Vector3d& pns0, const Eigen::Vector3d& pns1,
                                   const Eigen::Matrix3d& R_imu_from_cam, 
                                   const Eigen::Vector3d& t_imu_from_cam,
                                   double std)
    : pns0_(pns0), pns1_(pns1), R_imu_from_cam_(R_imu_from_cam)
    , t_imu_from_cam_(t_imu_from_cam) {
    // 赋值信息矩阵
    sqrt_info_.setZero();
    sqrt_info_(0, 0) = 1.0 / std;
    sqrt_info_(1, 1) = 1.0 / std;

    // 赋值IMU系到相机系的变换矩阵
    R_cam_from_imu_ = R_imu_from_cam_.transpose();
}

// 做更新和优化
bool MonoVisionFactor::Evaluate(const double* const* parameters, double* residuals,
                                double** jacobians) const {
    // 拿到参考帧的状态量（T_w_from_bi）
    Eigen::Vector3d p0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond q0(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    // 拿到当前帧的状态量（T_w_from_bj）
    Eigen::Vector3d p1(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond q1(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // 拿到特征点的逆深度
    double id0 = parameters[2][0];

    // 拿到特征点在参考帧和当前帧的归一化坐标
    Eigen::Vector3d pns_0_td = pns0_;
    Eigen::Vector3d pns_1_td = pns1_;

    // 求特征点在参考帧C系下的坐标
    Eigen::Vector3d pns_c_0 = pns_0_td / id0;
    // 把特征点从参考帧C系转到参考帧的IMU系
    Eigen::Vector3d pns_b_0 = R_imu_from_cam_ * pns_c_0 + t_imu_from_cam_;
    // 把特征点从参考帧的IMU系转到W系
    Eigen::Vector3d pns_n = q0.toRotationMatrix() * pns_b_0 + p0;

    // 现在计算两个变换矩阵
    const Eigen::Matrix3d R_w_from_imu0 = q0.toRotationMatrix();
    const Eigen::Matrix3d R_imu1_from_w = q1.toRotationMatrix().transpose();

    // 把特征点从世界系转到当前帧的IMU系下
    Eigen::Vector3d pns_b_1 = R_imu1_from_w * (pns_n - p1);

    // 把特征点转从当前帧的IMU坐标系转到当前帧的相机坐标系
    Eigen::Vector3d pns_c_1 = R_cam_from_imu_ * (pns_b_1 - t_imu_from_cam_);

    // 定义残差对特征点当前相机系坐标的雅可比（不需要雅可比时不写入）
    Eigen::Matrix<double, 2, 3> reduce;

    // 定义残差，计算当前帧下面的残差
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = safeProject(pns_c_1, jacobians ? &reduce : nullptr) - pns_1_td.head(2);
    residual = sqrt_info_ * residual;

    // 计算雅可比矩阵
    if (jacobians) {
        // 乘信息矩阵
        reduce = sqrt_info_ * reduce;

        // 定义矩阵A(A = R_cam1_from_imu1 * R_imu1_from_w = R_cam1_from_w)
        const Eigen::Matrix3d A = R_cam_from_imu_ * R_imu1_from_w;
        // 定义矩阵B(B = R_cam1_from_w * R_w_from_imu0 = R_cam1_from_imu0)
        const Eigen::Matrix3d B = A * R_w_from_imu0;

        // 求对位姿0的雅可比矩阵
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            Eigen::Matrix<double, 3, 6> jaco_i;
            const Eigen::Matrix3d skew0 = Rotation::skewSymmetric(pns_b_0);

            jaco_i.leftCols<3>() = A;
            jaco_i.rightCols<3>() = -B * skew0;

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        // 求对位姿1的雅可比矩阵
         if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            Eigen::Matrix<double, 3, 6> jaco_j;
            const Eigen::Matrix3d skew1 = Rotation::skewSymmetric(pns_b_1);

            jaco_j.leftCols<3>() = -A;
            jaco_j.rightCols<3>() = R_cam_from_imu_ * skew1;

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }

        // 求对逆深度的雅可比矩阵
        if (jacobians[2]) {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[2]);
            // reuse B = cbc * cnb1 * cb0n computed above
            Eigen::Matrix3d M3 = B * R_cam_from_imu_.transpose();
            jacobian_feature = -reduce * M3 * pns_0_td / (id0 * id0);
        }
    }

    return true;
}

// 构造函数
StereoVisionFactor::StereoVisionFactor(const Eigen::Vector3d& ref_pn, 
                                       const Eigen::Vector3d& obs_pn0,
                                       const Eigen::Vector3d& obs_pn1, 
                                       const Eigen::Matrix3d& R_imu_from_cam0,
                                       const Eigen::Vector3d& t_imu_from_cam0,
                                       const Eigen::Matrix3d& R_cam1_from_cam0,
                                       const Eigen::Vector3d& t_cam1_from_cam0,
                                       double std)
    : ref_pn_(ref_pn), obs_pn0_(obs_pn0), obs_pn1_(obs_pn1)
    , R_imu_from_cam0_(R_imu_from_cam0)
    , t_imu_from_cam0_(t_imu_from_cam0)
    , R_cam1_from_cam0_(R_cam1_from_cam0)
    , t_cam1_from_cam0_(t_cam1_from_cam0) {
    // 赋值信息矩阵
    sqrt_info_.setZero();
    sqrt_info_(0, 0) = 1.0 / std;
    sqrt_info_(1, 1) = 1.0 / std;
    sqrt_info_(2, 2) = 1.0 / std;
    sqrt_info_(3, 3) = 1.0 / std;

    // 赋值IMU系到相机系的变换矩阵
    R_cam0_from_imu_ = R_imu_from_cam0.transpose();
}

// 做更新和优化
bool StereoVisionFactor::Evaluate(const double* const* parameters, double* residuals,
                                 double** jacobians) const {
    // 拿到参考帧的状态量（T_w_from_bi）
    Eigen::Vector3d p0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond q0(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    // 拿到当前帧的状态量（T_w_from_bj）
    Eigen::Vector3d p1(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond q1(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // 拿到优化变量的逆深度
    double id0 = parameters[2][0];

    // 拿到特征点在参考帧和当前帧的归一化坐标
    Eigen::Vector3d pns_0_td = ref_pn_;
    Eigen::Vector3d pns_1_td_left = obs_pn0_;
    Eigen::Vector3d pns_1_td_right = obs_pn1_;

    // 求特征点在参考帧C系下的坐标
    Eigen::Vector3d pns_c_0 = pns_0_td / id0;
    // 把特征点从参考帧C系转到参考帧的IMU系
    Eigen::Vector3d pns_b_0 = R_imu_from_cam0_ * pns_c_0 + t_imu_from_cam0_;
    // 把特征点从参考帧的IMU系转到W系
    Eigen::Vector3d pns_w = q0.toRotationMatrix() * pns_b_0 + p0;

    // 拿到参考帧IMU系到世界系的变换矩阵
    const Eigen::Matrix3d R_w_from_imu0 = q0.toRotationMatrix();
    // 拿到世界系到当前帧IMU系的变换矩阵
    const Eigen::Matrix3d R_imu1_from_w = q1.toRotationMatrix().transpose();

    // 把特征点从世界系转换到当前帧的IMU系坐标系下
    Eigen::Vector3d pns_b_1 = R_imu1_from_w * (pns_w - p1);

    // 把特征点从当前帧的IMU系坐标系转到当前帧的左相机坐标系
    Eigen::Vector3d pns_c_1_left = R_cam0_from_imu_ * (pns_b_1 - t_imu_from_cam0_);

    // 把特征点从左相机转到右相机坐标系下
    Eigen::Vector3d pns_c_1_right = R_cam1_from_cam0_ * pns_c_1_left + t_cam1_from_cam0_;

    // 定义残差对特征点当前相机系坐标的雅可比
    Eigen::Matrix<double, 2, 3> reduce_left, reduce_right;

    // 定义残差，计算当前帧下面的残差
    Eigen::Map<Eigen::Vector4d> residual(residuals);
    residual.head(2) =
        safeProject(pns_c_1_left, jacobians ? &reduce_left : nullptr) - pns_1_td_left.head(2);
    residual.tail(2) =
        safeProject(pns_c_1_right, jacobians ? &reduce_right : nullptr) - pns_1_td_right.head(2);
    residual = sqrt_info_ * residual;

    // 计算雅可比矩阵
    if (jacobians) {
        // 乘信息矩阵
        reduce_left = sqrt_info_.block<2, 2>(0, 0) * reduce_left;
        reduce_right = sqrt_info_.block<2, 2>(2, 2) * reduce_right;

        // 定义矩阵A(A = R_cam1_from_imu1 * R_imu1_from_w = R_cam1_from_w)
        const Eigen::Matrix3d A = R_cam0_from_imu_ * R_imu1_from_w;
        // 定义矩阵B(B = R_cam1_from_w * R_w_from_imu0 = R_cam1_from_imu0)
        const Eigen::Matrix3d B = A * R_w_from_imu0;

        // 求对位姿0的雅可比矩阵
        if (jacobians[0]) {
            // 定义雅可比矩阵
            Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            const Eigen::Matrix3d skew0 = Rotation::skewSymmetric(pns_b_0);

            jaco_i.leftCols<3>() = A;
            jaco_i.rightCols<3>() = -B * skew0;

            // 求前两维残差对位姿0的雅可比矩阵
            jacobian_pose_i.block<2, 6>(0, 0).leftCols<6>() = reduce_left * jaco_i;
            jacobian_pose_i.block<2, 1>(0, 6).setZero();

            // 求后两维残差对位姿0的雅可比矩阵
            jacobian_pose_i.block<2, 6>(2, 0).leftCols<6>() = reduce_right * R_cam1_from_cam0_ * jaco_i;
            jacobian_pose_i.block<2, 1>(2, 6).setZero();
        }

        // 求对位姿1的雅可比矩阵
        if (jacobians[1]) {
            // 定义雅可比矩阵
            Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            const Eigen::Matrix3d skew1 = Rotation::skewSymmetric(pns_b_1);

            jaco_j.leftCols<3>() = -A;
            jaco_j.rightCols<3>() = R_cam0_from_imu_ * skew1;

            // 求前两维残差对位姿1的雅可比矩阵  
            jacobian_pose_j.block<2, 6>(0, 0).leftCols<6>() = reduce_left * jaco_j;
            jacobian_pose_j.block<2, 1>(0, 6).setZero();

            // 求后两维残差对位姿1的雅可比矩阵
            jacobian_pose_j.block<2, 6>(2, 0).leftCols<6>() = reduce_right * R_cam1_from_cam0_ * jaco_j;
            jacobian_pose_j.block<2, 1>(2, 6).setZero();
        }

        // 求对逆深度的雅可比矩阵
        if (jacobians[2]) {
            Eigen::Map<Eigen::Vector4d> jacobian_feature(jacobians[2]);
            // reuse B = cbc * cnb1 * cb0n computed above
            Eigen::Matrix3d M3 = B * R_cam0_from_imu_.transpose();

            // 求前两维残差对逆深度的雅可比矩阵
            jacobian_feature.head(2) = -reduce_left * M3 * pns_0_td / (id0 * id0);

            // 求后两维残差对逆深度的雅可比矩阵
            jacobian_feature.tail(2) = -reduce_right * R_cam1_from_cam0_ * M3 * pns_0_td / (id0 * id0);
        }
    }
    
    return true;
}

} // namespace ms_fusion
