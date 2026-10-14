#ifndef IMU_PRIOR_FACTOR_H
#define IMU_PRIOR_FACTOR_H

#include "ms_fusion/common/rotation.h"
#include "ms_fusion/preintegration/preintegration.h"

#include <ceres/ceres.h>

namespace ms_fusion {

class IMUPosePriorFactor : public ceres::CostFunction {
public:
    IMUPosePriorFactor(double* pose, double* std) {
        memcpy(pose_, pose, sizeof(double) * 7);

        sqrt_info_.setZero();

        for (size_t k = 0; k < 6; k++) {
            sqrt_info_(k, k) = 1.0 / std[k];
        }

        *mutable_parameter_block_sizes() = std::vector<int>{7};
        set_num_residuals(6);
    }

    bool Evaluate(const double* const* parameters, double* residuals,
                  double** jacobians) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);

        for (size_t k = 0; k < 3; k++) {
            residual(k, 0) = (parameters[0][k] - pose_[k]);
        }

        Eigen::Quaterniond q_p(pose_[6], pose_[3], pose_[4], pose_[5]);
        Eigen::Quaterniond q(parameters[0][6], parameters[0][3], parameters[0][4],
                             parameters[0][5]);

        q = q.normalized();

        residual.block<3, 1>(3, 0) = 2 * (q.inverse() * q_p).vec();

        residual = sqrt_info_ * residual;

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jaco(jacobians[0]);
            jaco.setZero();

            jaco.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            jaco.block<3, 3>(3, 3) =
                -Rotation::quaternionright(q.inverse() * q_p).bottomRightCorner<3, 3>();

            jaco = sqrt_info_ * jaco;
        }

        return true;
    }

private:
    double pose_[7];

    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> sqrt_info_;
};

class IMUMixPriorFactor : public ceres::CostFunction {
public:
    IMUMixPriorFactor(const double* mix, const double* mix_std) {
        // 拷贝数据
        memcpy(mix_, mix, sizeof(double) * 9);
        memcpy(mix_std_, mix_std, sizeof(double) * 9);

        *mutable_parameter_block_sizes() =
            std::vector<int>{Preintegration::numMixParameter()};
        set_num_residuals(Preintegration::numMixParameter());
    }

    bool Evaluate(const double* const* parameters, double* residuals,
                  double** jacobians) const override {
        for (size_t k = 0; k < 9; k++) {
            residuals[k] = (parameters[0][k] - mix_[k]) / mix_std_[k];
        }

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> jaco(jacobians[0]);
            jaco.setZero();
            for (size_t k = 0; k < 9; k++) {
                jaco(k, k) = 1.0 / mix_std_[k];
            }
        }

        return true;
    }

private:
    double mix_[9], mix_std_[9];
};

} // namespace ms_fusion

#endif // IMU_PRIOR_FACTOR_H