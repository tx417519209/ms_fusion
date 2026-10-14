#ifndef IMU_ERROR_FACTOR_H
#define IMU_ERROR_FACTOR_H

#include "ms_fusion/preintegration/preintegration.h"

#include <ceres/ceres.h>

namespace ms_fusion {

class IMUErrorFactor : public ceres::CostFunction {
public:
    // gyr_bias_std/acc_bias_std：与 PreintegrationNormal::setNoiseMatrix() 用的是
    // 同一份换算结果（配置里的 gb_std/ab_std 转成 SI 单位），保证这里的绝对零偏
    // 先验和预积分里的零偏随机游走过程噪声出自同一个配置量，不再各自为政。
    explicit IMUErrorFactor(double gyr_bias_std, double acc_bias_std)
        : gyr_bias_std_(gyr_bias_std), acc_bias_std_(acc_bias_std) {
        *mutable_parameter_block_sizes() =
            std::vector<int>{Preintegration::numMixParameter()};

        set_num_residuals(6);
    }

    bool Evaluate(const double* const* parameters, double* residuals,
                  double** jacobians) const override {
        // bg, ba
        for (size_t k = 0; k < 3; k++) {
            residuals[k + 0] = parameters[0][k + 3] / gyr_bias_std_;
            residuals[k + 3] = parameters[0][k + 6] / acc_bias_std_;
        }

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 6, 9, Eigen::RowMajor>> jaco(jacobians[0]);
            jaco.setZero();

            for (size_t k = 0; k < 3; k++) {
                jaco(k + 0, k + 3) = 1.0 / gyr_bias_std_;
                jaco(k + 3, k + 6) = 1.0 / acc_bias_std_;
            }
        }

        return true;
    }

private:
    double gyr_bias_std_;
    double acc_bias_std_;
};

} // namespace ms_fusion

#endif // IMU_ERROR_FACTOR_H