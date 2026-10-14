#ifndef ZERO_VELOCITY_FACTOR_H
#define ZERO_VELOCITY_FACTOR_H

#include "eigen_utils.hpp"

#include <ceres/ceres.h>

namespace ms_fusion {

// 静止判据确认后加入的零速伪观测：约束最新状态的速度趋近于0。静止时视觉视差
// 接近0、平移方向观测本就弱，加表零偏又和重力倾角耦合，零偏会在滑窗优化里自由
// 游走并通过二次积分表现成位置来回摆动；这个因子把速度钉在0附近，压住这个游走。
// 参数块是 mix[9]（v, bg, ba），只用前3维（v）。
class ZeroVelocityFactor : public ceres::SizedCostFunction<3, 9> {
public:
    explicit ZeroVelocityFactor(double velocity_std) : velocity_std_(velocity_std) {}

    bool Evaluate(const double* const* parameters, double* residuals,
                  double** jacobians) const override {
        for (int k = 0; k < 3; ++k) {
            residuals[k] = parameters[0][k] / velocity_std_;
        }

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> jaco(jacobians[0]);
            jaco.setZero();
            for (int k = 0; k < 3; ++k) {
                jaco(k, k) = 1.0 / velocity_std_;
            }
        }

        return true;
    }

private:
    double velocity_std_;
};

} // namespace ms_fusion

#endif // ZERO_VELOCITY_FACTOR_H
