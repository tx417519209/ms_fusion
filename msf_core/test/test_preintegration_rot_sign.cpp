// PreintegrationNormal 旋转残差 / 解析雅可比的四元数符号分支。
//
// 旋转残差由旋转矩阵恢复四元数 q_rot 后取 2*vec，而解析雅可比按状态四元数乘积
// q_prod = corrected_q^-1 * q0^-1 * q1 推导。q 与 -q 表示同一姿态：状态四元数符号
// （例如 consistencyCheck 经矩阵恢复后，yaw 为负且越过约 120° 时 w 的符号会翻转）与 q_rot
// 相反时，残差不变而旋转雅可比若不处理会反号。三个含旋转行的雅可比（Pose0 / Pose1 / Mix0）
// 乘以 sign(q_prod · q_rot) 使其一致。
// 本测试对真实的 PreintegrationNormal 做中心差分：
//   1. 残差与状态四元数符号无关；
//   2. 符号一致与符号相反两种情形下，解析雅可比都与数值差分一致。
// 只依赖 Eigen、Sophus（经 rotation.cpp）和 gtest。
#include "ms_fusion/common/rotation.h"
#include "ms_fusion/preintegration/preintegration_normal.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>

namespace ms_fusion {
namespace {

constexpr double kG = 9.81;
constexpr double kDt = 0.005;

std::shared_ptr<IntegrationParameters> makeParams() {
    auto p = std::make_shared<IntegrationParameters>();
    p->acc_vrw = 0.02;
    p->gyr_arw = 0.002;
    p->gyr_bias_std = 1e-3;
    p->acc_bias_std = 1e-2;
    p->corr_time = 100.0;
    p->gravity = kG;
    return p;
}

ImuMeasurement makeImu(double t, double dt, const Eigen::Vector3d& w, const Eigen::Vector3d& a) {
    ImuMeasurement m;
    m.timestamp = t;
    m.dt = dt;
    m.wm = w;
    m.am = a;
    m.dtheta = w * dt;
    m.dvel = a * dt;
    return m;
}

// 有旋转、有加速度的时变激励（与 test_preintegration.cpp 同款），保证各耦合项都被激活
void wavyImu(double t, Eigen::Vector3d& w, Eigen::Vector3d& a) {
    w = Eigen::Vector3d(0.3 * std::sin(2.0 * t), -0.2 * std::cos(3.0 * t), 0.4 + 0.1 * std::sin(t));
    a = Eigen::Vector3d(0.8 * std::cos(2.5 * t), 0.5 * std::sin(1.5 * t),
                        -kG + 0.3 * std::sin(3.0 * t));
}

std::unique_ptr<PreintegrationNormal> integrate(const IntegrationState& s0) {
    Eigen::Vector3d w, a;
    wavyImu(0.0, w, a);
    auto pre = std::make_unique<PreintegrationNormal>(makeParams(),
                                                      makeImu(0.0, 0.0, w, a), s0);
    for (int i = 1; i <= 100; ++i) {
        const double t = i * kDt;
        wavyImu(t, w, a);
        pre->addNewImu(makeImu(t, kDt, w, a));
    }
    return pre;
}

// 大旋转（绕 -Z 约 121°，贴近转换分支边界）作为起点姿态，带少量倾斜
IntegrationState makeState0() {
    IntegrationState s;
    s.timestamp = 0.0;
    s.p = Eigen::Vector3d(1.0, -2.0, 0.5);
    s.q = Rotation::rotvec2quaternion(Eigen::Vector3d(0.15, -0.25, -2.11));
    s.v = Eigen::Vector3d(0.5, 0.2, -0.1);
    s.bg = Eigen::Vector3d(0.002, -0.001, 0.003);
    s.ba = Eigen::Vector3d(0.02, -0.01, 0.03);
    return s;
}

using Residual = Eigen::Matrix<double, 15, 1>;

// opposite_sign=true 时把 s1.q 取负：同一姿态，四元数符号相反
class Fixture {
public:
    explicit Fixture(bool opposite_sign) : pre(integrate(makeState0())) {
        s0 = makeState0();
        // 与预积分时的零偏有差异，激活一阶零偏修正；让残差不为零，雅可比才有判别力
        s0.bg += Eigen::Vector3d(0.02, -0.015, 0.01);
        s0.ba += Eigen::Vector3d(-0.05, 0.03, 0.04);
        s1 = pre->currentState();
        s1.p += Eigen::Vector3d(0.02, -0.01, 0.015);
        s1.v += Eigen::Vector3d(-0.01, 0.02, 0.005);
        s1.q = s1.q * Rotation::rotvec2quaternion(Eigen::Vector3d(0.01, -0.02, 0.015));
        s1.bg += Eigen::Vector3d(0.001, 0.002, -0.001);
        s1.ba += Eigen::Vector3d(0.004, -0.002, 0.001);
        // 先把 s1.q 规范到与 q0^-1 乘积 w>0 的一侧，再按需取负，保证两种情形的差别只是符号
        Eigen::Quaterniond prod = s0.q.inverse() * s1.q;
        if (prod.w() < 0.0) {
            s1.q.coeffs() *= -1.0;
        }
        if (opposite_sign) {
            s1.q.coeffs() *= -1.0;
        }
    }

    Residual eval(const IntegrationState& a, const IntegrationState& b) {
        Residual r;
        pre->evaluate(a, b, r.data());
        return r;
    }

    static void perturbPose(IntegrationState& s, int col, double h) {
        if (col < 3) {
            s.p[col] += h;
        } else {
            Eigen::Vector3d d = Eigen::Vector3d::Zero();
            d[col - 3] = h;
            s.q = (s.q * Rotation::rotvec2quaternion(d)).normalized();  // 右乘扰动，保持符号连续
        }
    }
    static void perturbMix(IntegrationState& s, int col, double h) {
        if (col < 3) {
            s.v[col] += h;
        } else if (col < 6) {
            s.bg[col - 3] += h;
        } else {
            s.ba[col - 6] += h;
        }
    }

    template <typename Perturb>
    Eigen::MatrixXd numeric(bool on_state0, int cols, Perturb perturb) {
        constexpr double h = 1e-6;
        Eigen::MatrixXd J(15, cols);
        for (int c = 0; c < cols; ++c) {
            IntegrationState a0 = s0, a1 = s1, b0 = s0, b1 = s1;
            perturb(on_state0 ? a0 : a1, c, h);
            perturb(on_state0 ? b0 : b1, c, -h);
            J.col(c) = (eval(a0, a1) - eval(b0, b1)) / (2 * h);
        }
        return J;
    }

    // 返回 Pose0 / Pose1 / Mix0 三个含旋转行雅可比的解析与数值差的最大范数（相对 max(1,|J|)）
    double worstRelativeError() {
        // 雅可比函数读取上一次 evaluate 缓存的 corrected_q_ / rot_sign_，先在名义点 evaluate，
        // 再取全部解析雅可比，最后做有限差分（有限差分会覆盖缓存）。
        eval(s0, s1);
        Eigen::Matrix<double, 15, 7, Eigen::RowMajor> Jp0, Jp1;
        Eigen::Matrix<double, 15, 9, Eigen::RowMajor> Jm0, Jm1;
        pre->residualJacobianPose0(s0, s1, Jp0.data());
        pre->residualJacobianPose1(s0, s1, Jp1.data());
        pre->residualJacobianMix0(s0, s1, Jm0.data());
        pre->residualJacobianMix1(s0, s1, Jm1.data());
        double worst = 0.0;
        const auto rel = [&](const Eigen::MatrixXd& analytic, const Eigen::MatrixXd& num) {
            worst = std::max(worst, (analytic - num).norm() / std::max(1.0, num.norm()));
        };
        rel(Jp0.leftCols<6>(), numeric(true, 6, perturbPose));
        rel(Jp1.leftCols<6>(), numeric(false, 6, perturbPose));
        rel(Jm0, numeric(true, 9, perturbMix));
        rel(Jm1, numeric(false, 9, perturbMix));
        return worst;
    }

    IntegrationState s0, s1;
    std::unique_ptr<PreintegrationNormal> pre;
};

TEST(PreintegrationRotSign, FixtureReallyProducesOppositeQuaternionSigns) {
    Fixture aligned(false), opposite(true);
    const Eigen::Quaterniond prod_a =
        aligned.pre->deltaState().q.inverse() * aligned.s0.q.inverse() * aligned.s1.q;
    const Eigen::Quaterniond prod_o =
        opposite.pre->deltaState().q.inverse() * opposite.s0.q.inverse() * opposite.s1.q;
    EXPECT_GT(prod_a.w(), 0.0);
    EXPECT_LT(prod_o.w(), 0.0);
}

TEST(PreintegrationRotSign, ResidualIsInvariantToStateQuaternionSign) {
    Fixture aligned(false), opposite(true);
    const Residual ra = aligned.eval(aligned.s0, aligned.s1);
    const Residual ro = opposite.eval(opposite.s0, opposite.s1);
    EXPECT_LT((ra - ro).norm(), 1e-9) << "aligned:\n" << ra.transpose() << "\nopposite:\n"
                                       << ro.transpose();
}

TEST(PreintegrationRotSign, AlignedSignMatchesNumericDifferentiation) {
    Fixture f(false);
    EXPECT_LT(f.worstRelativeError(), 1e-4);
}

TEST(PreintegrationRotSign, OppositeSignMatchesNumericDifferentiation) {
    Fixture f(true);
    EXPECT_LT(f.worstRelativeError(), 1e-4);
}

}  // namespace
}  // namespace ms_fusion
