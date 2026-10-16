// PreintegrationNormal 单测：解析解、零残差自洽、雅可比数值对拍、零偏一阶修正。
// 只依赖 Eigen / OpenCV 头文件、Sophus（经 rotation.cpp）和 gtest。
#include "ms_fusion/common/rotation.h"
#include "ms_fusion/preintegration/preintegration_normal.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <functional>
#include <memory>

namespace ms_fusion {
namespace {

// 世界系 z 向下（gravity 取 +9.81），静止时加表比力为 (0, 0, -g)
constexpr double kG = 9.81;
constexpr double kDt = 0.005;  // 200 Hz

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

// integration() 只读 dt / dtheta / dvel，am / wm 一并填上
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

using ImuFunc = std::function<void(double, Eigen::Vector3d&, Eigen::Vector3d&)>;

// 积分 n 个采样，返回对象；imu0 为 t=0 的第一帧
std::unique_ptr<PreintegrationNormal> integrate(const IntegrationState& s0, int n,
                                                const ImuFunc& f, double scale_std = 0.0) {
    Eigen::Vector3d w, a;
    f(0.0, w, a);
    auto params = makeParams();
    params->gyr_scale_std = scale_std;
    auto pre = std::make_unique<PreintegrationNormal>(params, makeImu(0.0, 0.0, w, a), s0);
    for (int i = 1; i <= n; ++i) {
        const double t = i * kDt;
        f(t, w, a);
        pre->addNewImu(makeImu(t, kDt, w, a));
    }
    return pre;
}

// 有旋转、有加速度的时变激励，保证各个耦合项都被激活
void wavyImu(double t, Eigen::Vector3d& w, Eigen::Vector3d& a) {
    w = Eigen::Vector3d(0.3 * std::sin(2.0 * t), -0.2 * std::cos(3.0 * t), 0.4 + 0.1 * std::sin(t));
    a = Eigen::Vector3d(0.8 * std::cos(2.5 * t), 0.5 * std::sin(1.5 * t), -kG + 0.3 * std::sin(3.0 * t));
}

IntegrationState makeState0() {
    IntegrationState s;
    s.timestamp = 0.0;
    s.p = Eigen::Vector3d(1.0, -2.0, 0.5);
    s.q = Rotation::rotvec2quaternion(Eigen::Vector3d(0.2, -0.3, 0.7));
    s.v = Eigen::Vector3d(0.5, 0.2, -0.1);
    s.bg = Eigen::Vector3d(0.002, -0.001, 0.003);
    s.ba = Eigen::Vector3d(0.02, -0.01, 0.03);
    return s;
}

TEST(PreintegrationNormal, ConstantAngularRateGivesExactRotation) {
    const Eigen::Vector3d w(0.0, 0.0, 0.7);
    const int n = 200;
    auto pre = integrate(IntegrationState{}, n, [&](double, Eigen::Vector3d& ww, Eigen::Vector3d& aa) {
        ww = w;
        aa = Eigen::Vector3d::Zero();
    });
    const double T = n * kDt;
    EXPECT_NEAR(pre->deltaTime(), T, 1e-12);
    const Eigen::Quaterniond q_ref = Rotation::rotvec2quaternion(w * T);
    EXPECT_LT((pre->deltaState().q.coeffs() - q_ref.coeffs()).norm(), 1e-12);
}

TEST(PreintegrationNormal, ConstantBodyAccelerationGivesExactPositionAndVelocity) {
    const Eigen::Vector3d a(0.4, -0.3, 0.2);
    const int n = 200;
    auto pre = integrate(IntegrationState{}, n, [&](double, Eigen::Vector3d& ww, Eigen::Vector3d& aa) {
        ww = Eigen::Vector3d::Zero();
        aa = a;
    });
    const double T = n * kDt;
    EXPECT_LT((pre->deltaState().v - a * T).norm(), 1e-12);
    EXPECT_LT((pre->deltaState().p - 0.5 * a * T * T).norm(), 1e-12);
    EXPECT_LT((pre->deltaState().q.coeffs() - Eigen::Quaterniond::Identity().coeffs()).norm(), 1e-15);
}

TEST(PreintegrationNormal, StaticImuDoesNotMoveCurrentState) {
    IntegrationState s0;
    auto pre = integrate(s0, 400, [](double, Eigen::Vector3d& w, Eigen::Vector3d& a) {
        w = Eigen::Vector3d::Zero();
        a = Eigen::Vector3d(0.0, 0.0, -kG);
    });
    EXPECT_LT(pre->currentState().v.norm(), 1e-10);
    EXPECT_LT(pre->currentState().p.norm(), 1e-10);
}

TEST(PreintegrationNormal, FreeFallFollowsGravity) {
    IntegrationState s0;
    const int n = 200;
    auto pre = integrate(s0, n, [](double, Eigen::Vector3d& w, Eigen::Vector3d& a) {
        w = Eigen::Vector3d::Zero();
        a = Eigen::Vector3d::Zero();  // 自由落体：比力为零
    });
    const double T = n * kDt;
    EXPECT_NEAR(pre->currentState().v.z(), kG * T, 1e-9);
    EXPECT_NEAR(pre->currentState().p.z(), 0.5 * kG * T * T, 1e-9);
}

TEST(PreintegrationNormal, BiasIsRemovedFromMeasurements) {
    // 传入的 IMU 含有恒定零偏，初始状态里声明同样的零偏，积分后应等价于零偏为 0 的无运动
    IntegrationState s0;
    s0.bg = Eigen::Vector3d(0.01, -0.02, 0.03);
    s0.ba = Eigen::Vector3d(0.1, -0.2, 0.3);
    auto pre = integrate(s0, 200, [&](double, Eigen::Vector3d& w, Eigen::Vector3d& a) {
        w = s0.bg;
        a = s0.ba;
    });
    EXPECT_LT(pre->deltaState().p.norm(), 1e-12);
    EXPECT_LT(pre->deltaState().v.norm(), 1e-12);
    EXPECT_LT((pre->deltaState().q.coeffs() - Eigen::Quaterniond::Identity().coeffs()).norm(), 1e-12);
}

TEST(PreintegrationNormal, ResidualIsZeroOnOwnIntegratedState) {
    const IntegrationState s0 = makeState0();
    auto pre = integrate(s0, 100, wavyImu);
    const IntegrationState s1 = pre->currentState();

    Eigen::Matrix<double, 15, 1> r;
    pre->evaluate(s0, s1, r.data());
    EXPECT_LT(r.norm(), 1e-6) << r.transpose();
}

TEST(PreintegrationNormal, ResidualGrowsWithStateError) {
    const IntegrationState s0 = makeState0();
    auto pre = integrate(s0, 100, wavyImu);
    IntegrationState s1 = pre->currentState();
    s1.p += Eigen::Vector3d(0.01, 0.0, 0.0);

    Eigen::Matrix<double, 15, 1> r;
    pre->evaluate(s0, s1, r.data());
    EXPECT_GT(r.head<3>().norm(), 1e-3);
}

TEST(PreintegrationNormal, ReintegrationMatchesFreshIntegration) {
    const IntegrationState s0 = makeState0();
    auto a = integrate(s0, 100, wavyImu);

    IntegrationState s0b = s0;
    s0b.bg += Eigen::Vector3d(0.004, -0.003, 0.002);
    s0b.ba += Eigen::Vector3d(-0.02, 0.01, 0.015);
    auto fresh = integrate(s0b, 100, wavyImu);

    IntegrationState tmp = s0b;
    a->reintegration(tmp);
    EXPECT_LT((a->deltaState().p - fresh->deltaState().p).norm(), 1e-12);
    EXPECT_LT((a->deltaState().v - fresh->deltaState().v).norm(), 1e-12);
    EXPECT_LT((a->deltaState().q.coeffs() - fresh->deltaState().q.coeffs()).norm(), 1e-12);
    EXPECT_NEAR(a->deltaTime(), fresh->deltaTime(), 1e-12);
}

// 零偏变化后用一阶雅可比修正预积分量，应比不修正好得多
TEST(PreintegrationNormal, FirstOrderBiasCorrectionBeatsNoCorrection) {
    const IntegrationState s0 = makeState0();
    auto pre = integrate(s0, 100, wavyImu);

    IntegrationState s0b = s0;
    s0b.bg += Eigen::Vector3d(0.004, -0.003, 0.002);
    s0b.ba += Eigen::Vector3d(-0.02, 0.01, 0.015);
    auto truth = integrate(s0b, 100, wavyImu);
    const IntegrationState s1 = truth->currentState();

    Eigen::Matrix<double, 15, 1> r_corr, r_nocorr;
    pre->evaluate(s0b, s1, r_corr.data());     // 带一阶零偏修正
    pre->evaluate(s0, s1, r_nocorr.data());    // state0 的零偏与预积分时一致，不修正
    // 去掉后两个零偏差分块（只看 p/v/q 三块）
    EXPECT_LT(r_corr.head<9>().norm(), 0.05 * r_nocorr.head<9>().norm())
        << "corr=" << r_corr.head<9>().norm() << " nocorr=" << r_nocorr.head<9>().norm();
}

class JacobianFixture : public ::testing::TestWithParam<double> {
protected:
    void SetUp() override {
        s0 = makeState0();
        // 与预积分时的零偏有差异，激活一阶零偏修正（及其 Jr 项）
        pre = integrate(s0, 100, wavyImu, GetParam());
        s0.bg += Eigen::Vector3d(0.02, -0.015, 0.01);
        s0.ba += Eigen::Vector3d(-0.05, 0.03, 0.04);
        s1 = pre->currentState();
        // 让残差不为零，雅可比才有判别力
        s1.p += Eigen::Vector3d(0.02, -0.01, 0.015);
        s1.v += Eigen::Vector3d(-0.01, 0.02, 0.005);
        s1.q = s1.q * Rotation::rotvec2quaternion(Eigen::Vector3d(0.01, -0.02, 0.015));
        s1.bg += Eigen::Vector3d(0.001, 0.002, -0.001);
        s1.ba += Eigen::Vector3d(0.004, -0.002, 0.001);
    }

    Eigen::Matrix<double, 15, 1> eval(const IntegrationState& a, const IntegrationState& b) {
        Eigen::Matrix<double, 15, 1> r;
        pre->evaluate(a, b, r.data());
        return r;
    }

    // 3 列为 p，后 3 列为 q 的右乘切空间扰动；mix 为 v/bg/ba 加性扰动
    static void perturbPose(IntegrationState& s, int col, double h) {
        if (col < 3) {
            s.p[col] += h;
        } else {
            Eigen::Vector3d d = Eigen::Vector3d::Zero();
            d[col - 3] = h;
            s.q = (s.q * Rotation::rotvec2quaternion(d)).normalized();
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

    void expectClose(const Eigen::MatrixXd& analytic, const Eigen::MatrixXd& num, const char* name) {
        const double err = (analytic - num).norm();
        EXPECT_LT(err, 1e-4 * std::max(1.0, num.norm()))
            << name << " err=" << err << " |J|=" << num.norm() << "\nanalytic:\n"
            << analytic << "\nnumeric:\n" << num;
    }

    IntegrationState s0, s1;
    std::unique_ptr<PreintegrationNormal> pre;
};

TEST_P(JacobianFixture, AllFourJacobiansMatchNumericDifferentiation) {
    // 雅可比函数读取上一次 evaluate 缓存的 corrected_q_，先在名义点 evaluate 并取出全部雅可比，
    // 再做有限差分（有限差分会覆盖缓存）。
    eval(s0, s1);
    Eigen::Matrix<double, 15, 7, Eigen::RowMajor> Jp0, Jp1;
    Eigen::Matrix<double, 15, 9, Eigen::RowMajor> Jm0, Jm1;
    pre->residualJacobianPose0(s0, s1, Jp0.data());
    pre->residualJacobianPose1(s0, s1, Jp1.data());
    pre->residualJacobianMix0(s0, s1, Jm0.data());
    pre->residualJacobianMix1(s0, s1, Jm1.data());

    expectClose(Jp0.leftCols<6>(), numeric(true, 6, perturbPose), "pose0");
    expectClose(Jp1.leftCols<6>(), numeric(false, 6, perturbPose), "pose1");
    expectClose(Jm0, numeric(true, 9, perturbMix), "mix0");
    expectClose(Jm1, numeric(false, 9, perturbMix), "mix1");
    // 位姿块第 7 列（四元数 w 分量）不参与切空间，应恒为零
    EXPECT_EQ(Jp0.col(6).norm(), 0.0);
    EXPECT_EQ(Jp1.col(6).norm(), 0.0);
}

INSTANTIATE_TEST_SUITE_P(GyroScaleProxy, JacobianFixture,
                         ::testing::Values(0.0, 0.005, 0.01));

// Inspect weighting without adding diagnostic accessors to the production API.
class CovarianceProbe : public PreintegrationNormal {
public:
    using PreintegrationNormal::PreintegrationNormal;
    const Eigen::MatrixXd& rawCovariance() const { return covariance_; }
    const Eigen::MatrixXd& sqrtInformation() const { return sqrt_information_; }
};

std::unique_ptr<CovarianceProbe> integrateProbe(
    const std::shared_ptr<IntegrationParameters>& params, const IntegrationState& state,
    int samples, const Eigen::Vector3d& measured_rate) {
    const Eigen::Vector3d accel(0.2, -0.1, -kG);
    auto pre = std::make_unique<CovarianceProbe>(
        params, makeImu(0.0, 0.0, measured_rate, accel), state);
    for (int i = 1; i <= samples; ++i) {
        pre->addNewImu(makeImu(i * kDt, kDt, measured_rate, accel));
    }
    return pre;
}

void evaluateProbe(CovarianceProbe& pre, const IntegrationState& state0) {
    Eigen::Matrix<double, 15, 1> residual;
    pre.evaluate(state0, pre.currentState(), residual.data());
    EXPECT_TRUE(residual.allFinite());
    EXPECT_TRUE(pre.sqrtInformation().allFinite());
}

TEST(PreintegrationNormal, ZeroScaleStdUsesOriginalCovariance) {
    auto params = makeParams();
    ASSERT_EQ(params->gyr_scale_std, 0.0);
    const IntegrationState state = makeState0();
    auto pre = integrateProbe(params, state, 40, Eigen::Vector3d(0.5, -0.3, 0.8));
    const Eigen::MatrixXd covariance = pre->rawCovariance();
    evaluateProbe(*pre, state);
    const Eigen::Matrix<double, 15, 15> expected =
        Eigen::LLT<Eigen::Matrix<double, 15, 15>>(covariance.inverse()).matrixL().transpose();
    EXPECT_TRUE(pre->sqrtInformation().isApprox(expected, 1e-12));
    EXPECT_TRUE(pre->rawCovariance().isApprox(covariance, 0.0));
}

TEST(PreintegrationNormal, ScaleProxyUsesDebiasedAngleWithoutAccumulatingIntoCovariance) {
    const IntegrationState state = makeState0();
    const Eigen::Vector3d rate(0.5, -0.3, 0.8);
    constexpr int samples = 40;
    auto baseline = integrateProbe(makeParams(), state, samples, rate);
    auto params = makeParams();
    params->gyr_scale_std = 0.01;
    auto pre = integrateProbe(params, state, samples, rate);
    const Eigen::MatrixXd covariance = pre->rawCovariance();
    EXPECT_TRUE(covariance.isApprox(baseline->rawCovariance(), 0.0));
    EXPECT_TRUE(pre->currentState().p.isApprox(baseline->currentState().p, 0.0));
    EXPECT_TRUE(pre->currentState().v.isApprox(baseline->currentState().v, 0.0));
    EXPECT_TRUE(pre->currentState().q.coeffs().isApprox(
        baseline->currentState().q.coeffs(), 0.0));

    Eigen::Matrix<double, 15, 15> effective_covariance = covariance;
    const Eigen::Vector3d angle_std =
        params->gyr_scale_std * (samples * kDt) * (rate - state.bg);
    effective_covariance.block<3, 3>(6, 6).diagonal() += angle_std.cwiseProduct(angle_std);
    const Eigen::Matrix<double, 15, 15> expected =
        Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
            effective_covariance.inverse()).matrixL().transpose();
    evaluateProbe(*pre, state);
    EXPECT_TRUE(pre->sqrtInformation().isApprox(expected, 1e-9));
    const Eigen::MatrixXd first_information = pre->sqrtInformation();
    for (int i = 0; i < 3; ++i) {
        evaluateProbe(*pre, state);
        EXPECT_TRUE(pre->sqrtInformation().isApprox(first_information, 0.0));
        EXPECT_TRUE(pre->rawCovariance().isApprox(covariance, 0.0));
    }

    // Rebuilding the information after another sample must still use only the
    // original propagated covariance plus the new interval's one-time proxy.
    pre->addNewImu(makeImu((samples + 1) * kDt, kDt, rate, Eigen::Vector3d(0.2, -0.1, -kG)));
    auto fresh = integrateProbe(params, state, samples + 1, rate);
    evaluateProbe(*pre, state);
    evaluateProbe(*fresh, state);
    EXPECT_TRUE(pre->sqrtInformation().isApprox(fresh->sqrtInformation(), 1e-12));
    EXPECT_TRUE(pre->rawCovariance().isApprox(fresh->rawCovariance(), 0.0));
}

TEST(PreintegrationNormal, ReintegrationResetsScaleAngleAndRefreshesInformation) {
    auto params = makeParams();
    params->gyr_scale_std = 0.005;
    IntegrationState state = makeState0();
    const Eigen::Vector3d rate(0.5, -0.3, 0.8);
    auto pre = integrateProbe(params, state, 80, rate);
    evaluateProbe(*pre, state);
    state.bg += Eigen::Vector3d(0.01, -0.02, 0.03);
    auto fresh = integrateProbe(params, state, 80, rate);
    pre->reintegration(state);
    evaluateProbe(*pre, state);
    evaluateProbe(*fresh, state);
    EXPECT_TRUE(pre->rawCovariance().isApprox(fresh->rawCovariance(), 1e-12));
    EXPECT_TRUE(pre->sqrtInformation().isApprox(fresh->sqrtInformation(), 1e-12));
}

TEST(PreintegrationNormal, ZeroDebiasedRotationAddsNoScaleCovariance) {
    const IntegrationState state = makeState0();
    auto baseline = integrateProbe(makeParams(), state, 40, state.bg);
    auto params = makeParams();
    params->gyr_scale_std = 0.01;
    auto pre = integrateProbe(params, state, 40, state.bg);
    evaluateProbe(*baseline, state);
    evaluateProbe(*pre, state);
    EXPECT_TRUE(pre->sqrtInformation().isApprox(baseline->sqrtInformation(), 0.0));
}

TEST(PreintegrationNormal, ScaleProxySumsSignedRatherThanAbsoluteAxisAngles) {
    const IntegrationState state;
    const Eigen::Vector3d rate(0.0, 0.0, 2.0);
    const Eigen::Vector3d accel(0.0, 0.0, -kG);
    auto baseline_params = makeParams();
    auto proxy_params = makeParams();
    proxy_params->gyr_scale_std = 0.01;
    CovarianceProbe baseline(baseline_params, makeImu(0.0, 0.0, rate, accel), state);
    CovarianceProbe proxy(proxy_params, makeImu(0.0, 0.0, rate, accel), state);
    for (int i = 1; i <= 40; ++i) {
        Eigen::Vector3d measured_rate = rate;
        if (i > 20) {
            measured_rate = -rate;
        }
        const auto imu = makeImu(i * kDt, kDt, measured_rate, accel);
        baseline.addNewImu(imu);
        proxy.addNewImu(imu);
    }
    evaluateProbe(baseline, state);
    evaluateProbe(proxy, state);
    EXPECT_TRUE(proxy.sqrtInformation().isApprox(baseline.sqrtInformation(), 1e-12));
}

TEST(PreintegrationNormal, StateDataRoundTrip) {
    const IntegrationState s = makeState0();
    const IntegrationStateData data = PreintegrationNormal::stateToData(s);
    const IntegrationState r = PreintegrationNormal::stateFromData(data);
    EXPECT_LT((r.p - s.p).norm(), 1e-15);
    EXPECT_LT((r.v - s.v).norm(), 1e-15);
    EXPECT_LT((r.bg - s.bg).norm(), 1e-15);
    EXPECT_LT((r.ba - s.ba).norm(), 1e-15);
    EXPECT_LT((r.q.coeffs() - s.q.coeffs()).norm(), 1e-15);
}

TEST(PreintegrationNormal, ConstructStateReadsCeresBlockLayout) {
    // 参数块布局：[p(3), q(x y z w)] [v(3), bg(3), ba(3)] ×2
    const IntegrationState a = makeState0();
    IntegrationState b = makeState0();
    b.p += Eigen::Vector3d(1, 2, 3);
    b.v += Eigen::Vector3d(0.1, 0.2, 0.3);

    const IntegrationStateData da = PreintegrationNormal::stateToData(a);
    const IntegrationStateData db = PreintegrationNormal::stateToData(b);
    const double* blocks[4] = {da.pose, da.mix, db.pose, db.mix};

    auto pre = integrate(IntegrationState{}, 5, wavyImu);
    IntegrationState ra, rb;
    pre->constructState(blocks, ra, rb);
    EXPECT_LT((ra.p - a.p).norm(), 1e-15);
    EXPECT_LT((ra.q.coeffs() - a.q.coeffs()).norm(), 1e-15);
    EXPECT_LT((ra.bg - a.bg).norm(), 1e-15);
    EXPECT_LT((rb.p - b.p).norm(), 1e-15);
    EXPECT_LT((rb.v - b.v).norm(), 1e-15);
    EXPECT_LT((rb.ba - b.ba).norm(), 1e-15);
}

TEST(PreintegrationNormal, DimensionsAreConsistent) {
    auto pre = integrate(IntegrationState{}, 5, wavyImu);
    EXPECT_EQ(pre->numResiduals(), 15);
    EXPECT_EQ(pre->numMixParametersBlocks(), 9);
    const std::vector<int> blocks = pre->numBlocksParameters();
    ASSERT_EQ(blocks.size(), 4u);
    EXPECT_EQ(blocks[0], 7);
    EXPECT_EQ(blocks[1], 9);
    EXPECT_EQ(blocks[2], 7);
    EXPECT_EQ(blocks[3], 9);
}

}  // namespace
}  // namespace ms_fusion
