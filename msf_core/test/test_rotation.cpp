// Rotation / IntegratedRotation 单测：只依赖 Eigen、Sophus 和 gtest。
#include "ms_fusion/common/rotation.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>

namespace ms_fusion {
namespace {

constexpr double kTol = 1e-12;

TEST(Rotation, RotvecQuaternionRoundTrip) {
    const Eigen::Vector3d cases[] = {
        {0.3, -0.2, 0.5}, {1.0, 2.0, -0.5}, {0.0, 0.0, 3.0}, {1e-8, 2e-8, -1e-8}};
    for (const auto& v : cases) {
        const Eigen::Quaterniond q = Rotation::rotvec2quaternion(v);
        EXPECT_NEAR(q.norm(), 1.0, kTol);
        EXPECT_LT((Rotation::quaternion2rotvec(q) - v).norm(), 1e-9) << v.transpose();
    }
}

TEST(Rotation, ZeroRotvecIsIdentityNotNaN) {
    const Eigen::Quaterniond q = Rotation::rotvec2quaternion(Eigen::Vector3d::Zero());
    EXPECT_TRUE(q.coeffs().allFinite());
    EXPECT_NEAR(q.w(), 1.0, kTol);
    EXPECT_NEAR(q.vec().norm(), 0.0, kTol);
}

TEST(Rotation, RotvecMatchesAngleAxis) {
    const Eigen::Vector3d v(0.4, -0.7, 0.2);
    const Eigen::Matrix3d R =
        Rotation::quaternion2matrix(Rotation::rotvec2quaternion(v));
    const Eigen::Matrix3d R_ref = Eigen::AngleAxisd(v.norm(), v.normalized()).toRotationMatrix();
    EXPECT_LT((R - R_ref).norm(), kTol);
}

TEST(Rotation, EulerMatrixRoundTrip) {
    // 比较旋转矩阵而不是角度本身，避免万向节附近表示不唯一、航向被折到 [0, 2π) 带来的歧义
    const Eigen::Vector3d cases[] = {{0.1, 0.2, 0.3}, {-0.5, 0.4, -2.0}, {1.0, -1.2, 3.0}};
    for (const auto& e : cases) {
        const Eigen::Matrix3d R = Rotation::euler2matrix(e);
        const Eigen::Matrix3d R2 = Rotation::euler2matrix(Rotation::matrix2euler(R));
        EXPECT_LT((R - R2).norm(), 1e-9) << e.transpose();
        const Eigen::Matrix3d R3 =
            Rotation::quaternion2matrix(Rotation::euler2quaternion(e));
        EXPECT_LT((R - R3).norm(), 1e-9) << e.transpose();
    }
}

TEST(Rotation, EulerHeadingInZeroTwoPi) {
    const Eigen::Vector3d e = Rotation::matrix2euler(Rotation::euler2matrix({0.0, 0.0, -0.5}));
    EXPECT_GE(e[2], 0.0);
    EXPECT_LT(e[2], 2.0 * M_PI);
    EXPECT_NEAR(e[2], 2.0 * M_PI - 0.5, 1e-9);
}

TEST(Rotation, SkewSymmetricIsCrossProduct) {
    const Eigen::Vector3d a(0.3, -1.2, 2.0), b(-0.7, 0.4, 1.1);
    EXPECT_LT((Rotation::skewSymmetric(a) * b - a.cross(b)).norm(), kTol);
    const Eigen::Matrix3d S = Rotation::skewSymmetric(a);
    EXPECT_LT((S + S.transpose()).norm(), kTol);
}

// 4x4 矩阵按 [w, x, y, z] 排列，Eigen 的 coeffs() 是 [x, y, z, w]
Eigen::Vector4d wxyz(const Eigen::Quaterniond& q) {
    return Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
}

TEST(Rotation, QuaternionLeftRightMultiplication) {
    const Eigen::Quaterniond q1 = Rotation::rotvec2quaternion({0.3, -0.4, 0.5});
    const Eigen::Quaterniond q2 = Rotation::rotvec2quaternion({-0.6, 0.1, 0.9});
    const Eigen::Vector4d prod = wxyz(q1 * q2);
    EXPECT_LT((Rotation::quaternionleft(q1) * wxyz(q2) - prod).norm(), kTol);
    EXPECT_LT((Rotation::quaternionright(q2) * wxyz(q1) - prod).norm(), kTol);
}

TEST(Rotation, NormalizeRotationProjectsToSO3) {
    const Eigen::Matrix3d R = Rotation::euler2matrix({0.2, -0.3, 0.9});
    Eigen::Matrix3d noisy = R;
    noisy(0, 0) += 1e-3;
    noisy(1, 2) -= 2e-3;
    const Eigen::Matrix3d fixed = Rotation::normalizeRotation(noisy);
    EXPECT_LT((fixed.transpose() * fixed - Eigen::Matrix3d::Identity()).norm(), 1e-12);
    EXPECT_NEAR(fixed.determinant(), 1.0, 1e-12);
    EXPECT_LT((fixed - R).norm(), 1e-2);
}

TEST(IntegratedRotation, MatchesExponentialMap) {
    const Eigen::Vector3d wm(0.8, -0.5, 1.3);
    const double dt = 0.2;  // |wm*dt| > 1e-4，走非小量分支
    IntegratedRotation ir(wm, dt);
    const Eigen::Matrix3d R_ref = Rotation::quaternion2matrix(Rotation::rotvec2quaternion(wm * dt));
    EXPECT_LT((ir.delta_R_ - R_ref).norm(), 1e-12);
}

TEST(IntegratedRotation, SmallAngleBranchIsAccurate) {
    const Eigen::Vector3d wm(1e-3, -2e-3, 1.5e-3);
    const double dt = 0.01;  // |wm*dt| ~ 3e-5 < 1e-4，走小量分支
    IntegratedRotation ir(wm, dt);
    const Eigen::Matrix3d R_ref = Rotation::quaternion2matrix(Rotation::rotvec2quaternion(wm * dt));
    EXPECT_LT((ir.delta_R_ - R_ref).norm(), 1e-8);
    EXPECT_LT((ir.right_J_ - Eigen::Matrix3d::Identity()).norm(), 1e-4);
}

// 两个独立实现之间的一致性：Jr(v) * Jr(v)^{-1} = I
TEST(IntegratedRotation, RightJacobianTimesInverseIsIdentity) {
    const Eigen::Vector3d cases[] = {{0.3, -0.2, 0.5}, {1.0, 0.5, -0.8}, {0.05, 0.02, -0.03}};
    for (const auto& v : cases) {
        IntegratedRotation ir(v, 1.0);
        const Eigen::Matrix3d prod = Rotation::inverseRightJacobianSO3(v) * ir.right_J_;
        EXPECT_LT((prod - Eigen::Matrix3d::Identity()).norm(), 1e-9) << v.transpose();
    }
}

TEST(IntegratedRotation, RightJacobianFirstOrderProperty) {
    // Exp(v + dv) ≈ Exp(v) * Exp(Jr(v) * dv)
    const Eigen::Vector3d v(0.4, -0.3, 0.6), dv(1e-6, -2e-6, 1.5e-6);
    IntegratedRotation ir(v, 1.0);
    const Eigen::Matrix3d lhs = Rotation::quaternion2matrix(Rotation::rotvec2quaternion(v + dv));
    const Eigen::Matrix3d rhs =
        Rotation::quaternion2matrix(Rotation::rotvec2quaternion(v)) *
        Rotation::quaternion2matrix(Rotation::rotvec2quaternion(ir.right_J_ * dv));
    EXPECT_LT((lhs - rhs).norm(), 1e-10);
}

}  // namespace
}  // namespace ms_fusion
