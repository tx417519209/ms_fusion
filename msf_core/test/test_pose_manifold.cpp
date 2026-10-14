// PoseManifold 单测：Plus / Minus 互逆、四元数保持单位长度、基本维度。
// 注意：PlusJacobian 是 VINS 风格的"前 6 行取单位阵"近似，不是 Plus 的真实导数，这里不对它做数值对拍。
#include "ms_fusion/factors/pose_mainfold.h"
#include "ms_fusion/common/rotation.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace ms_fusion {
namespace {

struct Pose7 {
    double v[7];
};

Pose7 makePose(const Eigen::Vector3d& p, const Eigen::Vector3d& rotvec) {
    Pose7 x;
    Eigen::Map<Eigen::Vector3d>(x.v) = p;
    const Eigen::Quaterniond q = Rotation::rotvec2quaternion(rotvec);
    x.v[3] = q.x();
    x.v[4] = q.y();
    x.v[5] = q.z();
    x.v[6] = q.w();
    return x;
}

TEST(PoseManifold, Sizes) {
    PoseManifold m;
    EXPECT_EQ(m.AmbientSize(), 7);
    EXPECT_EQ(m.TangentSize(), 6);
}

TEST(PoseManifold, PlusZeroIsIdentity) {
    PoseManifold m;
    const Pose7 x = makePose({1, 2, 3}, {0.3, -0.2, 0.5});
    const double zero[6] = {0, 0, 0, 0, 0, 0};
    Pose7 y;
    ASSERT_TRUE(m.Plus(x.v, zero, y.v));
    for (int i = 0; i < 7; ++i) EXPECT_NEAR(y.v[i], x.v[i], 1e-12);
}

TEST(PoseManifold, MinusInvertsPlus) {
    PoseManifold m;
    const Pose7 x = makePose({1, 2, 3}, {0.3, -0.2, 0.5});
    const double delta[6] = {0.1, -0.2, 0.3, 0.05, -0.04, 0.02};
    Pose7 y;
    ASSERT_TRUE(m.Plus(x.v, delta, y.v));
    double back[6];
    ASSERT_TRUE(m.Minus(y.v, x.v, back));
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(back[i], delta[i], 1e-9) << i;
}

TEST(PoseManifold, PlusKeepsUnitQuaternionAndRightMultiplies) {
    PoseManifold m;
    const Pose7 x = makePose({0, 0, 0}, {0.5, 0.1, -0.7});
    const double delta[6] = {0, 0, 0, 0.2, -0.1, 0.3};
    Pose7 y;
    ASSERT_TRUE(m.Plus(x.v, delta, y.v));
    const Eigen::Quaterniond qy(y.v[6], y.v[3], y.v[4], y.v[5]);
    EXPECT_NEAR(qy.norm(), 1.0, 1e-12);

    const Eigen::Quaterniond qx(x.v[6], x.v[3], x.v[4], x.v[5]);
    const Eigen::Quaterniond expect = qx * Rotation::rotvec2quaternion(Eigen::Vector3d(0.2, -0.1, 0.3));
    EXPECT_LT((qy.coeffs() - expect.coeffs()).norm(), 1e-12);
}

TEST(PoseManifold, PlusJacobianLayout) {
    PoseManifold m;
    const Pose7 x = makePose({0, 0, 0}, {0.1, 0.2, 0.3});
    Eigen::Matrix<double, 7, 6, Eigen::RowMajor> J;
    ASSERT_TRUE(m.PlusJacobian(x.v, J.data()));
    EXPECT_LT((J.topRows<6>() - Eigen::Matrix<double, 6, 6>::Identity()).norm(), 1e-15);
    EXPECT_EQ(J.row(6).norm(), 0.0);
}

}  // namespace
}  // namespace ms_fusion
