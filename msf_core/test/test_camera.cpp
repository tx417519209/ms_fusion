// pinhole-radtan4 / radtan5 相机模型单测：project / unproject 往返一致性。
#include "ms_fusion/camera/pinhole_radtan4_camera.h"
#include "ms_fusion/camera/pinhole_radtan5_camera.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>

namespace ms_fusion {
namespace {

// EuRoC cam0 一类的典型内参、畸变
const Eigen::Vector4d kIntrinsic(458.654, 457.296, 367.215, 248.375);

TEST(PinholeRadtan4Camera, ZeroDistortionIsPlainPinhole) {
    PinholeRadtan4Camera cam(kIntrinsic, Eigen::Vector4d::Zero());
    Eigen::Vector2d px;
    cam.project(Eigen::Vector3d(0.2, -0.1, 1.0), px);
    EXPECT_NEAR(px[0], kIntrinsic[0] * 0.2 + kIntrinsic[2], 1e-9);
    EXPECT_NEAR(px[1], kIntrinsic[1] * -0.1 + kIntrinsic[3], 1e-9);
}

TEST(PinholeRadtan4Camera, ProjectScalesWithDepth) {
    // 输入是相机系 3D 点，除以 z 后才进入模型
    PinholeRadtan4Camera cam(kIntrinsic, Eigen::Vector4d(-0.28, 0.07, 1.9e-4, 1.8e-5));
    Eigen::Vector2d a, b;
    cam.project(Eigen::Vector3d(0.3, 0.2, 1.0), a);
    cam.project(Eigen::Vector3d(0.9, 0.6, 3.0), b);
    EXPECT_LT((a - b).norm(), 1e-9);
}

TEST(PinholeRadtan4Camera, PrincipalPointMapsToAxis) {
    PinholeRadtan4Camera cam(kIntrinsic, Eigen::Vector4d(-0.28, 0.07, 1.9e-4, 1.8e-5));
    Eigen::Vector3d p;
    cam.unproject(Eigen::Vector2d(kIntrinsic[2], kIntrinsic[3]), p);
    EXPECT_NEAR(p[0], 0.0, 1e-12);
    EXPECT_NEAR(p[1], 0.0, 1e-12);
    EXPECT_NEAR(p[2], 1.0, 1e-12);
}

// unproject 用固定 10 次不动点迭代去畸变，不是精确反解：畸变大的图像边角处会留下残差。
// 实测 k1=-0.28 时角点约 0.03 px，对 VIO（亚像素~像素级噪声）可忽略；画面中部应当近乎精确。
TEST(PinholeRadtan4Camera, ProjectUnprojectRoundTrip) {
    PinholeRadtan4Camera cam(kIntrinsic, Eigen::Vector4d(-0.28, 0.07, 1.9e-4, 1.8e-5));
    for (double u = 20; u < 740; u += 90) {
        for (double v = 20; v < 460; v += 70) {
            Eigen::Vector3d ray;
            cam.unproject(Eigen::Vector2d(u, v), ray);
            EXPECT_DOUBLE_EQ(ray[2], 1.0);
            Eigen::Vector2d back;
            cam.project(ray, back);
            const bool central = std::abs(u - kIntrinsic[2]) < 150 && std::abs(v - kIntrinsic[3]) < 100;
            EXPECT_LT((back - Eigen::Vector2d(u, v)).norm(), central ? 1e-4 : 0.05) << u << "," << v;
        }
    }
}

TEST(PinholeRadtan4Camera, UnprojectProjectRoundTripInNormalizedPlane) {
    PinholeRadtan4Camera cam(kIntrinsic, Eigen::Vector4d(-0.28, 0.07, 1.9e-4, 1.8e-5));
    const Eigen::Vector3d pts[] = {{0.1, 0.05, 1.0}, {-0.4, 0.3, 1.0}, {0.5, -0.35, 1.0}};
    for (const auto& p : pts) {
        Eigen::Vector2d px;
        cam.project(p, px);
        Eigen::Vector3d q;
        cam.unproject(px, q);
        EXPECT_LT((q - p).norm(), 1e-6) << p.transpose();
    }
}

TEST(PinholeRadtan4Camera, UnprojectsBatchMatchesSingle) {
    PinholeRadtan4Camera cam(kIntrinsic, Eigen::Vector4d(-0.28, 0.07, 1.9e-4, 1.8e-5));
    std::vector<cv::Point2f> px = {{100.f, 80.f}, {400.f, 300.f}, {700.f, 450.f}};
    Eigen::aligned_vector<Eigen::Vector3d> rays;
    cam.unprojects(px, rays);
    ASSERT_EQ(rays.size(), px.size());
    for (size_t i = 0; i < px.size(); ++i) {
        Eigen::Vector3d r;
        cam.unproject(Eigen::Vector2d(px[i].x, px[i].y), r);
        EXPECT_LT((rays[i] - r).norm(), 1e-12);
    }
}

TEST(PinholeRadtan4Camera, FocalLengthIsMean) {
    PinholeRadtan4Camera cam(kIntrinsic, Eigen::Vector4d::Zero());
    EXPECT_NEAR(cam.getFocalLength(), (kIntrinsic[0] + kIntrinsic[1]) / 2.0, 1e-12);
}

TEST(PinholeRadtan5Camera, ZeroDistortionIsPlainPinhole) {
    PinholeRadtan5Camera cam(kIntrinsic, Eigen::Matrix<double, 5, 1>::Zero());
    Eigen::Vector2d px;
    cam.project(Eigen::Vector3d(0.2, -0.1, 1.0), px);
    EXPECT_NEAR(px[0], kIntrinsic[0] * 0.2 + kIntrinsic[2], 1e-9);
    EXPECT_NEAR(px[1], kIntrinsic[1] * -0.1 + kIntrinsic[3], 1e-9);
}

TEST(PinholeRadtan5Camera, ProjectUnprojectRoundTrip) {
    Eigen::Matrix<double, 5, 1> d;
    d << -0.28, 0.07, 1.9e-4, 1.8e-5, -0.001;  // k1, k2, p1, p2, k3（顺序以实现为准）
    PinholeRadtan5Camera cam(kIntrinsic, d);
    for (double u = 20; u < 740; u += 90) {
        for (double v = 20; v < 460; v += 70) {
            Eigen::Vector3d ray;
            cam.unproject(Eigen::Vector2d(u, v), ray);
            Eigen::Vector2d back;
            cam.project(ray, back);
            const bool central = std::abs(u - kIntrinsic[2]) < 150 && std::abs(v - kIntrinsic[3]) < 100;
            EXPECT_LT((back - Eigen::Vector2d(u, v)).norm(), central ? 1e-4 : 0.05) << u << "," << v;
        }
    }
}

TEST(PinholeRadtan5Camera, FifthCoefficientChangesProjection) {
    Eigen::Matrix<double, 5, 1> d0, d1;
    d0 << -0.28, 0.07, 0, 0, 0;
    d1 << -0.28, 0.07, 0, 0, 0.01;
    PinholeRadtan5Camera c0(kIntrinsic, d0), c1(kIntrinsic, d1);
    Eigen::Vector2d a, b;
    c0.project(Eigen::Vector3d(0.5, 0.4, 1.0), a);
    c1.project(Eigen::Vector3d(0.5, 0.4, 1.0), b);
    EXPECT_GT((a - b).norm(), 1e-3);
}

}  // namespace
}  // namespace ms_fusion
