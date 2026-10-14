// 直立 2 点绝对位姿解析解（upright_p2_solver）单测：只依赖 Eigen 和 gtest。
#include "ms_fusion/vision/upright_p2_solver.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <vector>

namespace ms_fusion {
namespace {

struct GoldenCase {
    double P1[3], P2[3], c1[3], d1[3], c2[3], d2[3];
    double yaw;
    double t[3];
    int num_roots;
};

// 由 scratchpad/golden.py 生成：网球场真实地标坐标、双目外参近似、无噪声视线；
// 真值 (yaw, t) 必须出现在解集里，num_roots 为参考实现给出的正深度根数。
const GoldenCase kGoldenCases[] = {
    {{0, -6.399, -1.0700000000000001}, {-6.4000000000000004, 0, 0}, {0.20000000000000001, 0.19, -0.47157392547017452}, {0.85795801156132112, -0.51174759510557766, -0.044971649974196233}, {0.20000000000000001, 0.19, -0.47157392547017452}, {0.99627210528745314, 0.018505046066025283, 0.08425826663420119}, 0.083975219669834608, {-12.130926395426059, -0.77699683735171199, 0}, 2},
    {{0, 4.1150000000000002, -0.97899999999999998}, {0, -4.1150000000000002, -0.97899999999999998}, {0.20000000000000001, -0.19, -0.47094127672921948}, {0.97655240918714725, 0.21154612290340757, -0.039920295531406996}, {0.20000000000000001, 0.19, -0.47094127672921948}, {0.91281522738786181, -0.40685781907693958, -0.035143643874849313}, -0.093431071006975244, {-12.806788930577722, 2.8017732009743437, 0}, 1},
    {{-6.4000000000000004, -4.1150000000000002, 0}, {-6.4000000000000004, 4.1150000000000002, 0}, {0.20000000000000001, -0.19, -0.70109283883163465}, {0.7036688903076852, -0.69903504242902337, 0.12727962236510715}, {0.20000000000000001, -0.19, -0.70109283883163465}, {0.70026117311085245, 0.70485904243883246, 0.11317252195486741}, 0.05617039301434152, {-10.696418966497721, -0.30971672759938063, 0}, 1},
    {{-6.4000000000000004, 4.1150000000000002, 0}, {0, 4.1150000000000002, 0}, {0.20000000000000001, 0.19, -0.61032673733809939}, {0.74459464669553999, 0.66350243728735858, 0.07309806992034025}, {0.20000000000000001, -0.19, -0.61032673733809939}, {0.94027358694764784, 0.33728437034681458, 0.046095934835480662}, 0.22920417904205723, {-11.347277518109617, -2.9229571057991608, 0}, 1},
    {{0, 5.4850000000000003, -1.0289999999999999}, {0, 4.1150000000000002, 0}, {0.20000000000000001, 0.19, -0.47815903994862163}, {0.99908337948833936, 0.015236493255338128, -0.040003126170826839}, {0.20000000000000001, 0.19, -0.47815903994862163}, {0.99639389340102436, -0.076705703805498279, 0.036269052880538125}, 0.47061295784491841, {-12.258747376197421, -1.2000509775291011, 0}, 2},
    {{0, -6.399, -1.0700000000000001}, {0, -4.1150000000000002, -0.97899999999999998}, {0.20000000000000001, -0.19, -0.64268387067399169}, {0.63689389438468313, -0.77012749396725444, -0.03563439814601406}, {0.20000000000000001, 0.19, -0.64268387067399169}, {0.74457960576139381, -0.66685036337097048, -0.030196085114002715}, 0.2910940400166272, {-10.212720836081406, 0.3802642470152735, 0}, 2},
    {{0, -5.4850000000000003, -1.0289999999999999}, {0, -4.1150000000000002, 0}, {0.20000000000000001, -0.19, -0.52773672156074503}, {0.88676409873214779, -0.4606401677092401, -0.038210850455921097}, {0.20000000000000001, 0.19, -0.52773672156074503}, {0.91878682564961134, -0.39263623456797025, 0.040835723539525186}, 0.17681479637887576, {-12.744718690950521, -1.4306817221292001, 0}, 2},
    {{0, -6.399, 0}, {0, 6.399, -1.0700000000000001}, {0.20000000000000001, -0.19, -0.63201138120822253}, {0.6991393842601531, -0.71324562174999551, 0.049847812698216885}, {0.20000000000000001, -0.19, -0.63201138120822253}, {0.96278173879246376, 0.26800342578942926, -0.035006959492219686}, 0.25123244418519758, {-11.075037366566221, 0.29089838774778176, 0}, 2},
    {{0, 4.1150000000000002, -0.97899999999999998}, {0, 6.399, 0}, {0.20000000000000001, -0.19, -0.56807730396825362}, {-0.97775835231618347, -0.20720840669047819, -0.032454285891621817}, {0.20000000000000001, 0.19, -0.56807730396825362}, {-0.91936190910094884, -0.39116144616348142, 0.042028598948247065}, -3.1212166551491314, {-12.120116350375321, 1.0538353370977385, 0}, 2},
    {{-6.4000000000000004, -4.1150000000000002, 0}, {0, 4.1150000000000002, -0.97899999999999998}, {0.20000000000000001, -0.19, -0.60775368845773425}, {-0.94736856164753303, 0.30436967642469054, 0.099256780498934535}, {0.20000000000000001, -0.19, -0.60775368845773425}, {-0.88612712659490456, -0.46265684110161603, -0.026969666192446114}, 3.1412343782507466, {-12.000179910355225, -2.4393243990322917, 0}, 1},
    {{0, -4.1150000000000002, -0.97899999999999998}, {0, 6.399, -1.0700000000000001}, {0.20000000000000001, 0.19, -0.6920258311276335}, {-0.96133447591692156, 0.27441191910753926, -0.023111124275017738}, {0.20000000000000001, 0.19, -0.6920258311276335}, {-0.85864895897463611, -0.5118406703646301, -0.027223030919782887}, 3.1401427876698613, {-11.731800966542204, -0.50057422610982094, 0}, 1},
    {{0, 0, 0}, {-6.4000000000000004, 0, 0}, {0.20000000000000001, -0.19, -0.46410972506722792}, {-0.9975942821467404, -0.061075667764209092, 0.032793460251788774}, {0.20000000000000001, -0.19, -0.46410972506722792}, {-0.99093841879400124, -0.12037886029421839, 0.059581709880712694}, -3.1301372193129153, {-13.905470285312896, -1.2137432614748314, 0}, 1},
};

Eigen::Vector3d vec(const double v[3]) { return Eigen::Vector3d(v[0], v[1], v[2]); }

Eigen::Matrix3d rotZ(double yaw) {
    return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

double angleDiff(double a, double b) { return std::abs(std::remainder(a - b, 2.0 * M_PI)); }

// 世界系"前右下"，网柱点 z 为负；左右目光心在机体系 (0.2, ∓0.19, -0.5)
const Eigen::Vector3d kCamLeft(0.2, -0.19, -0.5);
const Eigen::Vector3d kCamRight(0.2, 0.19, -0.5);

struct PairInput {
    Eigen::Vector3d P1, P2, c1, d1, c2, d2;
};

PairInput makePair(double yaw, const Eigen::Vector3d& t, const Eigen::Vector3d& P1,
                   const Eigen::Vector3d& P2, const Eigen::Vector3d& c1,
                   const Eigen::Vector3d& c2) {
    const Eigen::Matrix3d R = rotZ(yaw);
    PairInput in{P1, P2, c1, (R.transpose() * (P1 - t) - c1).normalized(), c2,
                 (R.transpose() * (P2 - t) - c2).normalized()};
    return in;
}

bool containsTruth(const std::vector<UprightP2Solution>& solutions, double yaw,
                   const Eigen::Vector3d& t, double tol) {
    for (const auto& s : solutions) {
        if (angleDiff(s.yaw_rad, yaw) < tol && (s.t - t).norm() < tol) {
            return true;
        }
    }
    return false;
}

UprightP2Status solve(const PairInput& in, std::vector<UprightP2Solution>& solutions,
                      UprightP2Diagnostics* diag = nullptr,
                      const UprightP2Options& options = UprightP2Options()) {
    return solveUprightP2(in.P1, in.P2, in.c1, in.d1, in.c2, in.d2, options, solutions, diag);
}

TEST(UprightP2Solver, GoldenCasesFromPythonReference) {
    for (const auto& gc : kGoldenCases) {
        std::vector<UprightP2Solution> solutions;
        const auto status = solveUprightP2(vec(gc.P1), vec(gc.P2), vec(gc.c1), vec(gc.d1),
                                           vec(gc.c2), vec(gc.d2), UprightP2Options(), solutions);
        ASSERT_EQ(status, UprightP2Status::kOk);
        EXPECT_EQ(static_cast<int>(solutions.size()), gc.num_roots);
        EXPECT_TRUE(containsTruth(solutions, gc.yaw, vec(gc.t), 1e-8));
        for (const auto& s : solutions) {
            EXPECT_GT(s.yaw_rad, -M_PI);
            EXPECT_LE(s.yaw_rad, M_PI);
        }
    }
}

TEST(UprightP2Solver, SameCameraAndCrossCamera) {
    const Eigen::Vector3d t(-12.0, 1.0, 0.0);
    const double yaw = 0.2;
    const Eigen::Vector3d P1(0.0, -4.115, 0.0), P2(-6.4, 4.115, 0.0);
    for (const auto& cams : {std::make_pair(kCamLeft, kCamLeft),
                             std::make_pair(kCamLeft, kCamRight),
                             std::make_pair(kCamRight, kCamLeft)}) {
        std::vector<UprightP2Solution> solutions;
        ASSERT_EQ(solve(makePair(yaw, t, P1, P2, cams.first, cams.second), solutions),
                  UprightP2Status::kOk);
        EXPECT_TRUE(containsTruth(solutions, yaw, t, 1e-9));
    }
}

TEST(UprightP2Solver, GroundGroundAndGroundNet) {
    const Eigen::Vector3d t(-13.0, -0.5, 0.0);
    const double yaw = -0.15;
    const Eigen::Vector3d ground(0.0, 4.115, 0.0), ground2(-6.4, 0.0, 0.0);
    const Eigen::Vector3d net(0.0, -5.485, -1.029);
    std::vector<UprightP2Solution> solutions;
    ASSERT_EQ(solve(makePair(yaw, t, ground, ground2, kCamLeft, kCamRight), solutions),
              UprightP2Status::kOk);
    EXPECT_TRUE(containsTruth(solutions, yaw, t, 1e-9));
    ASSERT_EQ(solve(makePair(yaw, t, ground, net, kCamLeft, kCamLeft), solutions),
              UprightP2Status::kOk);
    EXPECT_TRUE(containsTruth(solutions, yaw, t, 1e-9));
}

TEST(UprightP2Solver, TwoRootsContainTruth) {
    bool found_two_root_case = false;
    for (const auto& gc : kGoldenCases) {
        if (gc.num_roots != 2) {
            continue;
        }
        found_two_root_case = true;
        std::vector<UprightP2Solution> solutions;
        ASSERT_EQ(solveUprightP2(vec(gc.P1), vec(gc.P2), vec(gc.c1), vec(gc.d1), vec(gc.c2),
                                 vec(gc.d2), UprightP2Options(), solutions),
                  UprightP2Status::kOk);
        ASSERT_EQ(solutions.size(), 2u);
        EXPECT_GT(angleDiff(solutions[0].yaw_rad, solutions[1].yaw_rad), 1e-6);
        EXPECT_TRUE(containsTruth(solutions, gc.yaw, vec(gc.t), 1e-8));
    }
    EXPECT_TRUE(found_two_root_case);
}

TEST(UprightP2Solver, YawAcrossPlusMinusPi) {
    const Eigen::Vector3d t(12.0, 0.5, 0.0);  // 从对面底线看向球网
    const Eigen::Vector3d P1(0.0, -4.115, 0.0), P2(0.0, 5.485, -1.029);
    for (const double yaw : {M_PI - 1e-3, -M_PI + 1e-3, M_PI}) {
        std::vector<UprightP2Solution> solutions;
        ASSERT_EQ(solve(makePair(yaw, t, P1, P2, kCamLeft, kCamRight), solutions),
                  UprightP2Status::kOk);
        EXPECT_TRUE(containsTruth(solutions, yaw, t, 1e-9));
    }
}

// 找一组 1 < |r| < 1 + eps 的输入：从网柱点对出发，绕竖直轴逐步扰动第二条视线
bool findNearTangentInput(PairInput& out, double& r_out) {
    const Eigen::Vector3d t(-12.0, 0.0, 0.0);
    const PairInput base = makePair(0.0, t, Eigen::Vector3d(0.0, -6.399, -1.07),
                                    Eigen::Vector3d(0.0, -4.115, 0.0), kCamLeft, kCamLeft);
    UprightP2Options wide;
    wide.eps_clip = 1e9;
    for (int axis = 0; axis < 3; ++axis) {
        for (int k = 1; k <= 2000; ++k) {
            for (const double sign : {1.0, -1.0}) {
                PairInput in = base;
                const double angle = sign * 1e-5 * k;
                in.d2 = (Eigen::AngleAxisd(angle, Eigen::Vector3d::Unit(axis)) * base.d2)
                            .normalized();
                std::vector<UprightP2Solution> solutions;
                UprightP2Diagnostics diag;
                solve(in, solutions, &diag, wide);
                if (std::abs(diag.r) > 1.0 + 1e-6 && std::abs(diag.r) < 1.05) {
                    out = in;
                    r_out = diag.r;
                    return true;
                }
            }
        }
    }
    return false;
}

TEST(UprightP2Solver, TangentClipAndNoRealRoot) {
    PairInput in;
    double r = 0.0;
    ASSERT_TRUE(findNearTangentInput(in, r));

    std::vector<UprightP2Solution> solutions;
    UprightP2Diagnostics diag;
    UprightP2Options clip;
    clip.eps_clip = 0.1;
    const auto status = solve(in, solutions, &diag, clip);
    EXPECT_TRUE(diag.clipped);
    if (status == UprightP2Status::kOk) {
        ASSERT_EQ(solutions.size(), 1u);
        EXPECT_TRUE(solutions[0].tangent);
    } else {
        // 相切根的深度不全为正时也合法，但不能是"无实根"
        EXPECT_EQ(status, UprightP2Status::kNoPositiveDepth);
    }

    UprightP2Options no_clip;
    no_clip.eps_clip = 0.0;
    EXPECT_EQ(solve(in, solutions, &diag, no_clip), UprightP2Status::kNoRealRoot);
    EXPECT_TRUE(solutions.empty());
    EXPECT_FALSE(diag.clipped);
}

TEST(UprightP2Solver, RootCountMatchesDiagnostics) {
    for (const auto& gc : kGoldenCases) {
        std::vector<UprightP2Solution> solutions;
        UprightP2Diagnostics diag;
        ASSERT_EQ(solveUprightP2(vec(gc.P1), vec(gc.P2), vec(gc.c1), vec(gc.d1), vec(gc.c2),
                                 vec(gc.d2), UprightP2Options(), solutions, &diag),
                  UprightP2Status::kOk);
        EXPECT_GE(diag.roots, static_cast<int>(solutions.size()));
        EXPECT_LE(diag.roots, 2);
    }
}

TEST(UprightP2Solver, ParallelRays) {
    PairInput in = makePair(0.1, Eigen::Vector3d(-12.0, 0.0, 0.0), Eigen::Vector3d(0, 0, 0),
                            Eigen::Vector3d(-6.4, 0, 0), kCamLeft, kCamLeft);
    in.d2 = in.d1;
    std::vector<UprightP2Solution> solutions;
    EXPECT_EQ(solve(in, solutions), UprightP2Status::kParallelRays);
    EXPECT_TRUE(solutions.empty());
}

TEST(UprightP2Solver, DuplicateAndVerticalOnlyWorldPoints) {
    const Eigen::Vector3d t(-12.0, 0.0, 0.0);
    std::vector<UprightP2Solution> solutions;
    // 同一个世界点被两目观测：两条视线不平行，但地标间距为 0
    PairInput dup = makePair(0.1, t, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0),
                             kCamLeft, kCamRight);
    EXPECT_EQ(solve(dup, solutions), UprightP2Status::kVerticalOnlySeparation);
    // 网柱顶点与正下方地面点：只有竖直差
    PairInput vertical = makePair(0.1, t, Eigen::Vector3d(0, 0, -0.914), Eigen::Vector3d(0, 0, 0),
                                  kCamLeft, kCamLeft);
    EXPECT_EQ(solve(vertical, solutions), UprightP2Status::kVerticalOnlySeparation);
    EXPECT_TRUE(solutions.empty());
}

TEST(UprightP2Solver, NonPositiveDepth) {
    // Python 参考实现搜出的确定输入：两个 yaw 实根（r=0）的深度都不全为正
    const Eigen::Vector3d c = Eigen::Vector3d::Zero();
    std::vector<UprightP2Solution> solutions;
    UprightP2Diagnostics diag;
    EXPECT_EQ(solveUprightP2(Eigen::Vector3d(10, 0, 0), Eigen::Vector3d(0, 10, 0), c,
                             Eigen::Vector3d(1, 0, 1).normalized(), c,
                             Eigen::Vector3d(1, 2, -1).normalized(), UprightP2Options(),
                             solutions, &diag),
              UprightP2Status::kNoPositiveDepth);
    EXPECT_LE(std::abs(diag.r), 1.0);
    EXPECT_TRUE(solutions.empty());

    // 两条视线同时反向时 n 不变、yaw 根相同；即使某个根恰好给出另一组合法几何，
    // 输出的每个解也必须是正深度
    PairInput in = makePair(0.1, Eigen::Vector3d(-12.0, 0.0, 0.0), Eigen::Vector3d(0, -4.115, 0),
                            Eigen::Vector3d(-6.4, 4.115, 0), kCamLeft, kCamLeft);
    in.d1 = -in.d1;
    in.d2 = -in.d2;
    solve(in, solutions);
    for (const auto& s : solutions) {
        EXPECT_GT(s.depths.x(), 0.0);
        EXPECT_GT(s.depths.y(), 0.0);
    }
    EXPECT_FALSE(containsTruth(solutions, 0.1, Eigen::Vector3d(-12.0, 0.0, 0.0), 1e-6));
}

TEST(UprightP2Solver, NonFiniteInput) {
    PairInput in = makePair(0.1, Eigen::Vector3d(-12.0, 0.0, 0.0), Eigen::Vector3d(0, -4.115, 0),
                            Eigen::Vector3d(-6.4, 4.115, 0), kCamLeft, kCamLeft);
    in.d1.x() = std::nan("");
    std::vector<UprightP2Solution> solutions;
    EXPECT_EQ(solve(in, solutions), UprightP2Status::kNonFinite);
}

}  // namespace
}  // namespace ms_fusion
