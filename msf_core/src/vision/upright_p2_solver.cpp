#include "ms_fusion/vision/upright_p2_solver.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace ms_fusion {

namespace {

Eigen::Matrix3d rotZ(double yaw) {
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    Eigen::Matrix3d R;
    R << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
    return R;
}

double normalizeAngle(double a) {
    a = std::remainder(a, 2.0 * M_PI);
    return a <= -M_PI ? a + 2.0 * M_PI : a;
}

}  // namespace

const char* uprightP2StatusName(UprightP2Status status) {
    switch (status) {
        case UprightP2Status::kOk:
            return "ok";
        case UprightP2Status::kParallelRays:
            return "parallel_rays";
        case UprightP2Status::kVerticalOnlySeparation:
            return "vertical_only_separation";
        case UprightP2Status::kYawUnobservable:
            return "yaw_unobservable";
        case UprightP2Status::kNoRealRoot:
            return "no_real_root";
        case UprightP2Status::kNoPositiveDepth:
            return "no_positive_depth";
        case UprightP2Status::kNonFinite:
        default:
            return "non_finite";
    }
}

UprightP2Status solveUprightP2(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2,
                               const Eigen::Vector3d& c1, const Eigen::Vector3d& d1,
                               const Eigen::Vector3d& c2, const Eigen::Vector3d& d2,
                               const UprightP2Options& options,
                               std::vector<UprightP2Solution>& solutions,
                               UprightP2Diagnostics* diagnostics) {
    solutions.clear();
    UprightP2Diagnostics diag;
    const auto finish = [&](UprightP2Status status) {
        if (diagnostics) {
            *diagnostics = diag;
        }
        if (status != UprightP2Status::kOk) {
            solutions.clear();
        }
        return status;
    };

    if (!P1.allFinite() || !P2.allFinite() || !c1.allFinite() || !c2.allFinite() ||
        !d1.allFinite() || !d2.allFinite()) {
        return finish(UprightP2Status::kNonFinite);
    }

    const Eigen::Vector3d n = d1.cross(d2);
    const Eigen::Vector3d dP = P1 - P2;
    const Eigen::Vector3d dc = c1 - c2;
    diag.sin_ray = n.norm();
    if (diag.sin_ray < options.min_sin_ray) {
        return finish(UprightP2Status::kParallelRays);
    }
    if (dP.head<2>().norm() < options.min_horizontal_separation_m) {
        return finish(UprightP2Status::kVerticalOnlySeparation);
    }

    const double A = n.x() * dP.x() + n.y() * dP.y();
    const double B = n.x() * dP.y() - n.y() * dP.x();
    const double C = n.z() * dP.z() - n.dot(dc);
    const double AB = std::hypot(A, B);
    diag.yaw_observability = AB / (diag.sin_ray * dP.norm());
    if (!std::isfinite(diag.yaw_observability) ||
        diag.yaw_observability < options.min_yaw_observability) {
        return finish(UprightP2Status::kYawUnobservable);
    }

    double r = -C / AB;
    diag.r = r;
    if (!std::isfinite(r)) {
        return finish(UprightP2Status::kNonFinite);
    }
    if (std::abs(r) > 1.0 + options.eps_clip) {
        return finish(UprightP2Status::kNoRealRoot);
    }
    bool tangent = false;
    if (std::abs(r) > 1.0) {
        r = std::copysign(1.0, r);
        tangent = true;
        diag.clipped = true;
    }

    // A cos + B sin = AB cos(yaw - base)，故 yaw = base ± acos(r)
    const double base = std::atan2(B, A);
    const double off = std::acos(r);
    double roots[2] = {base + off, base - off};
    // r = ±1 时两根重合（r = -1 时 base±π 归一化后是同一个角），只取一个
    const int root_count = (tangent || off == 0.0 || off == M_PI) ? 1 : 2;
    diag.roots = root_count;

    Eigen::Matrix<double, 3, 2> M;
    M.col(0) = d1;
    M.col(1) = -d2;
    const Eigen::Matrix2d MtM = M.transpose() * M;  // det = ||d1×d2||^2 > 0（单位视线）
    for (int k = 0; k < root_count; ++k) {
        const double yaw = normalizeAngle(roots[k]);
        const Eigen::Matrix3d R = rotZ(yaw);
        const Eigen::Vector3d rhs = R.transpose() * dP - dc;
        const Eigen::Vector2d lambda = MtM.ldlt().solve(M.transpose() * rhs);
        if (!lambda.allFinite() || lambda.x() <= 0.0 || lambda.y() <= 0.0) {
            continue;
        }
        // 两个地标各给出一个平移估计，取平均（无噪声时二者相同）
        const Eigen::Vector3d t1 = P1 - R * (c1 + lambda.x() * d1);
        const Eigen::Vector3d t2 = P2 - R * (c2 + lambda.y() * d2);
        const Eigen::Vector3d t = 0.5 * (t1 + t2);
        if (!t.allFinite()) {
            continue;
        }
        UprightP2Solution solution;
        solution.yaw_rad = yaw;
        solution.t = t;
        solution.depths = lambda;
        solution.tangent = tangent;
        solutions.push_back(solution);
    }
    if (solutions.empty()) {
        return finish(UprightP2Status::kNoPositiveDepth);
    }
    return finish(UprightP2Status::kOk);
}

}  // namespace ms_fusion
