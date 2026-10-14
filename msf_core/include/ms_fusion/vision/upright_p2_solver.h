#ifndef MS_FUSION_VISION_UPRIGHT_P2_SOLVER_H
#define MS_FUSION_VISION_UPRIGHT_P2_SOLVER_H

#include <Eigen/Core>

#include <vector>

namespace ms_fusion {

// 已知竖直方向（roll=pitch=0）的广义相机 2 点绝对位姿解析解。
//
// 模型：P_i = Rz(yaw) * (c_i + λ_i * d_i) + t，i = 1, 2
//   P_i：地图地标（世界系）
//   c_i：观测该地标的相机光心（机体/虚拟 IMU 系）
//   d_i：单位视线（机体系，= R_imu_from_cam * 相机系视线）
//   λ_i：深度（必须为正）
// 输出 T_w_from_body = (Rz(yaw), t)。
//
// 两式相减并左乘 Rz(-yaw)：Rz(-yaw)ΔP - Δc = λ1 d1 - λ2 d2，再点乘
// n = d1 × d2 消去深度：A cos(yaw) + B sin(yaw) + C = 0，其中
//   A = n_x ΔP_x + n_y ΔP_y，B = n_x ΔP_y - n_y ΔP_x，C = n_z ΔP_z - n·Δc。
// 记 r = -C / hypot(A, B)：|r| <= 1 时最多两个根；1 < |r| <= 1 + eps_clip
// 时截到 ±1 得到一个相切根（噪声把本应相切/接近相切的解推到无实根）；
// 更大则无解。得到 yaw 后用 3x2 最小二乘求两个深度，再反算平移。
enum class UprightP2Status {
    kOk = 0,
    kParallelRays,            // ||d1 × d2|| 过小，两条视线近平行
    kVerticalOnlySeparation,  // 两个地标水平间距过小（只有竖直差），yaw 不可观
    kYawUnobservable,         // hypot(A, B) 数值上为 0
    kNoRealRoot,              // |r| > 1 + eps_clip
    kNoPositiveDepth,         // 所有根的深度都不全为正
    kNonFinite,               // 输入或中间量非有限
};

const char* uprightP2StatusName(UprightP2Status status);

struct UprightP2Options {
    // 合成数据（网球场布局、0.5~2px 噪声）上：截断容差对整帧命中率几乎无影响，
    // 0.1 时单个点对层面正确对应多保留约 9%，错误对应只多保留约 1%。
    double eps_clip = 0.1;
    // 只拒绝数值上接近奇异的样本；可观性差但非奇异的点对保留，交给全局评分淘汰。
    double min_sin_ray = 1e-6;
    double min_horizontal_separation_m = 1e-3;
    double min_yaw_observability = 1e-9;
};

struct UprightP2Solution {
    double yaw_rad = 0.0;  // 归一化到 (-pi, pi]
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector2d depths = Eigen::Vector2d::Zero();
    bool tangent = false;  // 由 |r| 截断得到的相切根
};

struct UprightP2Diagnostics {
    double sin_ray = 0.0;            // ||d1 × d2||
    double yaw_observability = 0.0;  // hypot(A, B) / (||d1 × d2|| · ||ΔP||)
    double r = 0.0;                  // -C / hypot(A, B)，截断前
    bool clipped = false;
    int roots = 0;                   // 深度检查前的实根数（0/1/2）
};

// 返回 kOk 时 solutions 至少有一个解；其他状态 solutions 为空。
UprightP2Status solveUprightP2(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2,
                               const Eigen::Vector3d& c1, const Eigen::Vector3d& d1,
                               const Eigen::Vector3d& c2, const Eigen::Vector3d& d2,
                               const UprightP2Options& options,
                               std::vector<UprightP2Solution>& solutions,
                               UprightP2Diagnostics* diagnostics = nullptr);

}  // namespace ms_fusion

#endif  // MS_FUSION_VISION_UPRIGHT_P2_SOLVER_H
