#include "ms_fusion/factors/pose_mainfold.h"
#include "ms_fusion/common/rotation.h"

namespace ms_fusion {

int PoseManifold::AmbientSize() const {
    return 7;
}

int PoseManifold::TangentSize() const {
    return 6;
}

bool PoseManifold::Plus(const double* x, const double* delta, double* x_plus_delta) const {
    Eigen::Map<const Eigen::Vector3d> p_x(x);
    Eigen::Map<const Eigen::Quaterniond> q_x(x + 3);

    Eigen::Map<const Eigen::Vector3d> p_delta(delta);

    Eigen::Quaterniond q_delta =
        Rotation::rotvec2quaternion(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p_x_plus(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q_x_plus(x_plus_delta + 3);

    p_x_plus = p_x + p_delta;
    q_x_plus = (q_x * q_delta).normalized();

    return true;
}

bool PoseManifold::PlusJacobian(const double*, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jaco(jacobian);

    jaco.topRows<6>().setIdentity();
    jaco.bottomRows<1>().setZero();

    return true;
}

bool PoseManifold::Minus(const double* y, const double* x, double* y_minus_x) const {
    Eigen::Map<const Eigen::Vector3d> p_y(y);
    Eigen::Map<const Eigen::Quaterniond> q_y(y + 3);

    Eigen::Map<const Eigen::Vector3d> p_x(x);
    Eigen::Map<const Eigen::Quaterniond> q_x(x + 3);

    Eigen::Map<Eigen::Vector3d> p_y_minus_x(y_minus_x);
    Eigen::Map<Eigen::Vector3d> q_y_minus_x(y_minus_x + 3);

    p_y_minus_x = p_y - p_x;
    q_y_minus_x = Rotation::quaternion2rotvec((q_x.inverse() * q_y).normalized());

    return true;
}

bool PoseManifold::MinusJacobian(const double*, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jaco(jacobian);

    jaco.rightCols<6>().setIdentity();
    jaco.leftCols<1>().setZero();

    return true;
}

} // namespace ms_fusion
