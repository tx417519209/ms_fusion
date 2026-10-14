#ifndef MS_FUSION_POSE_MANIFOLD_H
#define MS_FUSION_POSE_MANIFOLD_H

#include <ceres/ceres.h>

namespace ms_fusion {

class PoseManifold : public ceres::Manifold {
public:
    int AmbientSize() const override;
    int TangentSize() const override;
    
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
    bool PlusJacobian(const double* x, double* jacobian) const override;
    
    bool Minus(const double* y, const double* x, double* y_minus_x) const override;
    bool MinusJacobian(const double* x, double* jacobian) const override;
};

} // namespace ms_fusion

#endif // MS_FUSION_POSE_MANIFOLD_H