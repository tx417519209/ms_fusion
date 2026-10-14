#ifndef MARGINALIZATION_FACTOR_H
#define MARGINALIZATION_FACTOR_H

#include "ms_fusion/factors/marginalization_info.h"

#include <ceres/ceres.h>

#include <memory>

namespace ms_fusion {

class MarginalizationFactor : public ceres::CostFunction
{
public:
    MarginalizationFactor() = delete;

    explicit MarginalizationFactor(std::shared_ptr<MarginalizationInfo> marg_info);

    bool Evaluate(const double* const* parameters, double* residuals,
                  double** jacobians) const override;

private:
    std::shared_ptr<MarginalizationInfo> marg_info_;
};

} // namespace ms_fusion

#endif // MARGINALIZATION_FACTOR_H