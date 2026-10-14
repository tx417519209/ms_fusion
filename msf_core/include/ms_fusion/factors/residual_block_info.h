#ifndef RESIDUAL_BLOCK_INFO_H
#define RESIDUAL_BLOCK_INFO_H

#include <ceres/ceres.h>

#include <memory>

#define POSE_LOCAL_SIZE 6
#define POSE_GLOBAL_SIZE 7

namespace ms_fusion {

// 残差块信息类 
class ResidualBlockInfo {
public:
    // 构造残差块信息
    ResidualBlockInfo(std::shared_ptr<ceres::CostFunction> cost_function,
                      std::shared_ptr<ceres::LossFunction> loss_function,
                      std::vector<double*> parameter_blocks, std::vector<int> marg_para_index);

    // 更新函数
    void Evaluate();

    // 拿到残差对应状态的雅可比矩阵
    const std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&
    jacobians() {
        return jacobians_;
    }

    // 拿到状态量对应的维度
    const std::vector<int>& parameterBlockSizes() {
        return cost_function_->parameter_block_sizes();
    }

    // 拿到状态量的指针
    const std::vector<double*>& parameterBlocks() {
        return parameter_blocks_;
    }

    // 拿到残差
    const Eigen::VectorXd& residuals() {
        return residuals_;
    }

    // 拿到要边缘化状态量的索引
    const std::vector<int>& marginalizationParametersIndex() {
        return marg_para_index_;
    }

private:
    // 残差对应的因子
    std::shared_ptr<ceres::CostFunction> cost_function_;
    // 残差的核函数
    std::shared_ptr<ceres::LossFunction> loss_function_;

    // 残差涉及到的所有状态量的指针
    std::vector<double*> parameter_blocks_;

    // 要边缘化的状态在状态量容器中的指针
    std::vector<int> marg_para_index_;

    // 残差对状态量的雅可比矩阵
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians_;

    // 残差
    Eigen::VectorXd residuals_;
};

} // namespace ms_fusion

#endif // RESIDUAL_BLOCK_INFO_H