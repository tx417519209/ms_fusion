#ifndef MS_FUSION_PREINTEGRATION_H
#define MS_FUSION_PREINTEGRATION_H

#include "ms_fusion/preintegration/preintegration_normal.h"
#include "ms_fusion/preintegration/preintegration_odom.h"

namespace ms_fusion {

class Preintegration {
public:
    Preintegration() = default;

    // 静态函数，构造预积分指针
    static std::shared_ptr<PreintegrationBase> createPreintegration(
        const std::shared_ptr<IntegrationParameters>& parameters, 
        const ImuMeasurement& imu0, const IntegrationState& state) {
        // 设置基类的指针对象
        std::shared_ptr<PreintegrationBase> preintegration = nullptr;

        // 构造预积分对象
        preintegration = 
            std::make_shared<PreintegrationNormal>(parameters, imu0, state);

        // 返回智能指针
        return preintegration;
    }

    // 静态函数，构造里程计预积分指针
    static std::shared_ptr<PreintegrationOdom> createOdomPreintegration(
        const std::shared_ptr<OdomIntegrationParameters>& parameters, const OdomMeasurement& odom0) {
        // 把里程计预积分指针设置为空
        std::shared_ptr<PreintegrationOdom> odom_preintegration = nullptr;

        // 构造预积分对象
        odom_preintegration = std::make_shared<PreintegrationOdom>(parameters, odom0);

        // 返回指针
        return odom_preintegration;
    }

    // 静态函数，拿到位姿状态的维度
    static int numPoseParameter() {
        return  PreintegrationBase::NUM_POSE;
    }

    // 静态函数，拿到MIX参数的维度
    static int numMixParameter() {
        return PreintegrationNormal::NUM_MIX;
    }

    // 静态函数，状态转数组
    static IntegrationStateData stateToData(const IntegrationState& state) {
        return PreintegrationNormal::stateToData(state);
    }

    // 静态函数，数组转状态
    static IntegrationState stateFromData(const IntegrationStateData& data) {
        return PreintegrationNormal::stateFromData(data);
    }
};

} // namespace ms_fusion

#endif // MS_FUSION_PREINTEGRATION_H