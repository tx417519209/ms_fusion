#ifndef MS_FUSION_MSF_IVPS_GRABBER_H
#define MS_FUSION_MSF_IVPS_GRABBER_H

#ifdef MSF_USE_AXERA_IVPS

#include "ms_fusion/common/ivps_frame_handle.h"
#include "ms_fusion/frame_synchronizer.h"
#include "ms_fusion/raw_msf_frame.h"
#include "ms_fusion_message.h"

#include <ax_ivps_api.h>
#include <ax_sys_api.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include <opencv2/core/mat.hpp>

namespace ms_fusion {

/**
 * @brief IVPS 取流类：左右两路各由一个独立线程常驻抓取 NV12 原始帧，互不等待、
 *        互不阻塞；抓图结果送进 FrameSynchronizer 做时间戳配对，配对成功后按需
 *        转换成 YUV400（tracking，始终需要）和 RGB888（reloc，仅
 *        MsFusionManager::needRelocColorImages() 为真时才转换），通过回调交给
 *        上层。reloc 图像现在和 tracking 图像来自同一次同步配对，天然对齐到同
 *        一关键帧，不再需要单独的抓图线程/同步器/时间窗口缓存补偿。
 */
class MsfIvpsGrabber {
    using RawSynchronizer = FrameSynchronizer<RawMsfFrame, RawMsfFrame>;

public:
    using ImageCallback = std::function<void(ImageMessage)>;

    MsfIvpsGrabber() = default;
    ~MsfIvpsGrabber();

    MsfIvpsGrabber(const MsfIvpsGrabber&) = delete;
    MsfIvpsGrabber& operator=(const MsfIvpsGrabber&) = delete;

    // ## 须在 start() 之前调用
    void setImageCallback(ImageCallback cb);

    // ## 启动左右两路独立抓图线程；已经启动过则直接返回
    void start();
    // ## 停止并 join 两路抓图线程；可重复调用
    void stop();

private:
    // ## 单路抓图线程主体：只负责这一路 grp/chn 的 GetChnFrame（NV12），抓到后喂给 sync
    void channelGrabLoop(IVPS_GRP grp, IVPS_CHN chn, RawSynchronizer* sync, bool is_left,
                        const char* tag);

    void onPairMatched(const RawMsfFramePtr& left, const RawMsfFramePtr& right);

    // ## NV12 -> YUV400：NV12 的 Y 平面与 YUV400 内存布局完全一致，只需要把 Y 平面
    // ## 硬件拷贝到新申请的独立物理内存、格式标成 YUV400，不需要真正的色彩空间转换
    static IvpsFramePtr convertToGray(const RawMsfFramePtr& raw);
    // ## NV12 -> RGB888：走 IVPS 硬件 CSC（AX_IVPS_CscVgp）做色彩空间转换，不缩放
    static IvpsFramePtr convertToRgb(const RawMsfFramePtr& raw);

private:
    std::thread capture_left_thread_;
    std::thread capture_right_thread_;

    std::atomic<bool> stop_requested_{false};

    bool is_grabbing_ = false;

    std::unique_ptr<RawSynchronizer> sync_;

    std::mutex callback_mutex_;
    ImageCallback image_callback_;
};

} // namespace ms_fusion

#endif // MSF_USE_AXERA_IVPS

#endif // MS_FUSION_MSF_IVPS_GRABBER_H
