#ifndef MS_FUSION_RAW_MSF_FRAME_H
#define MS_FUSION_RAW_MSF_FRAME_H

#ifdef MSF_USE_AXERA_IVPS

#include <ax_ivps_api.h>
#include <ax_sys_api.h>

#include <cstdint>
#include <memory>

namespace ms_fusion {

/**
 * @brief 单路 IVPS 通道帧的 RAII 包装：只持有 AX_IVPS_GetChnFrame 拿到的原始帧，
 *        析构时（尚未被显式还给硬件的情况下）自动 AX_IVPS_ReleaseChnFrame。
 *
 * 用于喂给 FrameSynchronizer：一帧要么被某次匹配消费（mmap+转换完，调用方显式
 * Munmap 后随 shared_ptr 出作用域自动释放），要么因为一直等不到配对/队列超限
 * 被丢弃（同样靠引用计数归零自动释放），不需要额外的手动释放路径。
 */
class RawMsfFrame {
public:
    RawMsfFrame(IVPS_GRP grp, IVPS_CHN chn, const AX_VIDEO_FRAME_T& frame)
        : grp_(grp), chn_(chn), frame_(frame)
    {
    }

    ~RawMsfFrame() { release(); }

    RawMsfFrame(const RawMsfFrame&) = delete;
    RawMsfFrame& operator=(const RawMsfFrame&) = delete;

    const AX_VIDEO_FRAME_T& frame() const { return frame_; }
    // ## mmap 需要往 frame 里写 u64VirAddr[0]，因此提供可写引用
    AX_VIDEO_FRAME_T& mutableFrame() { return frame_; }
    int64_t ptsUs() const { return static_cast<int64_t>(frame_.u64PTS); }

private:
    void release()
    {
        if (frame_.u64PhyAddr[0] != 0) {
            AX_IVPS_ReleaseChnFrame(grp_, chn_, &frame_);
            frame_.u64PhyAddr[0] = 0;
        }
    }

    IVPS_GRP grp_;
    IVPS_CHN chn_;
    AX_VIDEO_FRAME_T frame_;
};

using RawMsfFramePtr = std::shared_ptr<RawMsfFrame>;

} // namespace ms_fusion

#endif // MSF_USE_AXERA_IVPS

#endif // MS_FUSION_RAW_MSF_FRAME_H
