#ifndef MS_FUSION_COMMON_IVPS_FRAME_HANDLE_H
#define MS_FUSION_COMMON_IVPS_FRAME_HANDLE_H

#ifdef MSF_USE_AXERA_IVPS

#include "ax_sys_api.h"
#include "ax_global_type.h"

#include <opencv2/core/mat.hpp>

#include <memory>

namespace ms_fusion {

class IvpsFrameHandle;
using IvpsFramePtr = std::shared_ptr<IvpsFrameHandle>;

/**
 * @brief 抓图流水线内部传递一帧图像的零拷贝句柄
 *
 * 只持有一块自己 AX_SYS_MemAllocCached 出来的物理连续内存，数据由抓图线程通过硬件
 * VGP 拷贝写入一次（见 IvpsFrameHandle::CopyFrom），析构时统一 AX_SYS_MemFree。
 *
 * @note 与 IVPS 通道帧本身（AX_VIDEO_FRAME_T）解耦：通道帧在拷贝完成后立刻
 *       AX_IVPS_ReleaseChnFrame 归还给硬件流水线，不会被这个 handle 持有，
 *       所以这里的释放语义是 AX_SYS_MemFree，而不是 AX_IVPS_ReleaseChnFrame。
 */
class IvpsFrameHandle {
public:
    IvpsFrameHandle() = default;
    ~IvpsFrameHandle() { release(); }

    IvpsFrameHandle(const IvpsFrameHandle&) = delete;
    IvpsFrameHandle& operator=(const IvpsFrameHandle&) = delete;

    // ## 按给定宽/高/stride/格式申请一块可缓存物理内存；stride<=0 时退化为 width
    bool alloc(AX_U32 width, AX_U32 height, AX_U32 stride, AX_IMG_FORMAT_E format);

    // ## 用硬件 VGP 把一帧仍在 IVPS out-FIFO 里的通道帧线性拷贝到新申请的独立
    // ## 物理内存；源帧不在这里释放，调用方拷贝完立刻归还源通道帧。
    // ## 失败（分配/拷贝失败）返回 nullptr
    static IvpsFramePtr CopyFrom(const AX_VIDEO_FRAME_T& frame);

    // ## 内存是否已经就绪
    bool valid() const { return phy_addr_ != 0 && vir_addr_ != nullptr; }

    AX_U64 phy() const { return phy_addr_; }
    void* vir() const { return vir_addr_; }
    AX_U32 width() const { return width_; }
    AX_U32 height() const { return height_; }
    AX_U32 stride() const { return stride_; }
    AX_U32 size() const { return size_; }
    AX_IMG_FORMAT_E format() const { return format_; }

    AX_U64 ptsUs() const { return pts_us_; }
    void setPtsUs(AX_U64 pts_us) { pts_us_ = pts_us; }

    // ## 生成一份拥有独立内存的 cv::Mat 深拷贝：供画图/去畸变等需要真正
    // ## cv::Mat 像素访问的消费者按需调用；推理路径（XFeat/SuperPoint 的零拷贝
    // ## detect(IvpsFramePtr) 重载）应直接用 phy()/vir()，不要走这里
    cv::Mat toCvMatCopy() const;

private:
    void release();

    AX_U64 phy_addr_ = 0;
    void* vir_addr_ = nullptr;
    AX_U32 width_ = 0;
    AX_U32 height_ = 0;
    AX_U32 stride_ = 0;
    AX_U32 size_ = 0;
    AX_IMG_FORMAT_E format_ = AX_FORMAT_YUV400;
    AX_U64 pts_us_ = 0;
};

} // namespace ms_fusion

#endif // MSF_USE_AXERA_IVPS

#endif // MS_FUSION_COMMON_IVPS_FRAME_HANDLE_H
