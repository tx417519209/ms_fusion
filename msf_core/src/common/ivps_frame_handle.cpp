#include "ms_fusion/common/ivps_frame_handle.h"

#ifdef MSF_USE_AXERA_IVPS

#include "ax_ivps_api.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include "ms_fusion/common/runtime_log.h"

namespace ms_fusion {

namespace {
constexpr AX_U32 kIvpsFrameAlign = 128;
constexpr const char* kIvpsFrameSessionName = "ms_fusion_frame";
} // namespace

bool IvpsFrameHandle::alloc(AX_U32 width, AX_U32 height, AX_U32 stride, AX_IMG_FORMAT_E format)
{
    // ## 重新申请前先释放已持有的内存，避免泄漏
    release();

    if (width == 0 || height == 0) {
        return false;
    }

    const AX_U32 use_stride = (stride > 0) ? stride : width;
    const AX_U32 bytes = use_stride * height;
    if (bytes == 0) {
        return false;
    }

    AX_U64 phy = 0;
    void* vir = nullptr;
    const int ret = AX_SYS_MemAllocCached(&phy, &vir, bytes, kIvpsFrameAlign,
                                          reinterpret_cast<const AX_S8*>(kIvpsFrameSessionName));
    if (ret != 0 || vir == nullptr) {
        MSF_ELOG << "[IvpsFrameHandle] AX_SYS_MemAllocCached failed, ret=0x" << std::hex << ret
                 << std::dec << "\n";
        return false;
    }

    phy_addr_ = phy;
    vir_addr_ = vir;
    width_ = width;
    height_ = height;
    stride_ = use_stride;
    size_ = bytes;
    format_ = format;
    return true;
}

/**
 * @brief 用硬件 VGP 把通道帧数据线性拷贝到一块新申请的物理内存并返回句柄
 *
 * 源帧仍然在 IVPS out-FIFO 里（本函数不释放它，调用方在拷贝完成后立刻
 * AX_IVPS_ReleaseChnFrame 归还），目的内存按源帧同样的宽/高/stride/格式申请，
 * 因此总是走 AX_IVPS_CmmCopyVgp 纯线性 DMA，不需要 CropResizeVgp 缩放流水线。
 */
IvpsFramePtr IvpsFrameHandle::CopyFrom(const AX_VIDEO_FRAME_T& frame)
{
    if (frame.u64PhyAddr[0] == 0 || frame.u32Width == 0 || frame.u32Height == 0) {
        return nullptr;
    }

    auto handle = std::make_shared<IvpsFrameHandle>();
    if (!handle->alloc(frame.u32Width, frame.u32Height, frame.u32PicStride[0], frame.enImgFormat)) {
        return nullptr;
    }

    const AX_U64 bytes =
        static_cast<AX_U64>(handle->stride()) * static_cast<AX_U64>(handle->height());
    const int ret = AX_IVPS_CmmCopyVgp(frame.u64PhyAddr[0], handle->phy(), bytes);
    if (ret != 0) {
        MSF_ELOG << "[IvpsFrameHandle] AX_IVPS_CmmCopyVgp failed, ret=0x" << std::hex << ret
                 << std::dec << "\n";
        return nullptr;
    }

    // ## 硬件 DMA 写入后，CPU 端读取前需要失效缓存，避免读到旧数据
    AX_SYS_MinvalidateCache(handle->phy(), handle->vir(), handle->size());
    handle->setPtsUs(frame.u64PTS);
    return handle;
}

void IvpsFrameHandle::release()
{
    if (phy_addr_ != 0 && vir_addr_ != nullptr) {
        AX_SYS_MemFree(phy_addr_, vir_addr_);
    }
    phy_addr_ = 0;
    vir_addr_ = nullptr;
    size_ = 0;
}

cv::Mat IvpsFrameHandle::toCvMatCopy() const
{
    if (!valid() || width_ == 0 || height_ == 0) {
        return cv::Mat();
    }

    const int cv_type = (format_ == AX_FORMAT_RGB888) ? CV_8UC3 : CV_8UC1;

    // ## 先构造一个不持有内存的视图，再 clone 出一份独立拥有内存的深拷贝
    const cv::Mat view(static_cast<int>(height_), static_cast<int>(width_), cv_type, vir_addr_,
                       static_cast<size_t>(stride_));
    return view.clone();
}

} // namespace ms_fusion

#endif // MSF_USE_AXERA_IVPS
