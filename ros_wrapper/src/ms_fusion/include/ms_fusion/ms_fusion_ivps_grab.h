#pragma once

#ifdef MSF_USE_AXERA_IVPS

#include <atomic>
#include <cstring>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <ax_sys_api.h>
#include <ax_ivps_api.h>
#include "grab_images_api.h"
#include "ms_fusion/common/ivps_frame_handle.h"
#include "ms_fusion/common/padded_image_geometry.h"
#include "ms_fusion/common/runtime_log.h"

namespace ms_fusion {

// tracking: GRP 7/8 chn=0 灰度金字塔。
constexpr int MSF_IVPS_GRAB_TIMEOUT_MS = 200;

inline void msf_ivps_ensure_sys_init() {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    const AX_S32 ret = AX_SYS_Init();
    if (ret != 0) {
        MSF_ELOG << "[IVPS] AX_SYS_Init failed, ret=0x" << std::hex << ret << std::dec << "\n";
        return;
    }
    initialized = true;
}

inline const char* msf_ivps_ret_to_string(AX_S32 ret) {
    switch (ret) {
    case AX_ERR_IVPS_BUF_EMPTY:
        return "IVPS buffer empty (no frame available)";
    case AX_ERR_IVPS_TIMED_OUT:
        return "IVPS timed out";
    case AX_ERR_IVPS_NOT_INIT:
        return "IVPS not initialized";
    case AX_ERR_IVPS_UNEXIST:
        return "IVPS channel/group does not exist";
    default:
        return "IVPS error";
    }
}

struct MsFusionIvpsGrabConfig {
    // tracking 双目（灰度金字塔）
    IVPS_GRP left_grp = LEFT_PRY_GRP;
    IVPS_CHN left_chn = LEFT_PRY_640;
    IVPS_GRP right_grp = RIGHT_PRY_GRP;
    IVPS_CHN right_chn = RIGHT_PRY_640;
    int tracking_fusion_width = 0;
    int tracking_fusion_height = 0;

    std::string first_frame_save_dir;
};

inline MsFusionIvpsGrabConfig& msf_ivps_grab_config() {
    static MsFusionIvpsGrabConfig cfg;
    return cfg;
}

inline void set_msf_ivps_grab_config(const MsFusionIvpsGrabConfig& cfg) {
    msf_ivps_grab_config() = cfg;
}

inline void set_msf_ivps_first_frame_save_dir(const std::string& dir) {
    msf_ivps_grab_config().first_frame_save_dir = dir;
}

inline AX_U32 msf_ivps_frame_mmap_size(const AX_VIDEO_FRAME_T& frame) {
    if (frame.u32FrameSize > 0) {
        return frame.u32FrameSize;
    }
    if (frame.u32Height > 0 && frame.u32PicStride[0] > 0) {
        return frame.u32PicStride[0] * frame.u32Height;
    }
    return 0;
}

inline void* msf_ivps_mmap_frame(AX_VIDEO_FRAME_T& frame) {
    if (frame.u64PhyAddr[0] == 0) {
        return nullptr;
    }
    if (frame.u64VirAddr[0] != 0) {
        return reinterpret_cast<void*>(frame.u64VirAddr[0]);
    }
    const AX_U32 mmap_size = msf_ivps_frame_mmap_size(frame);
    if (mmap_size == 0) {
        return nullptr;
    }
    void* vir_addr = AX_SYS_Mmap(frame.u64PhyAddr[0], mmap_size);
    if (vir_addr == nullptr) {
        return nullptr;
    }
    frame.u64VirAddr[0] = reinterpret_cast<AX_U64>(vir_addr);
    return vir_addr;
}

inline bool msf_ivps_invalidate_frame_cache(const AX_VIDEO_FRAME_T& frame, void* vir_addr) {
    if (vir_addr == nullptr || frame.u32FrameSize == 0) {
        return false;
    }
    const AX_S32 ret = AX_SYS_MinvalidateCache(frame.u64PhyAddr[0], vir_addr, frame.u32FrameSize);
    return ret == 0;
}

inline bool msf_ivps_frame_to_gray(const AX_VIDEO_FRAME_T& frame, void* vir_addr, cv::Mat& gray_out) {
    if (vir_addr == nullptr || frame.u32Width == 0 || frame.u32Height == 0) {
        return false;
    }
    cv::Mat gray_view(static_cast<int>(frame.u32Height), static_cast<int>(frame.u32Width), CV_8UC1,
                      vir_addr, static_cast<size_t>(frame.u32PicStride[0]));
    gray_out = gray_view.clone();
    return true;
}

inline void msf_ivps_apply_tracking_resize(const cv::Mat& gray, cv::Mat& out) {
    const auto& cfg = msf_ivps_grab_config();
    if (cfg.tracking_fusion_width > 0 && cfg.tracking_fusion_height > 0 &&
        (gray.cols != cfg.tracking_fusion_width || gray.rows != cfg.tracking_fusion_height)) {
        const auto geometry = makeTopLeftPaddedImageGeometry(
            gray.cols, gray.rows, cfg.tracking_fusion_width, cfg.tracking_fusion_height);
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(geometry.content_width, geometry.content_height),
                   0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(resized, out, 0, geometry.pad_bottom, 0, geometry.pad_right,
                           cv::BORDER_CONSTANT, cv::Scalar::all(0));
        return;
    }
    out = gray;
}

// IVPS RGB888（内存布局 BGR）-> OpenCV BGR 三通道
inline bool msf_ivps_frame_to_bgr_rgb(const AX_VIDEO_FRAME_T& frame, void* vir_addr, cv::Mat& bgr_out) {
    if (vir_addr == nullptr || frame.u32Width == 0 || frame.u32Height == 0) {
        return false;
    }
    msf_ivps_invalidate_frame_cache(frame, vir_addr);
    cv::Mat bgr_view(static_cast<int>(frame.u32Height), static_cast<int>(frame.u32Width), CV_8UC3,
                     vir_addr, static_cast<size_t>(frame.u32PicStride[0]));
    bgr_out = bgr_view.clone();
    return !bgr_out.empty();
}

inline bool msf_ivps_frame_to_tracking_gray(const AX_VIDEO_FRAME_T& frame, void* vir_addr,
                                            cv::Mat& gray_out) {
    cv::Mat gray;
    if (!msf_ivps_frame_to_gray(frame, vir_addr, gray)) {
        return false;
    }
    msf_ivps_apply_tracking_resize(gray, gray_out);
    return true;
}

inline void msf_ivps_save_first_frame_if_needed(const IvpsFramePtr& frame_left,
                                                const IvpsFramePtr& frame_right) {
    // ## FrameSynchronizer 的配对回调是在锁外执行的（见 frame_synchronizer.h 的
    // ## 设计说明），左右抓图线程可能并发触发到这里，普通 static bool 在多线程下
    // ## 是数据竞争（可能重复保存，且是未定义行为）。用 atomic CAS 原子地"抢"到
    // ## 第一次保存的名额；保存失败时把名额还回去，允许后续帧重试。
    static std::atomic<bool> saved{false};
    if (!frame_left || !frame_right) {
        return;
    }
    const std::string& save_dir = msf_ivps_grab_config().first_frame_save_dir;
    if (save_dir.empty()) {
        return;
    }
    bool expected = false;
    if (!saved.compare_exchange_strong(expected, true)) {
        return;
    }

    // toCvMatCopy() 是真正的整帧内存拷贝，只在上面的 CAS 抢到"仅此一次"的名额之后
    // 才做——调用点不需要（也不应该）在每一帧都为这次诊断性落盘付出深拷贝的代价。
    const cv::Mat gray_left = frame_left->toCvMatCopy();
    const cv::Mat gray_right = frame_right->toCvMatCopy();
    if (gray_left.empty() || gray_right.empty()) {
        saved.store(false, std::memory_order_relaxed);
        return;
    }

    const std::string left_path = save_dir + "/ivps_first_frame_left.png";
    const std::string right_path = save_dir + "/ivps_first_frame_right.png";
    const bool left_ok = cv::imwrite(left_path, gray_left);
    const bool right_ok = cv::imwrite(right_path, gray_right);
    if (left_ok && right_ok) {
        MSF_CLOG << "[IVPS] saved first tracking frame " << gray_left.cols << "x" << gray_left.rows
                 << ": " << left_path << ", " << right_path << std::endl;
    } else {
        MSF_ELOG << "[IVPS] failed to save first tracking frame under " << save_dir << " (left_ok="
                 << left_ok << ", right_ok=" << right_ok << ")" << std::endl;
        saved.store(false, std::memory_order_relaxed);
    }
}

inline void msf_ivps_log_frame_once(const AX_VIDEO_FRAME_T& frame_left,
                                    const AX_VIDEO_FRAME_T& frame_right,
                                    const char* tag) {
    static AX_U32 last_w = 0;
    static AX_U32 last_h = 0;
    if (frame_left.u32Width == last_w && frame_left.u32Height == last_h) {
        return;
    }
    last_w = frame_left.u32Width;
    last_h = frame_left.u32Height;
    MSF_VLOG << "[IVPS] " << tag << " " << frame_left.u32Width << "x" << frame_left.u32Height
             << " stride=" << frame_left.u32PicStride[0] << " PTS L/R="
             << static_cast<unsigned long long>(frame_left.u64PTS) << "/"
             << static_cast<unsigned long long>(frame_right.u64PTS) << std::endl;
}

inline bool msf_ivps_grab_sync_stereo_pair(IVPS_GRP left_grp,
                                           IVPS_CHN left_chn,
                                           IVPS_GRP right_grp,
                                           IVPS_CHN right_chn,
                                           AX_VIDEO_FRAME_T& frame_left,
                                           AX_VIDEO_FRAME_T& frame_right,
                                           AX_S32* last_error = nullptr) {
    if (last_error != nullptr) {
        *last_error = 0;
    }
    msf_ivps_ensure_sys_init();

    int ret_left = AX_IVPS_GetChnFrame(left_grp, left_chn, &frame_left, MSF_IVPS_GRAB_TIMEOUT_MS);
    if (ret_left != 0) {
        if (last_error != nullptr) {
            *last_error = ret_left;
        }
        return false;
    }

    int ret_right =
        AX_IVPS_GetChnFrame(right_grp, right_chn, &frame_right, MSF_IVPS_GRAB_TIMEOUT_MS);
    if (ret_right != 0) {
        if (last_error != nullptr) {
            *last_error = ret_right;
        }
        AX_IVPS_ReleaseChnFrame(left_grp, left_chn, &frame_left);
        return false;
    }

    int sync_retries = 0;
    while (PTS_DIFF_US(frame_left.u64PTS, frame_right.u64PTS) > MAX_SYNC_PTS_DIFF_US &&
           sync_retries < MAX_SYNC_RETRIES) {
        sync_retries++;
        if (frame_left.u64PTS < frame_right.u64PTS) {
            AX_IVPS_ReleaseChnFrame(left_grp, left_chn, &frame_left);
            ret_left = AX_IVPS_GetChnFrame(left_grp, left_chn, &frame_left, MSF_IVPS_GRAB_TIMEOUT_MS);
            if (ret_left != 0) {
                if (last_error != nullptr) {
                    *last_error = ret_left;
                }
                AX_IVPS_ReleaseChnFrame(right_grp, right_chn, &frame_right);
                return false;
            }
        } else {
            AX_IVPS_ReleaseChnFrame(right_grp, right_chn, &frame_right);
            ret_right =
                AX_IVPS_GetChnFrame(right_grp, right_chn, &frame_right, MSF_IVPS_GRAB_TIMEOUT_MS);
            if (ret_right != 0) {
                if (last_error != nullptr) {
                    *last_error = ret_right;
                }
                AX_IVPS_ReleaseChnFrame(left_grp, left_chn, &frame_left);
                return false;
            }
        }
    }

    if (PTS_DIFF_US(frame_left.u64PTS, frame_right.u64PTS) > MAX_SYNC_PTS_DIFF_US) {
        AX_IVPS_ReleaseChnFrame(left_grp, left_chn, &frame_left);
        AX_IVPS_ReleaseChnFrame(right_grp, right_chn, &frame_right);
        return false;
    }

    return true;
}

inline void msf_ivps_release_stereo_pair(IVPS_GRP left_grp,
                                         IVPS_CHN left_chn,
                                         IVPS_GRP right_grp,
                                         IVPS_CHN right_chn,
                                         AX_VIDEO_FRAME_T& frame_left,
                                         AX_VIDEO_FRAME_T& frame_right) {
    if (frame_left.u64PhyAddr[0] != 0) {
        if (frame_left.u64VirAddr[0] != 0) {
            AX_SYS_Munmap(reinterpret_cast<AX_VOID*>(frame_left.u64VirAddr[0]),
                          msf_ivps_frame_mmap_size(frame_left));
        }
        AX_IVPS_ReleaseChnFrame(left_grp, left_chn, &frame_left);
    }
    if (frame_right.u64PhyAddr[0] != 0) {
        if (frame_right.u64VirAddr[0] != 0) {
            AX_SYS_Munmap(reinterpret_cast<AX_VOID*>(frame_right.u64VirAddr[0]),
                          msf_ivps_frame_mmap_size(frame_right));
        }
        AX_IVPS_ReleaseChnFrame(right_grp, right_chn, &frame_right);
    }
    memset(&frame_left, 0, sizeof(AX_VIDEO_FRAME_T));
    memset(&frame_right, 0, sizeof(AX_VIDEO_FRAME_T));
}

inline bool msf_ivps_grab_sync_stereo(AX_VIDEO_FRAME_T& frame_left, AX_VIDEO_FRAME_T& frame_right,
                                      AX_S32* last_error = nullptr) {
    const auto& cfg = msf_ivps_grab_config();
    const bool ok = msf_ivps_grab_sync_stereo_pair(cfg.left_grp, cfg.left_chn, cfg.right_grp,
                                                   cfg.right_chn, frame_left, frame_right,
                                                   last_error);
    if (ok) {
        msf_ivps_log_frame_once(frame_left, frame_right, "tracking mono");
    }
    return ok;
}

inline void msf_ivps_release_stereo(AX_VIDEO_FRAME_T& frame_left, AX_VIDEO_FRAME_T& frame_right) {
    const auto& cfg = msf_ivps_grab_config();
    msf_ivps_release_stereo_pair(cfg.left_grp, cfg.left_chn, cfg.right_grp, cfg.right_chn,
                                 frame_left, frame_right);
}

} // namespace ms_fusion

#endif // MSF_USE_AXERA_IVPS
