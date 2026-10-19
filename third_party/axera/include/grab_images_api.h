#pragma once
// sys dependency
#include <stdio.h>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <limits>
#include <algorithm>

// axera dependency
#include <ax_sys_api.h>
#include <ax_ivps_api.h>
#include <ax_global_type.h>
#include "infer_api.h"
#define POSE_CHN_GRP 2
#define POSE_CHN_ID 0
#define LEFT_CHN_GRP 1
#define LEFT_CHN_ID 0
#define RIGHT_CHN_GRP 0
#define RIGHT_CHN_ID 0

#define LEFT_PRY_GRP 8
#define LEFT_PRY_640 0
#define LEFT_PRY_320 1
#define LEFT_PRY_160 2
#define LEFT_PRY_80 3
#define RIGHT_PRY_GRP 7
#define RIGHT_PRY_640 0
#define RIGHT_PRY_320 1
#define RIGHT_PRY_160 2
#define RIGHT_PRY_80 3

#define ENABLE_DEBUG_OUTPUT true
#define MAX_SYNC_RETRIES  10 // 最大重试次数，防止无限循环
#define MAX_SYNC_PTS_DIFF_US 2000 // 允许的最大时间戳差值（微秒，2毫秒）
#define PTS_DIFF_US(a, b) ((a) > (b) ? ((a) - (b)) : ((b) - (a)))

inline bool grab_sync_images(AX_VIDEO_FRAME_T& frame_left, AX_VIDEO_FRAME_T& frame_right){
    int ret_left = AX_IVPS_GetChnFrame(LEFT_CHN_GRP, LEFT_CHN_ID, &frame_left, 5);
    if (ret_left != 0)
    {
        if (ENABLE_DEBUG_OUTPUT)
        {
            printf("AX_IVPS_GetChnFrame (left) failed, ret=0x%x\n", ret_left);
        }
        return false;
    }

    int ret_right = AX_IVPS_GetChnFrame(RIGHT_CHN_GRP, RIGHT_CHN_ID, &frame_right, 5);
    if (ret_right != 0)
    {
        if (ENABLE_DEBUG_OUTPUT)
        {
            printf("AX_IVPS_GetChnFrame (right) failed, ret=0x%x\n", ret_right);
        }
        AX_IVPS_ReleaseChnFrame(LEFT_CHN_GRP, LEFT_CHN_ID, &frame_left);
        return false;
    }

    // 1.1) 校验左右图像的时间戳，允许 2ms 以内的时间差
    int sync_retries = 0;
    
    while (PTS_DIFF_US(frame_left.u64PTS, frame_right.u64PTS) > MAX_SYNC_PTS_DIFF_US
           && sync_retries < MAX_SYNC_RETRIES)
    {
        sync_retries++;
        
        // 判断哪个时间戳更小（更早的帧）
        if (frame_left.u64PTS < frame_right.u64PTS)
        {
            // 左侧帧更早，丢弃左侧帧并重新获取
            if (ENABLE_DEBUG_OUTPUT)
            {
                printf("时间戳不匹配: Left=%llu, Right=%llu, 丢弃左侧帧并重新获取 (重试 %d/%d)\n", 
                      frame_left.u64PTS, frame_right.u64PTS, sync_retries, MAX_SYNC_RETRIES);
            }
            
            AX_IVPS_ReleaseChnFrame(LEFT_CHN_GRP, LEFT_CHN_ID, &frame_left);
            
            int ret_left_new = AX_IVPS_GetChnFrame(LEFT_CHN_GRP, LEFT_CHN_ID, &frame_left, 5);
            if (ret_left_new != 0)
            {
                if (ENABLE_DEBUG_OUTPUT)
                {
                    fprintf(stderr, "Left camera: 重新获取帧失败 GRP:%d CHN:%d ERROR:0x%x\n", 
                            LEFT_CHN_GRP, LEFT_CHN_ID, ret_left_new);
                }
                // 释放右侧帧
                AX_IVPS_ReleaseChnFrame(RIGHT_CHN_GRP, RIGHT_CHN_ID, &frame_right);
                return false;
            }
        }
        else
        {
            // 右侧帧更早，丢弃右侧帧并重新获取
            if (ENABLE_DEBUG_OUTPUT)
            {
                printf("时间戳不匹配: Left=%llu, Right=%llu, 丢弃右侧帧并重新获取 (重试 %d/%d)\n", 
                      frame_left.u64PTS, frame_right.u64PTS, sync_retries, MAX_SYNC_RETRIES);
            }
            
            AX_IVPS_ReleaseChnFrame(RIGHT_CHN_GRP, RIGHT_CHN_ID, &frame_right);
            
            int ret_right_new = AX_IVPS_GetChnFrame(RIGHT_CHN_GRP, RIGHT_CHN_ID, &frame_right, 5);
            if (ret_right_new != 0)
            {
                if (ENABLE_DEBUG_OUTPUT)
                {
                    fprintf(stderr, "Right camera: 重新获取帧失败 GRP:%d CHN:%d ERROR:0x%x\n", 
                            RIGHT_CHN_GRP, RIGHT_CHN_ID, ret_right_new);
                }
                // 释放左侧帧
                AX_IVPS_ReleaseChnFrame(LEFT_CHN_GRP, LEFT_CHN_ID, &frame_left);
                return false;
            }
        }
    }
    
    // 检查是否达到最大重试次数
    if (PTS_DIFF_US(frame_left.u64PTS, frame_right.u64PTS) > MAX_SYNC_PTS_DIFF_US)
    {
        if (ENABLE_DEBUG_OUTPUT)
        {
            fprintf(stderr, "时间戳同步失败: 达到最大重试次数 %d, Left=%llu, Right=%llu\n", 
                    MAX_SYNC_RETRIES, frame_left.u64PTS, frame_right.u64PTS);
        }
        // 释放两个帧
        AX_IVPS_ReleaseChnFrame(LEFT_CHN_GRP, LEFT_CHN_ID, &frame_left);
        AX_IVPS_ReleaseChnFrame(RIGHT_CHN_GRP, RIGHT_CHN_ID, &frame_right);
        return false;
    }
    
    // 时间戳已同步
    if (sync_retries > 0 && ENABLE_DEBUG_OUTPUT)
    {
        printf("时间戳已同步: Left=%llu, Right=%llu, 重试次数=%d\n", 
              frame_left.u64PTS, frame_right.u64PTS, sync_retries);
    }

    return true;
}


inline bool grab_single_images(AX_VIDEO_FRAME_T& frame_left){
    int ret_left = AX_IVPS_GetChnFrame(POSE_CHN_GRP, POSE_CHN_ID, &frame_left, 5);
    if (ret_left != 0)
    {
        if (ENABLE_DEBUG_OUTPUT)
        {
            printf("AX_IVPS_GetChnFrame (left) failed, ret=0x%x\n", ret_left);
        }
        return false;
    }

    return true;
}


inline void* get_images_addr(AX_VIDEO_FRAME_T& frame_src) {
    if (frame_src.u64PhyAddr[0] == 0) {
        std::cerr << "get_images_addr invalid src addr, phy=" << frame_src.u64PhyAddr[0] << "\n";
        return nullptr;
    }
    if(frame_src.u64VirAddr[0] != 0) {
        return reinterpret_cast<void*>(frame_src.u64VirAddr[0]);
    }
    void* vir_addr = AX_SYS_MmapCache(frame_src.u64PhyAddr[0], frame_src.u32FrameSize);
    if (vir_addr == nullptr) {
        std::cerr << "AX_SYS_MmapCache failed.\n";
        return nullptr;
    }
    frame_src.u64VirAddr[0] = reinterpret_cast<AX_U64>(vir_addr);
    return vir_addr;
}

inline void release_sync_images(AX_VIDEO_FRAME_T& frame_left, AX_VIDEO_FRAME_T& frame_right) {
    if (frame_left.u64PhyAddr[0] != 0) {
        if(frame_left.u64VirAddr[0] != 0) {
            AX_SYS_Munmap(reinterpret_cast<AX_VOID*>(frame_left.u64VirAddr[0]), frame_left.u32FrameSize);
        }
        AX_IVPS_ReleaseChnFrame(LEFT_CHN_GRP, LEFT_CHN_ID, &frame_left);
    }
    if (frame_right.u64PhyAddr[0] != 0) {
        if(frame_right.u64VirAddr[0] != 0) {
            AX_SYS_Munmap(reinterpret_cast<AX_VOID*>(frame_right.u64VirAddr[0]), frame_right.u32FrameSize);
        }        
        AX_IVPS_ReleaseChnFrame(RIGHT_CHN_GRP, RIGHT_CHN_ID, &frame_right);
    }
    memset(&frame_left, 0, sizeof(AX_VIDEO_FRAME_T));
    memset(&frame_right, 0, sizeof(AX_VIDEO_FRAME_T));
}

inline bool resize_image_ivps(const AX_VIDEO_FRAME_T& frame_src, AX_VIDEO_FRAME_T& frame_dst, int input_h, int input_w){
    if (input_h <= 0 || input_w <= 0) {
        std::cerr << "resize_image_ivps invalid target size, h=" << input_h << ", w=" << input_w << "\n";
        return false;
    }
    if (frame_src.u32Width == 0 || frame_src.u32Height == 0) {
        std::cerr << "resize_image_ivps invalid src size, w=" << frame_src.u32Width
                  << ", h=" << frame_src.u32Height << "\n";
        return false;
    }
    if (frame_src.u64PhyAddr[0] == 0) {
        std::cerr << "resize_image_ivps invalid src phys addr.\n";
        return false;
    }
    if (frame_dst.u64PhyAddr[0] == 0 || frame_dst.u64VirAddr[0] == 0) {
        std::cerr << "resize_image_ivps invalid dst addr, phy=" << frame_dst.u64PhyAddr[0]
                  << ", vir=" << frame_dst.u64VirAddr[0] << "\n";
        return false;
    }
    if (frame_dst.u32Width != static_cast<AX_U32>(input_w) || frame_dst.u32Height != static_cast<AX_U32>(input_h)) {
        std::cerr << "resize_image_ivps dst size mismatch, dst(" << frame_dst.u32Width << "x" << frame_dst.u32Height
                  << ") expect(" << input_w << "x" << input_h << ")\n";
        return false;
    }
    if (frame_dst.enImgFormat != AX_FORMAT_RGB888) {
        std::cerr << "resize_image_ivps dst img format mismatch, got=" << static_cast<int>(frame_dst.enImgFormat)
                  << ", expect=" << static_cast<int>(AX_FORMAT_RGB888) << "\n";
        return false;
    }
    const AX_U32 expected_stride = static_cast<AX_U32>(input_w * 3);
    const AX_U32 expected_size = static_cast<AX_U32>(input_h) * expected_stride;
    if (frame_dst.u32PicStride[0] != expected_stride) {
        std::cerr << "resize_image_ivps dst stride mismatch, got=" << frame_dst.u32PicStride[0]
                  << ", expect=" << expected_stride << "\n";
        return false;
    }
    if (frame_dst.u32FrameSize < expected_size) {
        std::cerr << "resize_image_ivps dst frame size too small, got=" << frame_dst.u32FrameSize
                  << ", expect>=" << expected_size << "\n";
        return false;
    }

    AX_IVPS_ASPECT_RATIO_T resize_ctrl;
    memset(&resize_ctrl, 0, sizeof(resize_ctrl));
    resize_ctrl.nBgColor = 0xFFFFFF;
    resize_ctrl.eMode = AX_IVPS_ASPECT_RATIO_STRETCH;
    resize_ctrl.eAligns[0] = AX_IVPS_ASPECT_RATIO_HORIZONTAL_CENTER;
    resize_ctrl.eAligns[1] = AX_IVPS_ASPECT_RATIO_VERTICAL_CENTER;

    int ret = AX_IVPS_CropResizeVgp(&frame_src, &frame_dst, &resize_ctrl);
    if(ret != 0){
        std::cerr << "AX_IVPS_CropResizeVgp failed, ret=" << ret << "\n";
        return false;
    }

    ret = AX_SYS_MinvalidateCache(frame_dst.u64PhyAddr[0],
        (AX_VOID*)frame_dst.u64VirAddr[0],
        static_cast<AX_U32>(frame_dst.u32FrameSize));
    if (ret != 0) {
        std::cerr << "AX_SYS_MinvalidateCache failed for left frame, ret=" << ret << "\n";
    }

    return true;
}

inline bool alloc_resized_rgb_frame(AX_VIDEO_FRAME_T& frame_dst, int input_h, int input_w, int input_c) {
    if (input_h <= 0 || input_w <= 0 || input_c <= 0) {
        std::cerr << "alloc_resized_rgb_frame invalid size: h=" << input_h
                  << ", w=" << input_w << ", c=" << input_c << "\n";
        return false;
    }

    memset(&frame_dst, 0, sizeof(AX_VIDEO_FRAME_T));
    const size_t input_bytes = static_cast<size_t>(input_h) * static_cast<size_t>(input_w) *
                               static_cast<size_t>(input_c);
    int ret = AX_SYS_MemAllocCached(reinterpret_cast<AX_U64*>(&frame_dst.u64PhyAddr[0]),
                                    reinterpret_cast<void**>(&frame_dst.u64VirAddr[0]),
                                    input_bytes,
                                    AX_CMM_ALIGN_SIZE,
                                    reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
    if (ret != 0 || frame_dst.u64PhyAddr[0] == 0 || frame_dst.u64VirAddr[0] == 0) {
        std::cerr << "AX_SYS_MemAllocCached failed, ret=" << ret << "\n";
        frame_dst.u64PhyAddr[0] = 0;
        frame_dst.u64VirAddr[0] = 0;
        return false;
    }

    frame_dst.u32Width = static_cast<AX_U32>(input_w);
    frame_dst.u32Height = static_cast<AX_U32>(input_h);
    frame_dst.enImgFormat = AX_FORMAT_RGB888;
    frame_dst.stDynamicRange = AX_DYNAMIC_RANGE_SDR8;
    frame_dst.u32PicStride[0] = static_cast<AX_U32>(input_w * input_c);
    frame_dst.u32FrameSize = static_cast<AX_U32>(input_h) * frame_dst.u32PicStride[0];
    return true;
}

inline void free_resized_frame(AX_VIDEO_FRAME_T& frame_dst) {
    if (frame_dst.u64PhyAddr[0] != 0 && frame_dst.u64VirAddr[0] != 0) {
        AX_SYS_MemFree(frame_dst.u64PhyAddr[0], reinterpret_cast<AX_VOID*>(frame_dst.u64VirAddr[0]));
    }
    memset(&frame_dst, 0, sizeof(AX_VIDEO_FRAME_T));
}

