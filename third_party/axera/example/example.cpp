/**
 * IVPS 金字塔单路取帧示例（原生 API）
 *
 * 1. AX_IVPS_GetChnFrame / AX_IVPS_ReleaseChnFrame
 * 2. AX_SYS_MmapCache / AX_SYS_MinvalidateCache / clone -> cv::Mat / AX_SYS_Munmap
 * 3. 打印 AX_VIDEO_FRAME_T 全部字段
 *
 * 右目或其它层级：将 LEFT_PRY_GRP / LEFT_PRY_640 换为 RIGHT_PRY_GRP / RIGHT_PRY_320 等即可。
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "grab_images_api.h"

static void print_video_frame(const AX_VIDEO_FRAME_T& f) {
    printf("u32Width              = %u\n", f.u32Width);
    printf("u32Height             = %u\n", f.u32Height);
    printf("enImgFormat           = %d\n", static_cast<int>(f.enImgFormat));
    printf("enVscanFormat         = %d\n", static_cast<int>(f.enVscanFormat));
    printf("stCompressInfo.enCompressMode = %d\n", static_cast<int>(f.stCompressInfo.enCompressMode));
    printf("stCompressInfo.u32CompressLevel = %u\n", f.stCompressInfo.u32CompressLevel);
    printf("stDynamicRange        = %d\n", static_cast<int>(f.stDynamicRange));
    printf("stColorGamut          = %d\n", static_cast<int>(f.stColorGamut));
    for (int i = 0; i < AX_MAX_COLOR_COMPONENT; ++i) {
        printf("u32PicStride[%d]        = %u\n", i, f.u32PicStride[i]);
        printf("u32ExtStride[%d]        = %u\n", i, f.u32ExtStride[i]);
        printf("u64PhyAddr[%d]          = 0x%llx\n", i, (unsigned long long)f.u64PhyAddr[i]);
        printf("u64VirAddr[%d]          = 0x%llx\n", i, (unsigned long long)f.u64VirAddr[i]);
        printf("u64ExtPhyAddr[%d]       = 0x%llx\n", i, (unsigned long long)f.u64ExtPhyAddr[i]);
        printf("u64ExtVirAddr[%d]       = 0x%llx\n", i, (unsigned long long)f.u64ExtVirAddr[i]);
        printf("u32HeaderSize[%d]      = %u\n", i, f.u32HeaderSize[i]);
        printf("u32BlkId[%d]           = %u\n", i, f.u32BlkId[i]);
    }
    printf("s16CropX              = %d\n", f.s16CropX);
    printf("s16CropY              = %d\n", f.s16CropY);
    printf("s16CropWidth          = %d\n", f.s16CropWidth);
    printf("s16CropHeight         = %d\n", f.s16CropHeight);
    printf("u32TimeRef            = %u\n", f.u32TimeRef);
    printf("u64PTS                = %llu\n", (unsigned long long)f.u64PTS);//时间戳单位微秒
    printf("u64SeqNum             = %llu\n", (unsigned long long)f.u64SeqNum);
    printf("u64UserData           = 0x%llx\n", (unsigned long long)f.u64UserData);
    printf("u64PrivateData        = 0x%llx\n", (unsigned long long)f.u64PrivateData);
    printf("u32FrameFlag          = 0x%x\n", f.u32FrameFlag);
    printf("u32FrameSize          = %u\n", f.u32FrameSize);
}

int main() {
    AX_SYS_Init();
    AX_VIDEO_FRAME_T frame {};
    memset(&frame, 0, sizeof(frame));

    // 1) 取帧：左 640 层-阻塞5ms获取
    int ret = AX_IVPS_GetChnFrame(LEFT_PRY_GRP, LEFT_PRY_640, &frame, 5);
    if (ret != 0) {
        fprintf(stderr, "AX_IVPS_GetChnFrame failed, ret=0x%x\n", ret);
        return -1;
    }

    // 2) 打印 AX_VIDEO_FRAME_T的全部字段
    print_video_frame(frame);

    // 3) MmapCache -> MinvalidateCache -> 拷贝到 cv::Mat
    cv::Mat mat;
    void* vir_addr = AX_SYS_MmapCache(frame.u64PhyAddr[0], frame.u32FrameSize);
    if (vir_addr == nullptr) {
        fprintf(stderr, "AX_SYS_MmapCache failed\n");
    } else {
        ret = AX_SYS_MinvalidateCache(frame.u64PhyAddr[0], vir_addr, frame.u32FrameSize);
        if (ret != 0) {
            fprintf(stderr, "AX_SYS_MinvalidateCache failed, ret=0x%x\n", ret);
        } else {
            const int cv_type = (frame.enImgFormat == AX_FORMAT_RGB888) ? CV_8UC3 : CV_8UC1;
            cv::Mat wrapped(static_cast<int>(frame.u32Height),
                            static_cast<int>(frame.u32Width),
                            cv_type,
                            vir_addr,
                            frame.u32PicStride[0]);
            const auto t0 = std::chrono::steady_clock::now();
            mat = wrapped.clone();
            const auto clone_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
            printf("wrapped.clone() %dx%d type=%d elapsed=%ld us\n",
                   wrapped.cols, wrapped.rows, wrapped.type(), static_cast<long>(clone_us));
            cv::imwrite("left_640.png", mat);
            printf("cv::Mat %dx%d, elemSize=%zu\n", mat.cols, mat.rows, mat.elemSize());
        }
        AX_SYS_Munmap(vir_addr, frame.u32FrameSize);
    }

    //  4) 释放帧
    AX_IVPS_ReleaseChnFrame(LEFT_PRY_GRP, LEFT_PRY_640, &frame);
    AX_SYS_Deinit();
    return 0;
}
