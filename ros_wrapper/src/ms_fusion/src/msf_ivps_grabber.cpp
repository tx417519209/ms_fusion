#include "ms_fusion/msf_ivps_grabber.h"

#ifdef MSF_USE_AXERA_IVPS

#include "ms_fusion/ms_fusion_ivps_grab.h"
#include "ms_fusion/manager/ms_fusion_manager.h"

#include <sys/epoll.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <iostream>
#include "ms_fusion/common/runtime_log.h"

namespace ms_fusion {

MsfIvpsGrabber::~MsfIvpsGrabber()
{
    // ## 防御性停止：即使调用方忘了先 stop()，也保证线程在析构前 join 掉
    stop();
}

void MsfIvpsGrabber::setImageCallback(ImageCallback cb)
{
    std::lock_guard<std::mutex> lock(callback_mutex_);
    image_callback_ = std::move(cb);
}

void MsfIvpsGrabber::start()
{
    if (is_grabbing_) {
        return;
    }

    msf_ivps_ensure_sys_init();

    stop_requested_.store(false, std::memory_order_release);

    // ## 队列容量限到 2：下游处理跟不上出帧速度时，堆积的旧帧本来就会被丢弃，
    // ## 缩小队列既降低配对延迟，也少占用硬件 buffer slot
    sync_ = std::make_unique<RawSynchronizer>(
        [](const RawMsfFramePtr& f) { return f->ptsUs(); },
        [](const RawMsfFramePtr& f) { return f->ptsUs(); }, MAX_SYNC_PTS_DIFF_US, 2);
    sync_->registerCallback(
        [this](const RawMsfFramePtr& l, const RawMsfFramePtr& r) { onPairMatched(l, r); });

    const auto& cfg = msf_ivps_grab_config();

    // ## 一个线程只负责一路 GetChnFrame（NV12），不再一个线程/定时器抓多路
    capture_left_thread_ = std::thread(&MsfIvpsGrabber::channelGrabLoop, this, cfg.left_grp,
                                       cfg.left_chn, sync_.get(), true, "capture_left");
    capture_right_thread_ = std::thread(&MsfIvpsGrabber::channelGrabLoop, this, cfg.right_grp,
                                        cfg.right_chn, sync_.get(), false, "capture_right");

    is_grabbing_ = true;
}

void MsfIvpsGrabber::stop()
{
    stop_requested_.store(true, std::memory_order_release);

    if (capture_left_thread_.joinable()) {
        capture_left_thread_.join();
    }
    if (capture_right_thread_.joinable()) {
        capture_right_thread_.join();
    }

    // ## 线程都 join 完了，同步器里残留的未匹配帧随对象销毁自动 ReleaseChnFrame
    sync_.reset();

    is_grabbing_ = false;
}

/**
 * @brief 单路抓图线程主体：只对着自己这一路 grp/chn 抓 NV12 帧
 *
 * 用 epoll 驱动而不是 AX_IVPS_GetChnFrame(..., timeout) 忙等式轮询：拿到该通道的
 * fd 后注册进 epoll，线程阻塞在 epoll_wait 上，通道真正有新帧时内核才唤醒线程，
 * 唤醒后 GetChnFrame 用 0 超时（非阻塞）直接取。epoll_wait 的超时值仅用于线程能
 * 定期醒来检查 stop_requested_，不是取帧节拍。
 */
void MsfIvpsGrabber::channelGrabLoop(IVPS_GRP grp, IVPS_CHN chn, RawSynchronizer* sync, bool is_left,
                                     const char* tag)
{
    constexpr int kFdRetryIntervalMs = 200;
    constexpr int kFdRetryLogEvery = 25;  // 每 ~5s 打一次日志，避免刷屏
    constexpr int kEpollTimeoutMs = 50;

    // ## 通道可能在节点启动时还没建好（相机管线晚起等），
    // ## 与旧的 wall-timer 抓流路径行为一致：无限期重试，不因为一时拿不到 fd 就
    // ## 让这一路永久停摆，只有收到 stop_requested_ 才退出
    int fd = -1;
    for (int attempt = 0; !stop_requested_.load(std::memory_order_acquire); ++attempt) {
        fd = AX_IVPS_GetChnFd(grp, chn);
        if (fd > 0) {
            break;
        }
        if (attempt == 0 || attempt % kFdRetryLogEvery == 0) {
            MSF_WLOG << "[IVPS] AX_IVPS_GetChnFd grp=" << grp << " chn=" << chn << " (" << tag
                     << ") not ready, retrying\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kFdRetryIntervalMs));
    }
    if (fd <= 0) {
        // ## 只有 stop_requested_ 触发才会走到这里
        return;
    }

    const int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd < 0) {
        MSF_ELOG << "[IVPS] epoll_create1 failed (" << tag << "), errno=" << errno << "\n";
        return;
    }

    epoll_event event{};
    event.events = EPOLLIN;
    event.data.fd = fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &event) != 0) {
        MSF_ELOG << "[IVPS] epoll_ctl ADD failed (" << tag << "), errno=" << errno << "\n";
        close(epoll_fd);
        return;
    }

    // ## fd 就绪不代表有帧在流：通道建好后如果相机管线一直不推流，epoll_wait 会
    // ## 一直超时、什么都不打印，现场看起来跟死了一样。这里用一个独立于抓帧节拍
    // ## 的墙钟计时器，每 ~5s 报一次"多久没收到帧了"，直到真正收到第一帧为止。
    constexpr auto kNoFrameLogInterval = std::chrono::seconds(5);
    const auto loop_start = std::chrono::steady_clock::now();
    auto last_frame_time = loop_start;
    auto last_no_frame_log = loop_start;

    while (!stop_requested_.load(std::memory_order_acquire)) {
        epoll_event ready_event{};
        const int nfds = epoll_wait(epoll_fd, &ready_event, 1, kEpollTimeoutMs);
        if (nfds < 0) {
            if (errno == EINTR) {
                continue;
            }
            MSF_ELOG << "[IVPS] epoll_wait failed (" << tag << "), errno=" << errno << "\n";
            break;
        }
        if (nfds == 0) {
            // 超时：没有新帧，回去重新检查 stop_requested_；顺带看看是不是该报一下
            // "已经多久没收到帧了"
            const auto now = std::chrono::steady_clock::now();
            if (now - last_no_frame_log >= kNoFrameLogInterval) {
                const auto silent_s =
                    std::chrono::duration_cast<std::chrono::seconds>(now - last_frame_time).count();
                MSF_WLOG << "[IVPS] grp=" << grp << " chn=" << chn << " (" << tag << ") no frame for "
                         << silent_s << "s; ensure MIPI/IVPS pyramid is streaming\n";
                last_no_frame_log = now;
            }
            continue;
        }

        // ## epoll 已确认这一路有帧可读，GetChnFrame 用 0 超时（非阻塞）直接取
        AX_VIDEO_FRAME_T frame{};
        const int ret = AX_IVPS_GetChnFrame(grp, chn, &frame, 0);
        if (ret != 0) {
            continue;
        }

        last_frame_time = std::chrono::steady_clock::now();

        auto raw = std::make_shared<RawMsfFrame>(grp, chn, frame);
        if (is_left) {
            sync->addLeft(raw);
        } else {
            sync->addRight(raw);
        }
    }

    close(epoll_fd);
}

/**
 * @brief NV12 -> YUV400：NV12 的 Y 平面（起始 stride*height 字节）与 YUV400 内存
 *        布局完全一致，所以只需要用硬件 VGP 把这一段线性拷贝到新申请的独立物理
 *        内存、把目标格式标成 YUV400，不需要真正的色彩空间转换。
 */
IvpsFramePtr MsfIvpsGrabber::convertToGray(const RawMsfFramePtr& raw)
{
    const AX_VIDEO_FRAME_T& frame = raw->frame();
    if (frame.u64PhyAddr[0] == 0 || frame.u32Width == 0 || frame.u32Height == 0) {
        return nullptr;
    }

    auto dst = std::make_shared<IvpsFrameHandle>();
    if (!dst->alloc(frame.u32Width, frame.u32Height, frame.u32PicStride[0], AX_FORMAT_YUV400)) {
        return nullptr;
    }

    const AX_U64 bytes = static_cast<AX_U64>(dst->stride()) * static_cast<AX_U64>(dst->height());
    const int ret = AX_IVPS_CmmCopyVgp(frame.u64PhyAddr[0], dst->phy(), bytes);
    if (ret != 0) {
        MSF_ELOG << "[IVPS] AX_IVPS_CmmCopyVgp (gray) failed, ret=0x" << std::hex << ret
                 << std::dec << "\n";
        return nullptr;
    }
    AX_SYS_MinvalidateCache(dst->phy(), dst->vir(), dst->size());
    dst->setPtsUs(frame.u64PTS);

    return dst;
}

/**
 * @brief NV12 -> RGB888：走 IVPS 硬件 CSC（AX_IVPS_CscVgp）做色彩空间转换，不缩放，
 *        写入新申请的独立物理内存。
 */
/** @brief 左右图配对成功：NV12 -> YUV400 组成 tracking 图像。 */
void MsfIvpsGrabber::onPairMatched(const RawMsfFramePtr& left, const RawMsfFramePtr& right)
{
    const auto acquire_begin = MonotonicRawClock::now();

    // ## left/right（源通道帧）出这个函数作用域时引用计数归零，RawMsfFrame 析构
    // ## 自动 AX_IVPS_ReleaseChnFrame，不需要手动 mmap/释放
    IvpsFramePtr handle_left = convertToGray(left);
    IvpsFramePtr handle_right = convertToGray(right);
    if (!handle_left || !handle_right) {
        return;
    }

    ImageMessage image_message;
    image_message.timestamp = static_cast<double>(left->ptsUs()) / 1000000.0;
    image_message.image_acquire_begin = acquire_begin;
    image_message.image_acquire_begin_valid = true;
    image_message.image0 = std::move(handle_left);
    image_message.image1 = std::move(handle_right);

    // ## 首帧落盘诊断：仅在配置了 first_frame_save_dir 时执行一次，用来确认板端
    // ## 摄像头管线出图正常，不依赖显示器。toCvMatCopy() 的深拷贝被推迟到函数内部
    // ## 的"仅一次"CAS 抢占成功之后才做，调用点每帧只传句柄，不产生额外拷贝。
    msf_ivps_save_first_frame_if_needed(image_message.image0, image_message.image1);

    ImageCallback cb;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        cb = image_callback_;
    }
    if (cb) {
        cb(std::move(image_message));
    }
}

} // namespace ms_fusion

#endif // MSF_USE_AXERA_IVPS
