#ifndef MS_FUSION_FRAME_SYNCHRONIZER_H
#define MS_FUSION_FRAME_SYNCHRONIZER_H

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace ms_fusion {

/**
 * @brief 双向近似时间同步器（微秒 PTS 版）
 *
 * 迁移自 front_end/include/front_end/interface/frame_synchronizer.h（双向最优匹配 +
 * 丢弃过期帧），用板端原生的整数微秒时间戳代替 rclcpp::Time，不依赖 rclcpp，抓图线程
 * 可以脱离 ROS 独立跑。
 *
 * MsgL / MsgR 通过 TimestampExtractor 仿函数注入时间戳（微秒），与消息类型解耦，
 * 线程安全：addLeft()/addRight() 可以分别从不同线程调用。
 */
template <typename MsgL, typename MsgR>
class FrameSynchronizer {
public:
    using MsgLPtr = std::shared_ptr<MsgL>;
    using MsgRPtr = std::shared_ptr<MsgR>;
    using Callback = std::function<void(const MsgLPtr&, const MsgRPtr&)>;
    using TsExtractL = std::function<int64_t(const MsgLPtr&)>;
    using TsExtractR = std::function<int64_t(const MsgRPtr&)>;

    FrameSynchronizer(TsExtractL extract_l, TsExtractR extract_r, int64_t max_diff_us,
                      size_t queue_size = 4)
        : extract_l_(std::move(extract_l)), extract_r_(std::move(extract_r)),
          max_diff_us_(max_diff_us), queue_size_(queue_size)
    {
    }

    void registerCallback(Callback cb)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callback_ = std::move(cb);
    }

    // ## 左右各自的抓图线程调用，互不阻塞对方
    void addLeft(const MsgLPtr& msg) { addImpl(left_q_, msg, true); }
    void addRight(const MsgRPtr& msg) { addImpl(right_q_, msg, false); }

private:
    template <typename Q, typename Msg>
    void addImpl(Q& own_q, const std::shared_ptr<Msg>& msg, bool is_left)
    {
        // ## 匹配结果和回调函数在锁内拷出，回调本身放到锁外调用：
        // ## 回调里可能有较慢的操作（如硬件拷贝、下游再加锁），锁内调用会通过这把
        // ## mutex_ 把本该互不等待的左右两路抓图线程串行化，回调若再次触达同一个
        // ## 同步器（递归 addLeft/addRight）还会直接死锁。
        MsgLPtr matched_l;
        MsgRPtr matched_r;
        Callback cb;
        bool matched = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);

            own_q.push_back(msg);
            // ## 队列超限时丢最老的一帧；shared_ptr 引用计数归零时该帧的 RAII
            // ## 析构会自动触发释放，不需要在这里手动释放
            if (own_q.size() > queue_size_) {
                own_q.pop_front();
            }

            matched = trySync(is_left, matched_l, matched_r);
            if (matched) {
                cb = callback_;
            }
        }

        if (matched && cb) {
            cb(matched_l, matched_r);
        }
    }

    // ## 仅在持有 mutex_ 时调用：找配对、丢弃过期帧；匹配到的一对通过 out 参数
    // ## 带出去，调用方负责在锁外执行回调
    bool trySync(bool new_msg_is_left, MsgLPtr& out_l, MsgRPtr& out_r)
    {
        if (left_q_.empty() || right_q_.empty()) {
            return false;
        }

        auto [la, ra, da] = new_msg_is_left ? bestMatchL2R(left_q_.back())
                                            : bestMatchR2L(right_q_.back());
        auto [lb, rb, db] = new_msg_is_left ? bestMatchR2L(right_q_.back())
                                            : bestMatchL2R(left_q_.back());

        const int64_t best_dt = std::min(da, db);
        if (best_dt > max_diff_us_) {
            return false;
        }

        out_l = (da <= db) ? la : lb;
        out_r = (da <= db) ? ra : rb;

        const int64_t tl = extract_l_(out_l);
        const int64_t tr = extract_r_(out_r);

        // ## 丢弃匹配帧及更早的帧：未被匹配上的旧帧引用计数归零，自动释放回硬件
        dropUpTo(left_q_, tl, extract_l_);
        dropUpTo(right_q_, tr, extract_r_);

        return true;
    }

    std::tuple<MsgLPtr, MsgRPtr, int64_t> bestMatchL2R(const MsgLPtr& anchor)
    {
        const int64_t t = extract_l_(anchor);
        const auto match = findBest(right_q_, t, extract_r_);
        return {anchor, right_q_[match.first], match.second};
    }

    std::tuple<MsgLPtr, MsgRPtr, int64_t> bestMatchR2L(const MsgRPtr& anchor)
    {
        const int64_t t = extract_r_(anchor);
        const auto match = findBest(left_q_, t, extract_l_);
        return {left_q_[match.first], anchor, match.second};
    }

    template <typename Q, typename Extract>
    std::pair<size_t, int64_t> findBest(const Q& q, int64_t t, const Extract& extract)
    {
        size_t best_idx = 0;
        int64_t best_diff = std::numeric_limits<int64_t>::max();
        for (size_t i = 0; i < q.size(); ++i) {
            const int64_t diff = std::abs(t - extract(q[i]));
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = i;
            }
        }
        return {best_idx, best_diff};
    }

    template <typename Q, typename Extract>
    void dropUpTo(Q& q, int64_t t, const Extract& extract)
    {
        while (!q.empty() && extract(q.front()) <= t) {
            q.pop_front();
        }
    }

    TsExtractL extract_l_;
    TsExtractR extract_r_;
    int64_t max_diff_us_;
    size_t queue_size_;

    std::deque<MsgLPtr> left_q_;
    std::deque<MsgRPtr> right_q_;

    std::mutex mutex_;
    Callback callback_;
};

} // namespace ms_fusion

#endif // MS_FUSION_FRAME_SYNCHRONIZER_H
