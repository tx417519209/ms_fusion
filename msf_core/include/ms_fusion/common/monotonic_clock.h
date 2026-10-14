#ifndef MS_FUSION_MONOTONIC_CLOCK_H
#define MS_FUSION_MONOTONIC_CLOCK_H

#include <chrono>
#include <cstdint>

namespace ms_fusion {

// Linux CLOCK_MONOTONIC_RAW — aligned with BSP clock_gettime(CLOCK_MONOTONIC_RAW).
struct MonotonicRawClock {
    using rep = int64_t;
    using period = std::nano;
    using duration = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<MonotonicRawClock, duration>;

    static time_point now() noexcept;

    static constexpr bool is_steady = true;
};

inline int64_t monotonicRawNowNs() noexcept {
    return MonotonicRawClock::now().time_since_epoch().count();
}

inline int64_t monotonicRawNowMs() noexcept {
    return monotonicRawNowNs() / 1000000LL;
}

} // namespace ms_fusion

#endif // MS_FUSION_MONOTONIC_CLOCK_H
