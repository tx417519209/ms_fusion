#include "ms_fusion/common/monotonic_clock.h"

#include <ctime>

namespace ms_fusion {

MonotonicRawClock::time_point MonotonicRawClock::now() noexcept {
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return time_point(duration(static_cast<rep>(ts.tv_sec) * 1000000000LL +
                               static_cast<rep>(ts.tv_nsec)));
}

} // namespace ms_fusion
