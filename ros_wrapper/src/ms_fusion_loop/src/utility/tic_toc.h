/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <cstdint>
#include <ctime>

// Wall-clock adjustments (NTP, settimeofday) must never affect latency
// measurements. This also matches the raw clock used by ms_fusion core.
inline int64_t loopMonotonicRawNowNs() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL +
         static_cast<int64_t>(ts.tv_nsec);
}

class TicToc {
 public:
  TicToc() { tic(); }

  void tic() { start_ns_ = loopMonotonicRawNowNs(); }

  double toc() const {
    return static_cast<double>(loopMonotonicRawNowNs() - start_ns_) * 1e-6;
  }

 private:
  int64_t start_ns_ = 0;
};
