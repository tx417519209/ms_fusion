#pragma once

// Precompiled header for msf_core translation units (PRIVATE, not propagated downstream).
// 只放稳定、很少改动的头文件：标准库、第三方库，以及几乎每个源文件都用的日志宏。
// 本项目里经常改的头文件（config_params.h、parameters.h 等）不放，否则改一次就全量重编。
// 受 MSF_USE_AXERA_* 控制的 Axera 头文件也不放。
// 源文件仍要显式 include 自己用到的头文件：关闭 MSF_CORE_USE_PCH 时必须能单独编译。

// ---- C++ standard library ----
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ---- Eigen ----
#include <Eigen/Dense>
#include <Eigen/Geometry>

// ---- OpenCV ----
#include <opencv2/opencv.hpp>

// ---- Ceres ----
#include <ceres/ceres.h>

// ---- Sophus ----
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

// ---- TBB / yaml-cpp ----
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>
#include <yaml-cpp/yaml.h>

// ---- msf_core 日志宏（几乎每个源文件都在用）----
#include "ms_fusion/common/runtime_log.h"
