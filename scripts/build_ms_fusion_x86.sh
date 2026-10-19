#!/bin/bash
# x86/amd64 CPU build using ONNX Runtime. Images are received from ROS topics.
# Usage: bash scripts/build_ms_fusion_x86.sh [extra colcon --cmake-args]
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
set -e

BUILD_JOBS="${BUILD_JOBS:-10}"
export MAKEFLAGS="-j${BUILD_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"

# Ceres 2.2.0 使用仓库内 third_party/ceres/x86_64（apt 的 2.0 没有 ceres::Manifold）。
# 同时传给 msf_core / ms_fusion / ms_fusion_loop，避免下游包通过导出依赖找到系统的 2.0。
CERES_DIR="${ROOT}/third_party/ceres/x86_64/lib/cmake/Ceres"

colcon build --packages-select msf_core ms_fusion ms_fusion_loop \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DMSF_USE_AXERA_IVPS=OFF \
  -DMSF_USE_AXERA_SUPERPOINT=OFF \
  -DCeres_DIR="${CERES_DIR}" \
  -DUSE_ONNX=ON \
  -DMSF_CORE_USE_PCH=OFF \
  -DMSF_USE_PCH=OFF \
  "$@"

echo "x86 build complete. Source install/setup.bash before running."
