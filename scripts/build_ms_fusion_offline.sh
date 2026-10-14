#!/bin/bash
# Offline ROS2 bag/topic build: ImageMessage carries cv::Mat, never IvpsFrameHandle.
# Do not use --symlink-install: deployed install trees must not link back to build/.
#
# Usage: bash scripts/build_ms_fusion_offline.sh [extra colcon --cmake-args]
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
set -e

# Keep the Axera NPU feature/reloc implementations available for board-side bag replay.
# MSF_USE_AXERA_IVPS must be identical for msf_core, ms_fusion and ms_fusion_loop because
# ImageMessage changes from IvpsFramePtr to cv::Mat at this compile-time boundary.
# -j10: swap filled up during a -j16 build, likely too many qemu-emulated PCH compiles at once.
export MAKEFLAGS="-j10"
export CMAKE_BUILD_PARALLEL_LEVEL=10

# Build core once as an exported package; offline ms_fusion and optional loop tools reuse it.
# Without CMAKE_BUILD_TYPE nothing passes -O, so Ceres AutoDiff cost functors build at -O0.
colcon build --packages-select msf_core ms_fusion ms_fusion_loop \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra" \
  -DMSF_USE_AXERA_IVPS=OFF \
  -DMSF_USE_AXERA_SUPERPOINT=ON \
  "$@"

if [[ "${PACK:-0}" == "1" ]]; then
  tar -C install -czf court_keypoints_install_offline.tar.gz .
  echo "packed: ${ROOT}/court_keypoints_install_offline.tar.gz"
fi
