#!/bin/bash
# 不要用 --symlink-install，否则 install 链到 build 路径，scp 到板端会坏
#
# 用法: bash scripts/build_ms_fusion.sh
#
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
set -e

# 旧版 colcon 无 --build-args；用环境变量让 cmake --build / make 并行 -j10（-j16 时观察到 swap 被占满，疑似 qemu 下多路 PCH 编译内存不足）
export MAKEFLAGS="-j10"
export CMAKE_BUILD_PARALLEL_LEVEL=10

# Build core once as an exported package; ms_fusion and optional loop tools link its target.
# Without CMAKE_BUILD_TYPE nothing passes -O, so Ceres AutoDiff cost functors build at -O0.
colcon build --packages-select msf_core ms_fusion ms_fusion_loop \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra" \
  -DMSF_USE_AXERA_IVPS=ON \
  -DMSF_USE_AXERA_SUPERPOINT=ON \
  "$@"

if [[ "${PACK:-0}" == "1" ]]; then
  tar -C install -czf court_keypoints_install.tar.gz .
  echo "packed: ${ROOT}/court_keypoints_install.tar.gz"
fi
