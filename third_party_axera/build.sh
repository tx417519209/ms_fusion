#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH

# ROS2 env: override with ROS_SETUP=/path/to/setup.bash if needed
if [ -n "${ROS_SETUP:-}" ] && [ -f "${ROS_SETUP}" ]; then
    source "${ROS_SETUP}"
elif [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo "Warning: ROS setup.bash not found; ROS targets may be skipped" >&2
fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. "$@"
if [ -n "${BUILD_TARGET}" ]; then
    make "${BUILD_TARGET}" -j8
else
    make -j8
fi
