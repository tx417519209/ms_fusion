#!/bin/bash
# Unified entry: uses install/.../config/msf_robot.yaml
# (copy a profile from calib_files/<name>/ onto the install path before run)
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL="${MSF_INSTALL_ROOT:-${ROOT}/install}"

cd "${ROOT}"
set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${INSTALL}/setup.bash"
set -e

export HOME="${HOME:-/root}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${HOME}/.ros/log}"
mkdir -p "${ROS_LOG_DIR}"

exec ros2 launch ms_fusion msf_robot.launch.py "$@"
