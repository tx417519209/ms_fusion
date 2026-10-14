#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"

# ONNX Runtime + CUDA 12.x (e.g. 12.8)
ORT_LIB="${ROOT}/msf_core/third_party/onnxruntime/lib"
export LD_LIBRARY_PATH="${ORT_LIB}:${LD_LIBRARY_PATH:-}"
if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
fi

CUDA_PROVIDER="${ORT_LIB}/libonnxruntime_providers_cuda.so"
if [[ -f "${CUDA_PROVIDER}" ]]; then
  if ! ldd "${CUDA_PROVIDER}" 2>/dev/null | grep -q 'libcudnn.so.9'; then
    echo "[WARN] cuDNN 9 not found for ONNX CUDA EP — will fall back to CPU (slow)."
    echo "       Install e.g.: sudo apt install libcudnn9-cuda-12"
    echo "       Verify: ldd ${CUDA_PROVIDER} | grep cudnn"
  fi
fi

# ReLoc debug: export MSF_RELOC_DEBUG=1
# ONNX GPU: ms_fusion_onnx.yaml onnx.use_cuda, or export MSF_ONNX_USE_CUDA=1
# ONNX perf: MSF_ONNX_INTRA_THREADS, MSF_ONNX_PARALLEL_STEREO, MSF_ONNX_GRAPH_OPT
export MSF_RELOC_DEBUG=0

CONFIG_FILE="${ROOT}/ros_wrapper/src/ms_fusion/config/ms_fusion_onnx.yaml"
CAMERA_FILE="${ROOT}/install/ms_fusion/share/ms_fusion/config/camera_imu.yaml"

TOPIC_IMU="/imu/data"
TOPIC_CAM0="/mipi_image_data_rgb1"
TOPIC_CAM1="/mipi_image_data_rgb0"

ros2 launch ms_fusion ms_fusion_node.launch.py \
  config_file:="${CONFIG_FILE}" \
  camera_file:="${CAMERA_FILE}" \
  topic_imu:="${TOPIC_IMU}" \
  topic_cam0:="${TOPIC_CAM0}" \
  topic_cam1:="${TOPIC_CAM1}"
