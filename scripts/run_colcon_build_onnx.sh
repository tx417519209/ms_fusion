#!/bin/bash
# Build ms_fusion with ONNX Runtime (x86). Default: GPU / CUDA 12.x package.
#
#   bash scripts/run_colcon_build_onnx.sh
#
# CPU-only ORT (no CUDA):
#   MSF_ONNX_GPU=0 bash scripts/run_colcon_build_onnx.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

export MSF_ONNX_GPU="${MSF_ONNX_GPU:-1}"

ORT_ROOT="${ROOT}/msf_core/third_party/onnxruntime"
if [[ "${MSF_ONNX_GPU}" == "1" ]]; then
  export ORT_VARIANT="${ORT_VARIANT:-gpu}"
else
  export ORT_VARIANT="${ORT_VARIANT:-cpu}"
fi

if [[ ! -f "${ORT_ROOT}/include/onnxruntime_cxx_api.h" ]] ||
   [[ "${ORT_VARIANT}" == "gpu" && ! -f "${ORT_ROOT}/lib/libonnxruntime_providers_cuda.so" ]]; then
  echo "Installing ONNX Runtime (${ORT_VARIANT}) to ${ORT_ROOT} ..."
  ORT_VARIANT="${ORT_VARIANT}" bash "${ROOT}/scripts/install_onnxruntime.sh"
fi

export ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-${ORT_ROOT}}"

CMAKE_EXTRA=(-DMSF_INFER_BACKEND=onnx)
if [[ "${MSF_ONNX_GPU}" == "1" ]]; then
  CMAKE_EXTRA+=(-DMSF_ONNX_CUDA=ON)
fi

echo "MSF_ONNX_GPU=${MSF_ONNX_GPU} ORT_VARIANT=${ORT_VARIANT}"
colcon build --cmake-args "${CMAKE_EXTRA[@]}" "$@"
