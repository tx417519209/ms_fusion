#!/bin/bash
# Download ONNX Runtime C++ prebuilt (x86_64 Linux) into msf_core/third_party/onnxruntime
#
# CPU (default):
#   bash scripts/install_onnxruntime.sh
#
# GPU / CUDA EP (CUDA 12.x, e.g. 12.8 — match your driver + ORT GPU package):
#   ORT_VARIANT=gpu bash scripts/install_onnxruntime.sh
#   # or: bash scripts/install_onnxruntime_gpu.sh
#
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.20.1}"
ORT_ARCH="${ORT_ARCH:-x64}"
ORT_VARIANT="${ORT_VARIANT:-cpu}"  # cpu | gpu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_ROOT="${SCRIPT_DIR}/../msf_core/third_party/onnxruntime"

if [[ "${ORT_VARIANT}" == "gpu" ]]; then
  ARCHIVE_NAME="onnxruntime-linux-${ORT_ARCH}-gpu-${ORT_VERSION}"
  MARKER_LIB="libonnxruntime_providers_cuda.so"
else
  ARCHIVE_NAME="onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}"
  MARKER_LIB="libonnxruntime.so"
fi

URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARCHIVE_NAME}.tgz"

if [[ -f "${DEST_ROOT}/include/onnxruntime_cxx_api.h" && -f "${DEST_ROOT}/lib/${MARKER_LIB}" ]]; then
  echo "ONNX Runtime (${ORT_VARIANT}) already installed at ${DEST_ROOT}"
  exit 0
fi

if [[ -d "${DEST_ROOT}" ]]; then
  echo "Replacing existing ONNX Runtime at ${DEST_ROOT} (switching to ${ORT_VARIANT}) ..."
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Downloading ${URL} ..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -o "${TMP_DIR}/ort.tgz" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "${TMP_DIR}/ort.tgz" "${URL}"
else
  echo "Error: need curl or wget" >&2
  exit 1
fi

tar -xzf "${TMP_DIR}/ort.tgz" -C "${TMP_DIR}"
SRC="${TMP_DIR}/${ARCHIVE_NAME}"
if [[ ! -d "${SRC}" ]]; then
  echo "Error: unexpected archive layout under ${TMP_DIR}" >&2
  ls -la "${TMP_DIR}" >&2
  exit 1
fi

rm -rf "${DEST_ROOT}"
mkdir -p "$(dirname "${DEST_ROOT}")"
mv "${SRC}" "${DEST_ROOT}"

echo "Installed ONNX Runtime ${ORT_VERSION} (${ORT_VARIANT}) -> ${DEST_ROOT}"
if [[ "${ORT_VARIANT}" == "gpu" ]]; then
  echo "GPU libs:"
  ls -la "${DEST_ROOT}/lib/"libonnxruntime*.so 2>/dev/null || true
  echo "Ensure CUDA 12.x is on LD_LIBRARY_PATH, e.g.:"
  echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi
echo "Export: export ONNXRUNTIME_ROOT=${DEST_ROOT}"
