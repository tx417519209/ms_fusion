#!/bin/bash
# Install ONNX Runtime GPU package (CUDA EP) for CUDA 12.x
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_VARIANT=gpu exec bash "${SCRIPT_DIR}/install_onnxruntime.sh"
