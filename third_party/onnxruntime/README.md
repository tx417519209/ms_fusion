# ONNX Runtime backend

This directory contains the ONNX Runtime 1.23.2 CPU shared library and C/C++ headers used by
`-DUSE_ONNX=ON`, plus the x86 models installed with `ms_fusion`.

Build with:

```bash
bash scripts/build_ms_fusion_x86.sh
```

The x86 build disables Axera IVPS/NPU code. Existing YAML files remain usable: at load time the
XFeat and YOLO-OBB `.axmodel` names are mapped to the corresponding `.onnx` files installed in the
same model directory. LightGlue currently falls back to the existing CPU MNN matcher on x86.
