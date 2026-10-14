# Vocabulary for ms_fusion_loop (DBoW3 + SuperPoint / XFeat)

Place trained `*.dbow3` files here (e.g. `xfeat_voc.dbow3`).

`ms_fusion` CMake installs them to:

```text
install/ms_fusion/share/ms_fusion/models/
```

Config: `loop.voc_path: ../models/xfeat_voc.dbow3` (relative to share config/).

## Build vocab on SoC

1. Collect gray PNGs (stop `ms_fusion` first):

```bash
bash scripts/dbow_step1_collect_gray_images.sh
```

2. Train into this `voc/` directory:

```bash
bash scripts/dbow_step2_train_dbow_xfeat.sh
```

3. Install via colcon:

```bash
colcon build --packages-select ms_fusion
```

Details: [tools/README.md](../tools/README.md).
