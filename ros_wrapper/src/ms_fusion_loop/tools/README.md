# Vocabulary / DBoW tools (Axera SoC)

**Run on board.** Collect images and train vocab there; deploy `install/` like `ms_fusion`.

## 1) Collect gray PNGs → local folder

Stop `ms_fusion` first (same IVPS channels). Camera/IVPS pyramid must be running.

```bash
ros2 run ms_fusion_loop collect_gray_images -- \
  --output_dir /data/voc_images \
  --interval_ms 200 \
  --max_frames 2000
# Ctrl-C to stop early; default left_grp=7 right_grp=8
```

Writes `gray_000000.png`, … under `--output_dir` (tracking mono, typically 640×384).

## 2) Train vocabulary from that folder

Write into source `voc/` (then colcon installs to `share/ms_fusion/models/`):

```bash
bash scripts/dbow_step2_train_dbow_xfeat.sh
# → .../ms_fusion_loop/voc/xfeat_voc.dbow3

colcon build --packages-select ms_fusion
# → install/ms_fusion/share/ms_fusion/models/xfeat_voc.dbow3
```

Config: `loop.voc_path: ../models/xfeat_voc.dbow3` (relative to install config/). Must match `tracking.feature_detector`.

## Build

```bash
bash scripts/build_ms_fusion.sh -DMSF_BUILD_LOOP_TOOLS=ON
# also enables the Axera IVPS / SuperPoint-XFeat build switches
```

The default production build only builds `ms_fusion_loop_node`; it does not
build these offline collection and vocabulary-training tools.

## Notes

- Collect needs IVPS only; train needs NPU (`.axmodel`).
- SuperPoint / XFeat vocabs are not interchangeable.
- Clustering (`DBoW3::create`) can take a long time on large sets.
