#!/bin/bash
# Board-side step 1/2: extract XFeat descriptors from TUM-VI outdoors4 images
# and dump them to a file. Does NOT cluster into a .dbow3 here -- that step
# (DBoW3::Vocabulary::create, hierarchical k-means) has no iteration cap in
# this repo's vendored DBoW3 and can stall for a very long time on deep tree
# levels once a node's descriptor count gets small relative to the branching
# factor k. Run scripts/dbow3_train_from_features.py on a regular dev machine
# instead (step 2/2) -- same math, sklearn KMeans, much faster there and no
# NPU/ROS2/colcon dependency for that half.
#
# 现成的 xfeat_voc_robot_0729.dbow3 是另一场景训的，跟 outdoors4 户外纹理对不上，
# 检索大概率一直是空的，所以要训一份 outdoors4 专用的。
#
# 图源用 tmp/tumvi_outdoors4/left/*.jpg——这份已经是 640x384 tracking canvas
# 分辨率（跟在线推理输入完全一致，比用 dataset/ 下的原始 512x512 mav0 图省了一次
# 分辨率/宽高比不一致的顾虑），不是 dataset/tumvi_outdoors4/dataset-outdoors4_512_16/
# 那份原始 mav0 数据。
#
# 用法（板子上，install/ 已经 build 过 ms_fusion_loop 且 MSF_BUILD_LOOP_TOOLS=ON）：
#   bash scripts/train_outdoor4_loop_voc.sh
#
# 如果 train_xfeat_voc 还没编译出来，先跑（outdoors4 是离线 bag/topic 回放，
# 要用 build_ms_fusion_offline.sh，不要用 build_ms_fusion.sh——两者
# MSF_USE_AXERA_IVPS 不一样，msf_core/ms_fusion/ms_fusion_loop 必须用同一个值，
# 混用会导致 ImageMessage 内存布局对不上）：
#   bash scripts/build_ms_fusion_offline.sh -DMSF_BUILD_LOOP_TOOLS=ON
# （MSF_USE_AXERA_SUPERPOINT=ON 已经是 build_ms_fusion_offline.sh 的默认值，不用单独加；
#  train_xfeat_voc 只看这个开关，不受 MSF_USE_AXERA_IVPS=OFF 影响，两种 build 都能编出来。
#  这次改动给 train_xfeat_voc 加了 --features_out，需要重新 colcon build 一次
#  ms_fusion_loop 才会生效，即使之前已经编译过。）
#
# 可选环境变量覆盖默认值，例如换 stride/图片上限：
#   STRIDE=10 MAX_IMAGES=2000 bash scripts/train_outdoor4_loop_voc.sh

set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# train_xfeat_voc 链的是 install/msf_core/lib/libmsf_core.so，它自己的 RPATH
# 只覆盖 /soc/lib 和 $ORIGIN/..（即 install/ms_fusion_loop/lib/），不含 msf_core
# 这个兄弟包的 lib 目录，所以必须先 source install/setup.bash 把它加进
# LD_LIBRARY_PATH，否则会报 "cannot open shared object file: libmsf_core.so"。
set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${ROOT}/install/setup.bash"
set -e

# ---- 可调参数 ----
IMAGE_DIR="${IMAGE_DIR:-${ROOT}/tmp/tumvi_outdoors4/left}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/install/ms_fusion/share/ms_fusion/models/xfeat_384x640_maxpool_5x5_top256_npu1.axmodel}"
FEATURES_OUT="${FEATURES_OUT:-${ROOT}/tmp/tumvi_outdoors4/xfeat_voc_outdoors4.msfv}"
SAMPLE_DIR="${SAMPLE_DIR:-${ROOT}/tmp/tumvi_outdoors4/voc_train_images}"
# outdoors4 比 corridor1 长（left/ 下近 14000 帧），全喂进去提特征也会比较慢；
# 隔 STRIDE 帧取一张，保留全程时间跨度覆盖
STRIDE="${STRIDE:-8}"
MAX_IMAGES="${MAX_IMAGES:-0}"   # 0 = 不再额外截断，交给 STRIDE 控制数量
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.005}"

echo "[train_outdoor4_loop_voc] IMAGE_DIR=${IMAGE_DIR}"
echo "[train_outdoor4_loop_voc] MODEL_PATH=${MODEL_PATH}"
echo "[train_outdoor4_loop_voc] FEATURES_OUT=${FEATURES_OUT}"
echo "[train_outdoor4_loop_voc] STRIDE=${STRIDE} MAX_IMAGES=${MAX_IMAGES}"

if [[ ! -d "${IMAGE_DIR}" ]]; then
  echo "[train_outdoor4_loop_voc] ERROR: IMAGE_DIR not found: ${IMAGE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[train_outdoor4_loop_voc] ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  echo "  跟 calib_files/tumvi_outdoors4/msf_robot.yaml 里 tracking.extractor.xfeat.model_path 必须是同一个 .axmodel" >&2
  exit 1
fi

# ---- 找 train_xfeat_voc 可执行文件 ----
TRAIN_BIN=""
for cand in \
  "${ROOT}/install/ms_fusion_loop/lib/ms_fusion_loop/train_xfeat_voc" \
  "$(command -v train_xfeat_voc 2>/dev/null || true)"; do
  if [[ -n "${cand}" && -x "${cand}" ]]; then
    TRAIN_BIN="${cand}"
    break
  fi
done

if [[ -z "${TRAIN_BIN}" ]]; then
  echo "[train_outdoor4_loop_voc] ERROR: 找不到 train_xfeat_voc 可执行文件。" >&2
  echo "  先编译（MSF_BUILD_LOOP_TOOLS 默认关，要显式打开；outdoors4 是离线回放，用 offline 版）：" >&2
  echo "    bash scripts/build_ms_fusion_offline.sh -DMSF_BUILD_LOOP_TOOLS=ON" >&2
  exit 1
fi
echo "[train_outdoor4_loop_voc] using binary: ${TRAIN_BIN}"
echo "[train_outdoor4_loop_voc] (若这个二进制是在加 --features_out 之前编译的，先重新" \
     "colcon build --packages-select ms_fusion_loop 再跑本脚本，否则会报 Unknown argument)"

# ---- 按 stride 抽帧，软链到独立目录（不拷贝原图，省空间） ----
rm -rf "${SAMPLE_DIR}"
mkdir -p "${SAMPLE_DIR}"

mapfile -t all_images < <(find "${IMAGE_DIR}" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort)
total="${#all_images[@]}"
if [[ "${total}" -eq 0 ]]; then
  echo "[train_outdoor4_loop_voc] ERROR: ${IMAGE_DIR} 下没找到 .jpg/.jpeg/.png" >&2
  exit 1
fi

picked=0
for ((i = 0; i < total; i += STRIDE)); do
  src="${all_images[$i]}"
  ln -s "${src}" "${SAMPLE_DIR}/$(printf '%06d' "${picked}")_$(basename "${src}")"
  picked=$((picked + 1))
  if [[ "${MAX_IMAGES}" -gt 0 && "${picked}" -ge "${MAX_IMAGES}" ]]; then
    break
  fi
done
echo "[train_outdoor4_loop_voc] sampled ${picked}/${total} images (stride=${STRIDE}) -> ${SAMPLE_DIR}"

# ---- 提特征，不聚类 ----
mkdir -p "$(dirname "${FEATURES_OUT}")"
"${TRAIN_BIN}" \
  --image_dir "${SAMPLE_DIR}" \
  --model "${MODEL_PATH}" \
  --features_out "${FEATURES_OUT}" \
  --score_threshold "${SCORE_THRESHOLD}"

echo "[train_outdoor4_loop_voc] done -> ${FEATURES_OUT}"
echo "[train_outdoor4_loop_voc] next steps:"
echo "  1) 把 ${FEATURES_OUT} 拷回本地开发机（scp/rsync 都行）。"
echo "  2) 本地跑（需要 numpy + scikit-learn: pip install numpy scikit-learn）："
echo "       python3 scripts/dbow3_train_from_features.py \\"
echo "         --features <本地路径>/xfeat_voc_outdoors4.msfv \\"
echo "         --output ros_wrapper/src/ms_fusion_loop/voc/xfeat_voc_outdoors4.dbow3 \\"
echo "         --k 10 --L 4"
echo "     产出的 .dbow3 再传回板子。"
echo "  3) ms_fusion_loop_node 启动时拿到的 config_file 跟 ms_fusion 主节点是同一份 yaml"
echo "     （见 msf_robot.launch.py: loop_cmd = [ms_fusion_loop_node, config_path]），"
echo "     loop.voc_path 是相对这份 yaml 所在目录解析的（parameters.cpp resolvePath），"
echo "     跟 tracking.extractor.xfeat.model_path 用的是同一条 '../models/' 规则、同一个目录。"
echo "     所以直接把训好的 .dbow3 拷到你实际会启动加载的那份 yaml 对应的 models/ 目录，例如："
echo "       cp xfeat_voc_outdoors4.dbow3 install/ms_fusion/share/ms_fusion/models/"
echo "     不要指望 'colcon build --packages-select ms_fusion_loop' 能把它放到正确位置——"
echo "     那条 target 装的是 install/ms_fusion_loop/share/ms_fusion_loop/voc/，运行时不会去那读。"
echo "  4) 你实际使用的 msf_robot.yaml（calib_files/tumvi_outdoors4/ 下的源文件，或者已装到"
echo "     install/ms_fusion/share/ms_fusion/config/ 的那份，取决于你怎么启动）里改："
echo "       loop.enable: true"
echo "       loop.voc_path: ../models/xfeat_voc_outdoors4.dbow3"
echo "  5) 评测录制话题从 /odometry_vio 换成 /odometry（回环校正后的输出），否则录出来的 est.tum 看不到回环效果"
