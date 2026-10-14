#!/bin/bash
# 一键离线回放评测：部署 profile → 启动 VIO(+Loop) → 录制 /odometry_vio 与 /odometry → 按原始时基回放。
# 产物（文件名均带数据集前缀）：
#   <out-dir>/<dataset>_<时间戳>.log                 ms_fusion / ms_fusion_loop 运行日志 (runtime_log/ 拷贝)
#   <out-dir>/<dataset>_<时间戳>_replay.log          回放器输出 (含 max_lag / n_behind)
#   <out-dir>/<dataset>_<时间戳>_vio_est.tum         /odometry_vio  原始 VIO
#   <out-dir>/<dataset>_<时间戳>_loop_est.tum        /odometry      回环校正输出 (profile 里 loop.enable=true 才有)
#
# 用法:
#   scripts/run_offline_eval.sh <dataset> [选项]
#     <dataset>  数据集目录，或 tmp/ 下的目录名 (如 tumvi_room1)
#   --profile NAME   calib_files/ 下的 profile 目录名；缺省按数据集名推断
#   --out-dir DIR    产物目录，缺省 <工程根>/statistics/；也可用环境变量 MSF_EVAL_OUT
#   --speed X        回放倍速，缺省 1.0 (评测请保持 1.0)
#   --settle-s N     回放结束后等待 N 秒再停止 (让回环/位姿图优化收尾)，缺省 15
#   --ready-timeout-s N  等待 ms_fusion_node 订阅就绪的超时，缺省 180
#   --no-deploy      不拷贝 profile，沿用 install 里已有的 <profile>.yaml
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL="${MSF_INSTALL_ROOT:-${ROOT}/install}"

list_datasets() {
  echo "可用数据集 (${ROOT}/tmp/ 下含 imu_data.txt 与 image_timestamps_left.txt 的目录):"
  local d n found=0
  for d in "${ROOT}"/tmp/*/; do
    [[ -f "${d}imu_data.txt" && -f "${d}image_timestamps_left.txt" ]] || continue
    n="$(basename "${d}")"; found=1
    printf '  %-44s profile=%-18s %s\n' "${n}" "$(infer_profile "${n}")" \
      "$([[ -f ${d}mocap_data.txt ]] && echo '有真值' || echo '无真值')"
  done
  [[ "${found}" == 1 ]] || echo "  (无)"
}
infer_profile() {
  case "$1" in
    tumvi_*)  echo "$1" ;;
    *)        echo "?(需 --profile)" ;;
  esac
}
usage() {
  sed -n '2,/^set -eo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  echo; list_datasets
  exit "${1:-1}"
}

DATASET="" PROFILE="" OUT_DIR="${MSF_EVAL_OUT:-${ROOT}/statistics}" SPEED=1.0 SETTLE=15 READY_TIMEOUT=180 DEPLOY=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --speed) SPEED="$2"; shift 2 ;;
    --settle-s) SETTLE="$2"; shift 2 ;;
    --ready-timeout-s) READY_TIMEOUT="$2"; shift 2 ;;
    --no-deploy) DEPLOY=0; shift ;;
    -h|--help|--list) usage 0 ;;
    -*) echo "未知选项: $1" >&2; usage 1 ;;
    *) [[ -z "${DATASET}" ]] || { echo "只能给一个数据集" >&2; exit 1; }; DATASET="$1"; shift ;;
  esac
done
[[ -n "${DATASET}" ]] || usage 1

# ---- 数据集 ----
if [[ -d "${DATASET}" ]]; then
  DATASET_DIR="$(cd "${DATASET}" && pwd)"
elif [[ -d "${ROOT}/tmp/${DATASET}" ]]; then
  DATASET_DIR="${ROOT}/tmp/${DATASET}"
else
  echo "找不到数据集: ${DATASET} (也不在 ${ROOT}/tmp/ 下)" >&2; list_datasets >&2; exit 1
fi
NAME="$(basename "${DATASET_DIR}")"
for f in imu_data.txt image_timestamps_left.txt left right; do
  [[ -e "${DATASET_DIR}/${f}" ]] || { echo "数据集缺少 ${f}: ${DATASET_DIR}" >&2; exit 1; }
done

# ---- profile ----
if [[ -z "${PROFILE}" ]]; then
  PROFILE="$(infer_profile "${NAME}")"
  [[ "${PROFILE}" != \?* ]] || { echo "无法从数据集名 ${NAME} 推断 profile，请用 --profile 指定" >&2; exit 1; }
fi
PROFILE_SRC="${ROOT}/calib_files/${PROFILE}/msf_robot.yaml"
[[ -f "${PROFILE_SRC}" ]] || { echo "profile 不存在: ${PROFILE_SRC}" >&2; exit 1; }
PROFILE_DST="${INSTALL}/ms_fusion/share/ms_fusion/config/${PROFILE}.yaml"

# ---- ROS 环境 ----
set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
[[ -f "${INSTALL}/setup.bash" ]] || { echo "未找到 ${INSTALL}/setup.bash，请先构建/安装" >&2; exit 1; }
source "${INSTALL}/setup.bash"
set -e
export HOME="${HOME:-/root}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${HOME}/.ros/log}"
mkdir -p "${ROS_LOG_DIR}" "${OUT_DIR}"

# ---- 部署 profile ----
if [[ "${DEPLOY}" == 1 ]]; then
  cp "${PROFILE_SRC}" "${PROFILE_DST}"
  echo "[run] 已部署 profile: ${PROFILE_SRC} -> ${PROFILE_DST}"
else
  [[ -f "${PROFILE_DST}" ]] || { echo "--no-deploy 但 ${PROFILE_DST} 不存在" >&2; exit 1; }
fi

# 从实际生效的 yaml（install 里的 ${PROFILE_DST}；--no-deploy 时它可能和源码不同）读话题与 loop 开关
read -r IMU_TOPIC ODOM_TOPIC CAM0_TOPIC CAM1_TOPIC VIO_TOPIC LOOP_ENABLE < <(python3 - "${PROFILE_DST}" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
g = lambda *k: __import__("functools").reduce(lambda a, b: a[b], k, d)
print(g("imu", "topic"), g("odom", "topic"),
      g("camera", "cam0", "offline", "topic"), g("camera", "cam1", "offline", "topic"),
      g("base", "odometry_topic"), str(bool(g("loop", "enable"))).lower())
PY
)

TS="$(date +%Y%m%d_%H%M%S)"
PREFIX="${OUT_DIR}/${NAME}_${TS}"
VIO_TUM="${PREFIX}_vio_est.tum"
LOOP_TUM="${PREFIX}_loop_est.tum"
RUN_LOG="${PREFIX}.log"
REPLAY_LOG="${PREFIX}_replay.log"

# runtime_log/ 写在启动目录，在 ROOT 下启动，之后按启动时刻挑出本次的日志
cd "${ROOT}"
MARK="$(mktemp)"

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for p in "${PIDS[@]}"; do kill -INT "-${p}" 2>/dev/null || kill -INT "${p}" 2>/dev/null || true; done
  for p in "${PIDS[@]}"; do wait "${p}" 2>/dev/null || true; done
  rm -f "${MARK}"
}
trap cleanup EXIT INT TERM

# ---- 启动 VIO (+Loop)；setsid 让整个 launch 进程组可一起收尾 ----
LAUNCH_LOG="${PREFIX}_launch.log"
setsid ros2 launch ms_fusion msf_robot.launch.py "config_file:=${PROFILE_DST}" >"${LAUNCH_LOG}" 2>&1 &
LAUNCH_PID=$!
echo "[run] launch pid=${LAUNCH_PID}, 控制台输出: ${LAUNCH_LOG}"

# ---- 录制 ----
RECORD_PIDS=()
setsid python3 "${ROOT}/scripts/tumvie_eval_trajectory.py" record --topic "${VIO_TOPIC}" --out "${VIO_TUM}" \
  >"${PREFIX}_rec_vio.log" 2>&1 &
RECORD_PIDS+=($!)
if [[ "${LOOP_ENABLE}" == true ]]; then
  setsid python3 "${ROOT}/scripts/tumvie_eval_trajectory.py" record --topic /odometry --out "${LOOP_TUM}" \
    >"${PREFIX}_rec_loop.log" 2>&1 &
  RECORD_PIDS+=($!)
else
  echo "[run] profile 里 loop.enable=false，不录制 /odometry"
fi
PIDS=("${RECORD_PIDS[@]}" "${LAUNCH_PID}")

# ---- 等 ms_fusion_node 订阅 IMU 话题 ----
echo "[run] 等待 ms_fusion_node 订阅 ${IMU_TOPIC} (最长 ${READY_TIMEOUT}s) ..."
ready=0
for ((i = 0; i < READY_TIMEOUT; i++)); do
  kill -0 "${LAUNCH_PID}" 2>/dev/null || { echo "launch 提前退出，见 ${LAUNCH_LOG}" >&2; exit 1; }
  info="$(ros2 topic info "${IMU_TOPIC}" 2>/dev/null || true)"
  if grep -Eq 'Subscription count: [1-9]' <<<"${info}"; then ready=1; break; fi
  sleep 1
done
[[ "${ready}" == 1 ]] || { echo "超时：ms_fusion_node 未订阅 ${IMU_TOPIC}，见 ${LAUNCH_LOG}" >&2; exit 1; }
echo "[run] 已就绪 (${i}s)，开始回放"

# ---- 按原始时基回放 ----
python3 "${ROOT}/scripts/tumvie_replay_node.py" "${DATASET_DIR}" \
  --imu-topic "${IMU_TOPIC}" --odom-topic "${ODOM_TOPIC}" \
  --cam0-topic "${CAM0_TOPIC}" --cam1-topic "${CAM1_TOPIC}" \
  --speed "${SPEED}" --start-delay-s 3 2>&1 | tee "${REPLAY_LOG}"
echo "[run] 回放结束，等待 ${SETTLE}s 让回环/位姿图收尾"
sleep "${SETTLE}"

# ---- 收尾：先停录制（写 tum），再停 launch ----
for p in "${RECORD_PIDS[@]}"; do kill -INT "-${p}" 2>/dev/null || kill -INT "${p}" 2>/dev/null || true; done
for p in "${RECORD_PIDS[@]}"; do wait "${p}" 2>/dev/null || true; done
kill -INT "-${LAUNCH_PID}" 2>/dev/null || true
wait "${LAUNCH_PID}" 2>/dev/null || true
PIDS=()

for f in "${VIO_TUM}" $([[ "${LOOP_ENABLE}" == true ]] && echo "${LOOP_TUM}"); do
  [[ -s "${f}" ]] || echo "[run] 警告：${f} 缺失或为空，见 ${PREFIX}_rec_*.log" >&2
done

# ---- 拷贝本次运行日志 (runtime_log/ 里启动后新建的最新文件) ----
SRC_LOG="$(find "${ROOT}/runtime_log" -maxdepth 1 -name '*.log' -newer "${MARK}" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -n "${SRC_LOG}" ]]; then cp "${SRC_LOG}" "${RUN_LOG}"; else echo "[run] 警告：未找到本次 runtime_log" >&2; fi

echo
echo "[run] 完成，产物:"
ls -l "${PREFIX}"* | awk '{print "  " $5 "\t" $9}'
grep -h "max_lag" "${REPLAY_LOG}" 2>/dev/null | tr '\r' '\n' | grep done | tail -n 1 | sed 's/^/[replay] /' || true
