#!/usr/bin/env bash
# ==============================================================================
# VINS-Fusion 自動化評估腳本 - 支援 FOG/IMU 雙模式切換
# ML pipeline 失敗不會讓 VINS 主流程被判定失敗
# ==============================================================================
set -eo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <SEQ_NAME> <SEQ_ROOT> <PLAYER_CONFIG_TEMPLATE> <GT_TUM> <CONFIG_MODE> [VINS_IMU_NOISE_MODE]"
  exit 1
fi

SEQ_NAME="$1"
SEQ_ROOT="$2"
PLAYER_CONFIG_TEMPLATE="$3"
GT_TUM="$4"
CONFIG_MODE="$5"
VINS_IMU_NOISE_MODE="${6:-}"

SEQ_BASENAME="$(basename "$SEQ_ROOT")"

case "${CONFIG_MODE}" in
  fog)
    PLAYER_IMU_SOURCE="fog_xsens"
    DEFAULT_VINS_IMU_NOISE_MODE="fog_xsens"
    ;;
  imu)
    PLAYER_IMU_SOURCE="xsens"
    DEFAULT_VINS_IMU_NOISE_MODE="xsens"
    ;;
  *)
    echo "[ERROR] CONFIG_MODE must be fog or imu, got: ${CONFIG_MODE}"
    exit 1
    ;;
esac

if [[ -z "${VINS_IMU_NOISE_MODE}" ]]; then
  VINS_IMU_NOISE_MODE="${DEFAULT_VINS_IMU_NOISE_MODE}"
fi

case "${VINS_IMU_NOISE_MODE}" in
  fog_xsens|xsens|custom)
    ;;
  *)
    echo "[ERROR] VINS_IMU_NOISE_MODE must be fog_xsens, xsens, or custom, got: ${VINS_IMU_NOISE_MODE}"
    exit 1
    ;;
esac

if [[ ! -d "${SEQ_ROOT}" ]]; then
  echo "[ERROR] SEQ_ROOT not found: ${SEQ_ROOT}"
  exit 1
fi

if [[ ! -f "${PLAYER_CONFIG_TEMPLATE}" ]]; then
  echo "[ERROR] PLAYER_CONFIG_TEMPLATE not found: ${PLAYER_CONFIG_TEMPLATE}"
  exit 1
fi

# =========================================================
# GT TUM 自動轉檔
# =========================================================
GT_POSE_DIR="$(dirname "$GT_TUM")"
FALLBACK_GT_TUM_SCRIPT="/mnt/sata4t/datasets/kaist_complex_urban/extracted/urban28-pankyo/pose/urban28-pankyo/global_pose_to_tum.py"

if [[ ! -f "$GT_TUM" ]]; then
  if [[ -f "$GT_POSE_DIR/global_pose.csv" ]]; then
    echo "[INFO] GT_TUM 不存在，嘗試由 global_pose.csv 轉檔..."
    [[ ! -f "$GT_POSE_DIR/global_pose_to_tum.py" ]] && cp "$FALLBACK_GT_TUM_SCRIPT" "$GT_POSE_DIR/global_pose_to_tum.py"
    ( cd "$GT_POSE_DIR" && python3 global_pose_to_tum.py )
  fi
fi

if [[ ! -f "$GT_TUM" ]]; then
  echo "[ERROR] GT_TUM still not found: ${GT_TUM}"
  exit 1
fi

# =========================================================
# 路徑設定
# =========================================================
SATA_BASE="/mnt/sata4t/ivlab3_data/vins_project"
VINS_WS="${HOME}/vins_ws"
PIPELINE_DIR="${HOME}/Lucas_ws/Visual/keyframe_pipeline"
VIO_TO_TUM="${SATA_BASE}/tools/vio_csv_to_tum.py"

VINS_CONFIG="${VINS_WS}/src/VINS-Fusion-ROS2-jazzy/config/kaist/kaist_stereo_xsens.yaml"
VINS_CONFIG_DIR="$(dirname "${VINS_CONFIG}")"

TMP_VINS_CONFIG="${VINS_CONFIG_DIR}/vins_config_tmp_${CONFIG_MODE}_${VINS_IMU_NOISE_MODE}_${SEQ_NAME}.yaml"

RESULT_ROOT_SAT="${SATA_BASE}/results/kaist/${CONFIG_MODE}/${SEQ_NAME}"
RUN_DIR="${RESULT_ROOT_SAT}/vins_raw"
FEATURE_DIR="${RESULT_ROOT_SAT}/admission/features"
LABEL_DIR="${RESULT_ROOT_SAT}/admission/labels"
DATASET_DIR="${RESULT_ROOT_SAT}/admission/datasets"
LOG_DIR="${RESULT_ROOT_SAT}/logs"
METRIC_DIR="${RESULT_ROOT_SAT}/metrics"

FEATURE_CSV="${FEATURE_DIR}/reliability_features_vins.csv"
TMP_PLAYER_CONFIG="${LOG_DIR}/kaist_player_config.yaml"

mkdir -p \
  "${RUN_DIR}/pose_graph" \
  "${FEATURE_DIR}" \
  "${LABEL_DIR}" \
  "${DATASET_DIR}" \
  "${LOG_DIR}" \
  "${METRIC_DIR}"

export REL_RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
export REL_DATASET_NAME="kaist"
export REL_SEQUENCE_NAME="${SEQ_NAME}"
export REL_FEATURE_CSV_PATH="${FEATURE_CSV}"

rm -f "${FEATURE_CSV}" "${TMP_PLAYER_CONFIG}" "${TMP_VINS_CONFIG}"
rm -f "${RUN_DIR}/"*.csv "${RUN_DIR}/"*.tum
rm -f "${LOG_DIR}/"*.log

# =========================================================
# 動態產生 player config + VINS config
# =========================================================
python3 - <<PY
from pathlib import Path
import re

seq_root = r"${SEQ_ROOT}"
seq_basename = r"${SEQ_BASENAME}"

player_template = Path(r"${PLAYER_CONFIG_TEMPLATE}")
tmp_player_config = Path(r"${TMP_PLAYER_CONFIG}")

vins_config = Path(r"${VINS_CONFIG}")
tmp_vins_config = Path(r"${TMP_VINS_CONFIG}")

run_dir = r"${RUN_DIR}"
player_imu_source = r"${PLAYER_IMU_SOURCE}"
vins_imu_noise_mode = r"${VINS_IMU_NOISE_MODE}"

def set_yaml_string(text: str, key: str, value: str) -> str:
    pattern = rf'^(\s*{re.escape(key)}\s*:\s*).*$'
    replacement = rf'\1"{value}"'

    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)

    if not text.endswith("\\n"):
        text += "\\n"

    return text + f'{key}: "{value}"\\n'

def replace_seq_placeholders(text: str) -> str:
    text = text.replace(
        "/mnt/sata4t/datasets/kaist_complex_urban/extracted/urban28-pankyo",
        seq_root
    )
    text = text.replace("urban28-pankyo", seq_basename)
    text = text.replace("urban28_pankyo", seq_basename.replace("-", "_"))
    return text

p_text = player_template.read_text()
p_text = replace_seq_placeholders(p_text)
p_text = set_yaml_string(p_text, "dataset_root", seq_root)
p_text = set_yaml_string(p_text, "imu_source", player_imu_source)
tmp_player_config.write_text(p_text)

v_text = vins_config.read_text()
v_text = set_yaml_string(v_text, "output_path", f"{run_dir}/")
v_text = set_yaml_string(v_text, "pose_graph_save_path", f"{run_dir}/pose_graph/")
v_text = set_yaml_string(v_text, "imu_noise_mode", vins_imu_noise_mode)
tmp_vins_config.write_text(v_text)

print("[INFO] Generated configs")
print(f"[INFO] CONFIG_MODE              = {r'${CONFIG_MODE}'}")
print(f"[INFO] PLAYER_IMU_SOURCE        = {player_imu_source}")
print(f"[INFO] VINS_IMU_NOISE_MODE      = {vins_imu_noise_mode}")
print(f"[INFO] TMP_PLAYER_CONFIG        = {tmp_player_config}")
print(f"[INFO] TMP_VINS_CONFIG          = {tmp_vins_config}")
PY

echo "------------------------------------------------------------"
echo ">>> 執行序列: ${SEQ_NAME}"
echo "    CONFIG_MODE         : ${CONFIG_MODE}"
echo "    PLAYER_IMU_SOURCE   : ${PLAYER_IMU_SOURCE}"
echo "    VINS_IMU_NOISE_MODE : ${VINS_IMU_NOISE_MODE}"
echo "    RESULT_ROOT         : ${RESULT_ROOT_SAT}"
echo "------------------------------------------------------------"

cleanup() {
  set +e
  echo "[INFO] cleanup..."
  pkill -P $$ 2>/dev/null || true
  rm -f "${TMP_VINS_CONFIG}"
}
trap cleanup EXIT

source /opt/ros/jazzy/setup.bash
source "${VINS_WS}/install/setup.bash"

# =========================================================
# 啟動系統
# =========================================================
ros2 run loop_fusion loop_fusion_node "${TMP_VINS_CONFIG}" \
  --ros-args -p use_sim_time:=true \
  > "${LOG_DIR}/loop_fusion.log" 2>&1 &
LOOP_PID=$!

sleep 2

ros2 run vins vins_node "${TMP_VINS_CONFIG}" \
  --ros-args -p use_sim_time:=true \
  > "${LOG_DIR}/vins.log" 2>&1 &
VINS_PID=$!

sleep 5

ros2 launch kaist_player kaist_player.launch.py \
  config_file:="${TMP_PLAYER_CONFIG}" \
  dataset_root:="${SEQ_ROOT}" \
  > "${LOG_DIR}/player.log" 2>&1 &
PLAYER_PID=$!

# =========================================================
# 監控播放進度
# =========================================================
while true; do
  if grep -q "Playback finished\." "${LOG_DIR}/player.log" 2>/dev/null; then
    break
  fi

  if ! kill -0 "${PLAYER_PID}" 2>/dev/null; then
    echo "[ERROR] kaist_player died unexpectedly"
    tail -n 80 "${LOG_DIR}/player.log" || true
    exit 1
  fi

  if ! kill -0 "${VINS_PID}" 2>/dev/null; then
    echo "[ERROR] vins_node died unexpectedly"
    tail -n 80 "${LOG_DIR}/vins.log" || true
    exit 1
  fi

  sleep 5
done

echo "[INFO] 播放完成，等待 VINS 寫檔..."
sleep 20

kill -INT "${PLAYER_PID}" "${VINS_PID}" "${LOOP_PID}" 2>/dev/null || true
sleep 15

# =========================================================
# 後處理與評估
# =========================================================
if [[ -s "${RUN_DIR}/vio.csv" ]]; then
  python3 "${VIO_TO_TUM}" "${RUN_DIR}/vio.csv" "${RUN_DIR}/vio.tum"

  if [[ -f "${RUN_DIR}/vio_loop.csv" ]]; then
    python3 "${VIO_TO_TUM}" "${RUN_DIR}/vio_loop.csv" "${RUN_DIR}/vio_loop.tum"
  fi

  if command -v evo_ape >/dev/null 2>&1; then
    evo_ape tum "${GT_TUM}" "${RUN_DIR}/vio.tum" \
      -va -a --t_max_diff 0.05 \
      | tee "${METRIC_DIR}/ape_vio.txt" || true

    evo_rpe tum "${GT_TUM}" "${RUN_DIR}/vio.tum" \
      -va -a --delta 10 --delta_unit m --t_max_diff 0.05 \
      | tee "${METRIC_DIR}/rpe_vio.txt" || true

    if [[ -s "${RUN_DIR}/vio_loop.tum" ]]; then
      evo_ape tum "${GT_TUM}" "${RUN_DIR}/vio_loop.tum" \
        -va -a --t_max_diff 0.05 \
        | tee "${METRIC_DIR}/ape_loop.txt" || true

      evo_rpe tum "${GT_TUM}" "${RUN_DIR}/vio_loop.tum" \
        -va -a --delta 10 --delta_unit m --t_max_diff 0.05 \
        | tee "${METRIC_DIR}/rpe_loop.txt" || true
    fi
  else
    echo "[WARN] evo_ape not found, skip evo metrics"
  fi

  # =========================================================
  # ML Pipeline
  # 注意：ML 失敗不會讓 VINS 主流程失敗
  # =========================================================
  ML_STATUS="SKIPPED"

  if [[ -d "${PIPELINE_DIR}" && -s "${FEATURE_CSV}" ]]; then
    cd "${PIPELINE_DIR}"

    set +e

    /home/ivlab3/miniconda3/envs/gf/bin/python build_vins_reliability_labels.py \
      --feature_csv "${FEATURE_CSV}" \
      --gt_path "${GT_TUM}" \
      --out_dir "${LABEL_DIR}" \
      --auto_shift_to_first_feature

    LABEL_STATUS=$?

    if [[ "${LABEL_STATUS}" -eq 0 && -s "${LABEL_DIR}/reliability_labels_vins.csv" ]]; then
      /home/ivlab3/miniconda3/envs/gf/bin/python build_vins_reliability_dataset.py \
        --feature_csv "${FEATURE_CSV}" \
        --label_csv "${LABEL_DIR}/reliability_labels_vins.csv" \
        --out_dir "${DATASET_DIR}" \
        --seq_len 8 \
        --split_mode block_class_aware \
        --block_size 512 \
        --seed 42

      DATASET_STATUS=$?
    else
      DATASET_STATUS=1
    fi

    set -e

    if [[ "${LABEL_STATUS}" -eq 0 && "${DATASET_STATUS}" -eq 0 ]]; then
      ML_STATUS="OK"
      echo "[ML OK] reliability dataset 建立成功"
    else
      ML_STATUS="FAIL"
      echo "[ML WARN] reliability dataset 建立失敗，但 VINS 主流程已完成"
      echo "[ML WARN] LABEL_STATUS=${LABEL_STATUS}, DATASET_STATUS=${DATASET_STATUS}"
    fi
  else
    echo "[ML WARN] Skip ML pipeline: PIPELINE_DIR missing or FEATURE_CSV empty"
  fi

  echo "[SUCCESS] ${SEQ_NAME} VINS 主流程完成，ML_STATUS=${ML_STATUS}"
  echo "[SUCCESS] CONFIG_MODE         = ${CONFIG_MODE}"
  echo "[SUCCESS] PLAYER_IMU_SOURCE   = ${PLAYER_IMU_SOURCE}"
  echo "[SUCCESS] VINS_IMU_NOISE_MODE = ${VINS_IMU_NOISE_MODE}"
  exit 0
else
  echo "[ERROR] vio.csv not found or empty: ${RUN_DIR}/vio.csv"
  tail -n 80 "${LOG_DIR}/vins.log" || true
  exit 1
fi