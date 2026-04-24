#!/usr/bin/env bash
# ==============================================================================
# VINS-Fusion 自動化評估腳本 - Experiment 1
# 目的：
#   1. 保留原本 loop_fusion / vins / kaist_player 啟動與關閉流程
#   2. 保留 fog / imu 結果資料夾隔離
#   3. 不修改 imu_noise_mode
#   4. 額外保存實際使用的 VINS config / player config 方便事後比對
# ==============================================================================

set -eo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <SEQ_NAME> <SEQ_ROOT> <PLAYER_CONFIG_TEMPLATE> <GT_TUM> <CONFIG_MODE>"
  echo "Example:"
  echo "$0 urban28_pankyo /mnt/sata4t/datasets/kaist_complex_urban/extracted/urban28-pankyo /home/ivlab3/vins_ws/src/kaist_player/config/urban28_pankyo_fog.yaml /mnt/sata4t/datasets/kaist_complex_urban/extracted/urban28-pankyo/pose/urban28-pankyo/global_pose.tum fog"
  exit 1
fi

SEQ_NAME="$1"
SEQ_ROOT="$2"
PLAYER_CONFIG_TEMPLATE="$3"
GT_TUM="$4"
CONFIG_MODE="$5"
SEQ_BASENAME="$(basename "$SEQ_ROOT")"

# =========================================================
# 【救援機制】自動轉檔 GT TUM
# =========================================================
GT_POSE_DIR="$(dirname "$GT_TUM")"
FALLBACK_GT_TUM_SCRIPT="/mnt/sata4t/datasets/kaist_complex_urban/extracted/urban28-pankyo/pose/urban28-pankyo/global_pose_to_tum.py"

if [[ ! -f "$GT_TUM" ]]; then
  echo "[WARN] 找不到 GT_TUM: $GT_TUM"
  echo "[INFO] 嘗試從 global_pose.csv 自動轉檔..."

  if [[ -f "$GT_POSE_DIR/global_pose.csv" ]]; then
    [[ ! -f "$GT_POSE_DIR/global_pose_to_tum.py" ]] && cp "$FALLBACK_GT_TUM_SCRIPT" "$GT_POSE_DIR/global_pose_to_tum.py"
    ( cd "$GT_POSE_DIR" && python3 global_pose_to_tum.py )
  fi
fi

if [[ ! -f "$GT_TUM" ]]; then
  echo "[ERROR] 仍然找不到 GT_TUM: $GT_TUM"
  exit 1
fi

# =========================================================
# 絕對路徑設定 - 依模式隔離資料
# =========================================================
SATA_BASE="/mnt/sata4t/ivlab3_data/vins_project"
VINS_WS="${HOME}/vins_ws"
PIPELINE_DIR="${HOME}/Lucas_ws/Visual/keyframe_pipeline"
VIO_TO_TUM="${SATA_BASE}/tools/vio_csv_to_tum.py"
VINS_CONFIG="${VINS_WS}/src/VINS-Fusion-ROS2-jazzy/config/kaist/kaist_stereo_xsens.yaml"

VINS_CONFIG_DIR="$(dirname "${VINS_CONFIG}")"
TMP_VINS_CONFIG="${VINS_CONFIG_DIR}/vins_config_tmp_exp1_${CONFIG_MODE}_${SEQ_NAME}.yaml"

RESULT_ROOT_SAT="${SATA_BASE}/results/kaist/exp1_no_imu_noise_mode/${CONFIG_MODE}/${SEQ_NAME}"
RUN_DIR="${RESULT_ROOT_SAT}/vins_raw"
FEATURE_DIR="${RESULT_ROOT_SAT}/admission/features"
LABEL_DIR="${RESULT_ROOT_SAT}/admission/labels"
DATASET_DIR="${RESULT_ROOT_SAT}/admission/datasets"
LOG_DIR="${RESULT_ROOT_SAT}/logs"
METRIC_DIR="${RESULT_ROOT_SAT}/metrics"

FEATURE_CSV="${FEATURE_DIR}/reliability_features_vins.csv"
LABEL_CSV="${LABEL_DIR}/reliability_labels_vins.csv"
TMP_PLAYER_CONFIG="${LOG_DIR}/kaist_player_config.yaml"

mkdir -p "${RUN_DIR}/pose_graph" "${FEATURE_DIR}" "${LABEL_DIR}" "${DATASET_DIR}" "${LOG_DIR}" "${METRIC_DIR}"

export REL_RUN_ID="exp1_$(date +%Y%m%d_%H%M%S)"
export REL_DATASET_NAME="kaist"
export REL_SEQUENCE_NAME="${SEQ_NAME}"
export REL_FEATURE_CSV_PATH="${FEATURE_CSV}"

rm -rf "${FEATURE_CSV}" "${LABEL_CSV}" "${RUN_DIR}/"*.csv "${RUN_DIR}/"*.tum "${TMP_PLAYER_CONFIG}" "${TMP_VINS_CONFIG}"
rm -f "${LOG_DIR}/"*.log

# =========================================================
# 動態產生 Config
# 注意：
#   Experiment 1 不修改 imu_noise_mode
# =========================================================
python3 - <<PY
from pathlib import Path
import re

def set_yaml_scalar(text, key, value):
    """
    比原本 re.sub(r'key:\\s*\".*\"') 更保守：
    不管原本有沒有雙引號，都會覆蓋該 key。
    但這裡只用來改 output_path / pose_graph_save_path，
    不改 imu_noise_mode。
    """
    pattern = rf'^\\s*{re.escape(key)}\\s*:\\s*.*$'
    new_line = f'{key}: "{value}"'

    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, new_line, text, flags=re.MULTILINE)
    else:
        return text.rstrip() + "\\n" + new_line + "\\n"

p_text = Path(r"${PLAYER_CONFIG_TEMPLATE}").read_text()
p_text = p_text.replace('/mnt/sata4t/datasets/kaist_complex_urban/extracted/urban28-pankyo', r"${SEQ_ROOT}")
p_text = p_text.replace('urban28-pankyo', r"${SEQ_BASENAME}")
Path(r"${TMP_PLAYER_CONFIG}").write_text(p_text)

v_text = Path(r"${VINS_CONFIG}").read_text()

v_text = set_yaml_scalar(v_text, "output_path", f'{r"${RUN_DIR}"}/')
v_text = set_yaml_scalar(v_text, "pose_graph_save_path", f'{r"${RUN_DIR}"}/pose_graph/')

# 這裡刻意不修改 imu_noise_mode
Path(r"${TMP_VINS_CONFIG}").write_text(v_text)

print("[INFO] Experiment 1 config generated")
print(f"[INFO] CONFIG_MODE = {r'${CONFIG_MODE}'}")
print("[INFO] imu_noise_mode is NOT modified by this script")
print(f"[INFO] output_path = {r'${RUN_DIR}'}/")
print(f"[INFO] pose_graph_save_path = {r'${RUN_DIR}'}/pose_graph/")
print(f"[INFO] TMP_VINS_CONFIG = {r'${TMP_VINS_CONFIG}'}")
print(f"[INFO] TMP_PLAYER_CONFIG = {r'${TMP_PLAYER_CONFIG}'}")
PY

# 保存當次實際用到的 config，避免 cleanup 後找不到
cp "${TMP_VINS_CONFIG}" "${LOG_DIR}/vins_config_used.yaml"
cp "${TMP_PLAYER_CONFIG}" "${LOG_DIR}/kaist_player_config_used.yaml"

echo "------------------------------------------------------------"
echo ">>> [EXP1 no imu_noise_mode] [${CONFIG_MODE^^}] 執行序列: ${SEQ_NAME} <<<"
echo "------------------------------------------------------------"

echo "[INFO] 檢查暫存 VINS config 關鍵欄位:"
grep -E "^(output_path|pose_graph_save_path|imu:|imu_noise_mode|image0_topic|image1_topic|imu_topic|loop_closure)" "${TMP_VINS_CONFIG}" || true

echo "[INFO] 檢查暫存 player config 關鍵欄位:"
grep -E "(dataset|root|sequence|imu|fog|xsens|rate|topic|stamp|time)" "${TMP_PLAYER_CONFIG}" || true

cleanup() {
  set +e
  pkill -P $$ 2>/dev/null || true
  rm -f "${TMP_VINS_CONFIG}"
}
trap cleanup EXIT

source /opt/ros/jazzy/setup.bash
source "${VINS_WS}/install/setup.bash"

# =========================================================
# 啟動系統
# 保持接近你原本腳本的流程，不額外改關機順序
# =========================================================
ros2 run loop_fusion loop_fusion_node "${TMP_VINS_CONFIG}" --ros-args -p use_sim_time:=true > "${LOG_DIR}/loop_fusion.log" 2>&1 &
sleep 2

ros2 run vins vins_node "${TMP_VINS_CONFIG}" --ros-args -p use_sim_time:=true > "${LOG_DIR}/vins.log" 2>&1 &
VINS_PID=$!
sleep 5

ros2 launch kaist_player kaist_player.launch.py config_file:="${TMP_PLAYER_CONFIG}" dataset_root:="${SEQ_ROOT}" > "${LOG_DIR}/player.log" 2>&1 &
PLAYER_PID=$!

# =========================================================
# 監控進度
# =========================================================
while true; do
  if grep -q "Playback finished\." "${LOG_DIR}/player.log" 2>/dev/null; then
    break
  fi

  if ! kill -0 $PLAYER_PID 2>/dev/null || ! kill -0 $VINS_PID 2>/dev/null; then
    echo "[ERROR] player 或 VINS 提前結束"
    echo "------ player.log tail ------"
    tail -n 80 "${LOG_DIR}/player.log" || true
    echo "------ vins.log tail ------"
    tail -n 80 "${LOG_DIR}/vins.log" || true
    echo "------ loop_fusion.log tail ------"
    tail -n 80 "${LOG_DIR}/loop_fusion.log" || true
    exit 1
  fi

  sleep 5
done

echo "[INFO] 播放完成，正在儲存資料..."
sleep 20

kill -INT $(jobs -p) 2>/dev/null || true
sleep 15

# =========================================================
# 後處理與評估
# =========================================================
echo "[INFO] 輸出檔案檢查:"
ls -lh "${RUN_DIR}/" || true

if [[ -s "${RUN_DIR}/vio.csv" ]]; then
    python3 "${VIO_TO_TUM}" "${RUN_DIR}/vio.csv" "${RUN_DIR}/vio.tum"

    if [[ -s "${RUN_DIR}/vio_loop.csv" ]]; then
      python3 "${VIO_TO_TUM}" "${RUN_DIR}/vio_loop.csv" "${RUN_DIR}/vio_loop.tum"
    else
      echo "[WARN] 沒有有效的 vio_loop.csv，請檢查 loop_fusion.log"
      tail -n 120 "${LOG_DIR}/loop_fusion.log" || true
    fi

    if command -v evo_ape >/dev/null 2>&1; then
      evo_ape tum "${GT_TUM}" "${RUN_DIR}/vio.tum" -va -a --t_max_diff 0.05 | tee "${METRIC_DIR}/ape_vio.txt" || true
      evo_rpe tum "${GT_TUM}" "${RUN_DIR}/vio.tum" -va -a --delta 10 --delta_unit m --t_max_diff 0.05 | tee "${METRIC_DIR}/rpe_vio.txt" || true

      if [[ -s "${RUN_DIR}/vio_loop.tum" ]]; then
        evo_ape tum "${GT_TUM}" "${RUN_DIR}/vio_loop.tum" -va -a --t_max_diff 0.05 | tee "${METRIC_DIR}/ape_loop.txt" || true
        evo_rpe tum "${GT_TUM}" "${RUN_DIR}/vio_loop.tum" -va -a --delta 10 --delta_unit m --t_max_diff 0.05 | tee "${METRIC_DIR}/rpe_loop.txt" || true
      fi
    fi

    # ML Pipeline
    if [[ -f "${FEATURE_CSV}" ]]; then
      cd "${PIPELINE_DIR}"

      /home/ivlab3/miniconda3/envs/gf/bin/python build_vins_reliability_labels.py \
        --feature_csv "${FEATURE_CSV}" \
        --gt_path "${GT_TUM}" \
        --out_dir "${LABEL_DIR}" \
        --auto_shift_to_first_feature || true

      /home/ivlab3/miniconda3/envs/gf/bin/python build_vins_reliability_dataset.py \
        --feature_csv "${FEATURE_CSV}" \
        --label_csv "${LABEL_DIR}/reliability_labels_vins.csv" \
        --out_dir "${DATASET_DIR}" \
        --seq_len 8 \
        --split_mode block_class_aware \
        --block_size 512 \
        --seed 42 || true
    else
      echo "[WARN] 找不到 FEATURE_CSV: ${FEATURE_CSV}"
    fi

    echo "[SUCCESS] ${SEQ_NAME} (${CONFIG_MODE}) Experiment 1 完成"
else
    echo "[ERROR] 找不到有效 vio.csv: ${RUN_DIR}/vio.csv"
    echo "------ vins.log tail ------"
    tail -n 120 "${LOG_DIR}/vins.log" || true
    echo "------ loop_fusion.log tail ------"
    tail -n 120 "${LOG_DIR}/loop_fusion.log" || true
    exit 1
fi
