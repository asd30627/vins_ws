#!/usr/bin/env bash
# ==============================================================================
# KAIST 批次執行腳本 - FOG / IMU 雙模式，自動同步 player config + VINS imu_noise_mode
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_one.sh"

FOG_TEMPLATE="/home/ivlab3/vins_ws/src/kaist_player/config/urban28_pankyo_fog.yaml"
IMU_TEMPLATE="/home/ivlab3/vins_ws/src/kaist_player/config/urban28_pankyo.yaml"

# 不建議把 sudo 密碼直接寫在腳本裡
# 如果你真的要自動 drop cache，可以這樣執行：
# SUDO_PASS='你的密碼' bash batch_run_kaist.sh
SUDO_PASS="${SUDO_PASS:-}"

MODES=("fog" "imu")

SEQ_NAMES=(
  "urban18-highway"
  "urban19-highway" "urban20-highway" "urban21-highway"
  "urban22-highway" "urban23-highway" "urban24-highway"
  "urban25-highway" "urban26-dongtan" "urban27-dongtan"
  "urban28-pankyo"  "urban29-pankyo"  "urban30-gangnam"
  "urban31-gangnam" "urban32-yeouido" "urban33-yeouido"
  "urban34-yeouido" "urban35-seoul"   "urban36-seoul"
  "urban37-seoul"   "urban38-pankyo"  "urban39-pankyo"
)

BASE_DIR="/mnt/sata4t/datasets/kaist_complex_urban/extracted"
LOG_DIR="${SCRIPT_DIR}/batch_logs"
mkdir -p "${LOG_DIR}"

get_player_template() {
  local mode="$1"

  case "$mode" in
    fog)
      echo "${FOG_TEMPLATE}"
      ;;
    imu)
      echo "${IMU_TEMPLATE}"
      ;;
    *)
      echo "[ERROR] Unknown mode: ${mode}" >&2
      return 1
      ;;
  esac
}

get_vins_noise_mode() {
  local mode="$1"

  case "$mode" in
    fog)
      echo "fog_xsens"
      ;;
    imu)
      echo "xsens"
      ;;
    *)
      echo "[ERROR] Unknown mode: ${mode}" >&2
      return 1
      ;;
  esac
}

force_cleanup() {
  echo "[SYS] 執行深度大掃除..."

  pkill -9 -f "vins" || true
  pkill -9 -f "loop_fusion" || true
  pkill -9 -f "kaist_player" || true
  pkill -9 -f "build_vins_reliability" || true

  # 這個會殺掉所有 ros2 command，請確認沒有其他 ROS2 實驗同時在跑
  pkill -9 -f "ros2" || true

  sleep 5

  echo "[SYS] 嘗試釋放系統快取..."
  if sudo -n true 2>/dev/null; then
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
  elif [[ -n "${SUDO_PASS}" ]]; then
    echo "${SUDO_PASS}" | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
  else
    echo "[WARN] 沒有 sudo 權限或 SUDO_PASS，略過 drop_caches"
  fi
}

for MODE in "${MODES[@]}"; do
  CURRENT_TEMPLATE="$(get_player_template "${MODE}")" || exit 1
  VINS_IMU_NOISE_MODE="$(get_vins_noise_mode "${MODE}")" || exit 1

  SUMMARY_LOG="${LOG_DIR}/summary_${MODE}_${VINS_IMU_NOISE_MODE}_$(date +%Y%m%d_%H%M%S).log"

  {
    echo "============================================================"
    echo "[BATCH START]"
    echo "MODE                : ${MODE}"
    echo "PLAYER_TEMPLATE     : ${CURRENT_TEMPLATE}"
    echo "VINS_IMU_NOISE_MODE : ${VINS_IMU_NOISE_MODE}"
    echo "============================================================"
  } | tee -a "${SUMMARY_LOG}"

  for SEQ_NAME in "${SEQ_NAMES[@]}"; do
    force_cleanup

    FREE_MEM="$(free -g | awk '/^Mem:/{print $4}')"
    FREE_MEM="${FREE_MEM:-0}"

    if [[ "${FREE_MEM}" -lt 10 ]]; then
      echo "[WARN] 記憶體僅剩 ${FREE_MEM}GB，等待 30 秒..." | tee -a "${SUMMARY_LOG}"
      sleep 30
    fi

    SEQ_ROOT="${BASE_DIR}/${SEQ_NAME}"
    GT_TUM="${SEQ_ROOT}/pose/${SEQ_NAME}/global_pose.tum"

    {
      echo ""
      echo ">>> [START] ${SEQ_NAME}"
      echo "    MODE                : ${MODE}"
      echo "    VINS_IMU_NOISE_MODE : ${VINS_IMU_NOISE_MODE}"
      echo "    SEQ_ROOT            : ${SEQ_ROOT}"
      echo "    GT_TUM              : ${GT_TUM}"
      echo "<<<"
    } | tee -a "${SUMMARY_LOG}"

    if bash "${RUN_ONE}" \
      "${SEQ_NAME}" \
      "${SEQ_ROOT}" \
      "${CURRENT_TEMPLATE}" \
      "${GT_TUM}" \
      "${MODE}" \
      "${VINS_IMU_NOISE_MODE}" 2>&1 | tee -a "${SUMMARY_LOG}"; then

      echo "[OK] ${SEQ_NAME} 執行成功 (${MODE}, ${VINS_IMU_NOISE_MODE})" | tee -a "${SUMMARY_LOG}"
    else
      echo "[FAIL] ${SEQ_NAME} 執行失敗 (${MODE}, ${VINS_IMU_NOISE_MODE})" | tee -a "${SUMMARY_LOG}"
    fi

    force_cleanup
    sleep 2
  done
done
