#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# KAIST 28~39 APE/RPE batch evaluator
# ------------------------------------------------------------
# 會自動對每條 sequence 跑：
#   1) vio.tum
#   2) vio_loop.tum
# 的 APE / RPE
#
# 輸出：
#   ~/vins_project/eval/kaist_batch_eval_<timestamp>/
#     ├── logs/
#     │    └── <seq>_<mode>_{ape,rpe}.txt
#     └── summary.csv
#
# 注意：
# - GT 預設從 /mnt/sata4t/datasets/kaist_complex_urban/extracted 讀
# - Result 預設從 ~/vins_project/results/kaist/<seq>/runs/baseline_current 讀
# ============================================================

DATASET_ROOT="/mnt/sata4t/datasets/kaist_complex_urban/extracted"
RESULT_ROOT="${HOME}/vins_project/results/kaist"
OUT_ROOT="${HOME}/vins_project/eval/kaist_batch_eval_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUT_ROOT}/logs"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

mkdir -p "${LOG_DIR}"

# 你要評估的 sequence
SEQ_PAIRS=(
  "urban28_pankyo|urban28-pankyo"
  "urban29_pankyo|urban29-pankyo"
  "urban30_gangnam|urban30-gangnam"
  "urban31_gangnam|urban31-gangnam"
  "urban32_yeouido|urban32-yeouido"
  "urban33_yeouido|urban33-yeouido"
  "urban34_yeouido|urban34-yeouido"
  "urban35_seoul|urban35-seoul"
  "urban36_seoul|urban36-seoul"
  "urban37_seoul|urban37-seoul"
  "urban38_pankyo|urban38-pankyo"
  "urban39_pankyo|urban39-pankyo"
)

# 檢查 evo 是否存在
for cmd in evo_ape evo_rpe; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] 找不到 $cmd，請先安裝 evo"
    exit 1
  fi
done

echo "sequence,mode,status,gt_path,est_path,ape_rmse,ape_mean,ape_median,rpe_rmse,rpe_mean,rpe_median" > "${SUMMARY_CSV}"

extract_stat() {
  # 用法: extract_stat <logfile> <key>
  # key 例如: rmse / mean / median
  local logfile="$1"
  local key="$2"
  awk -v k="$key" '$1==k {print $2; exit}' "$logfile"
}

run_eval_one() {
  local seq_name="$1"      # e.g. urban28_pankyo
  local seq_base="$2"      # e.g. urban28-pankyo
  local mode="$3"          # vio or vio_loop
  local est_path="$4"
  local gt_path="$5"

  local ape_log="${LOG_DIR}/${seq_name}_${mode}_ape.txt"
  local rpe_log="${LOG_DIR}/${seq_name}_${mode}_rpe.txt"

  local status="OK"
  local ape_rmse="NA"
  local ape_mean="NA"
  local ape_median="NA"
  local rpe_rmse="NA"
  local rpe_mean="NA"
  local rpe_median="NA"

  if [[ ! -f "${gt_path}" ]]; then
    status="MISSING_GT"
    echo "${seq_name},${mode},${status},${gt_path},${est_path},${ape_rmse},${ape_mean},${ape_median},${rpe_rmse},${rpe_mean},${rpe_median}" >> "${SUMMARY_CSV}"
    return
  fi

  if [[ ! -f "${est_path}" ]]; then
    status="MISSING_EST"
    echo "${seq_name},${mode},${status},${gt_path},${est_path},${ape_rmse},${ape_mean},${ape_median},${rpe_rmse},${rpe_mean},${rpe_median}" >> "${SUMMARY_CSV}"
    return
  fi

  echo "------------------------------------------------------------"
  echo "[RUN] ${seq_name} | ${mode}"
  echo "[GT ] ${gt_path}"
  echo "[EST] ${est_path}"
  echo "------------------------------------------------------------"

  # APE
  if evo_ape tum "${gt_path}" "${est_path}" -va -a > "${ape_log}" 2>&1; then
    ape_rmse="$(extract_stat "${ape_log}" rmse)"
    ape_mean="$(extract_stat "${ape_log}" mean)"
    ape_median="$(extract_stat "${ape_log}" median)"
  else
    status="APE_FAIL"
  fi

  # RPE
  if evo_rpe tum "${gt_path}" "${est_path}" -va -a --delta 10 --delta_unit m > "${rpe_log}" 2>&1; then
    rpe_rmse="$(extract_stat "${rpe_log}" rmse)"
    rpe_mean="$(extract_stat "${rpe_log}" mean)"
    rpe_median="$(extract_stat "${rpe_log}" median)"
  else
    if [[ "${status}" == "OK" ]]; then
      status="RPE_FAIL"
    else
      status="${status}+RPE_FAIL"
    fi
  fi

  echo "${seq_name},${mode},${status},${gt_path},${est_path},${ape_rmse},${ape_mean},${ape_median},${rpe_rmse},${rpe_mean},${rpe_median}" >> "${SUMMARY_CSV}"
}

for item in "${SEQ_PAIRS[@]}"; do
  IFS='|' read -r seq_name seq_base <<< "${item}"

  gt_path="${DATASET_ROOT}/${seq_base}/pose/${seq_base}/global_pose.tum"
  run_dir="${RESULT_ROOT}/${seq_name}/runs/baseline_current"

  # no-loop
  run_eval_one "${seq_name}" "${seq_base}" "vio" "${run_dir}/vio.tum" "${gt_path}"

  # loop
  run_eval_one "${seq_name}" "${seq_base}" "vio_loop" "${run_dir}/vio_loop.tum" "${gt_path}"
done

echo
echo "[DONE] summary -> ${SUMMARY_CSV}"
echo "[DONE] logs    -> ${LOG_DIR}"
