#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# KAIST FOG future-drift reliability pipeline v4
#
# v4 改動：
#   1. labels_v4：不覆蓋舊 labels，方便比較。
#   2. label：使用較不極端的 log_exp continuous drift label。
#   3. threshold/scale：比 v3 放寬，避免所有 sequence 幾乎全 fail。
#   4. dataset：使用 sequence_stratified split，避免 val/test 單一類別。
#   5. train：降低 fail/cls loss 權重，先讓 reg_head 學穩。
# ============================================================

PIPE_DIR="$HOME/vins_ws/src/Visual/keyframe_pipeline"

RESULT_ROOT="/mnt/sata4t/ivlab3_data/vins_project/results/kaist/fog"
GT_ROOT="/mnt/sata4t/datasets/kaist_complex_urban/extracted"

COMBINED_DIR="${RESULT_ROOT}/_combined_future_reliability_v4"

SEQ_LEN=16
HORIZON_STEPS=50

# v3 使用 0.50m / 5deg 太容易讓資料全 fail；
# v4 先放寬，讓 fail / non-fail 都有樣本。
TRANS_THRESHOLD_M=2.0
ROT_THRESHOLD_DEG=10.0

# 用於 continuous reliability label = exp(-log-scaled drift severity)
TRANS_SCALE_M=2.0
ROT_SCALE_DEG=10.0
RISK_MODE="log_exp"
ROT_WEIGHT=0.35

HELPFUL_THR=0.65
HARMFUL_THR=0.35

MAX_GT_TIME_GAP=0.05
ALLOW_PROXY=0

SPLIT_MODE="sequence_stratified"
SPLIT_SEED=42

MODEL_TYPE="gru"
EPOCHS=40
BATCH_SIZE=64
HIDDEN_DIM=64
LR=3e-4
WEIGHT_DECAY=1e-3
DROPOUT=0.30
PATIENCE=6

LAMBDA_REG=1.0
LAMBDA_FAIL=0.2
LAMBDA_CLS=0.1

echo "========== [0] Check scripts =========="
test -f "${PIPE_DIR}/build_reliability_labels.py"
test -f "${PIPE_DIR}/build_reliability_dataset.py"
test -f "${PIPE_DIR}/train_reliability_model.py"

echo "PIPE_DIR     = ${PIPE_DIR}"
echo "RESULT_ROOT  = ${RESULT_ROOT}"
echo "GT_ROOT      = ${GT_ROOT}"
echo "COMBINED_DIR = ${COMBINED_DIR}"

echo
echo "========== [1] Clean combined v4 output only =========="
rm -rf "${COMBINED_DIR}"
mkdir -p "${COMBINED_DIR}"

mapfile -t SEQS < <(
    find "${RESULT_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
    | grep '^urban' \
    | sort
)

if [ "${#SEQS[@]}" -eq 0 ]; then
    echo "[ERROR] 找不到任何 urban sequence in ${RESULT_ROOT}"
    exit 1
fi

echo "[INFO] Found sequences:"
printf '  %s\n' "${SEQS[@]}"

echo
echo "========== [2] Build v4 labels for each sequence =========="

for seq in "${SEQS[@]}"; do
    ADM="${RESULT_ROOT}/${seq}/admission"
    FEATURE_CSV="${ADM}/features/reliability_features_vins.csv"
    LABEL_DIR="${ADM}/labels_v4"

    if [ ! -f "${FEATURE_CSV}" ]; then
        echo "[SKIP] ${seq}: no feature csv: ${FEATURE_CSV}"
        continue
    fi

    GT_CANDIDATES=(
        "${GT_ROOT}/${seq}/pose/${seq}/global_pose.tum"
        "${GT_ROOT}/${seq}/pose/global_pose.tum"
        "${GT_ROOT}/${seq}/pose/${seq}/global_pose.csv"
        "${GT_ROOT}/${seq}/pose/global_pose.csv"
        "${GT_ROOT}/${seq}/pose/${seq}/global_pose.txt"
        "${GT_ROOT}/${seq}/pose/global_pose.txt"
    )

    GT_CSV=""
    for p in "${GT_CANDIDATES[@]}"; do
        if [ -f "${p}" ]; then
            GT_CSV="${p}"
            break
        fi
    done

    echo
    echo "------------------------------------------------------------"
    echo "[SEQ] ${seq}"
    echo "[FEATURE] ${FEATURE_CSV}"

    rm -rf "${LABEL_DIR}"
    mkdir -p "${LABEL_DIR}"

    if [ -n "${GT_CSV}" ]; then
        echo "[GT] ${GT_CSV}"
        python3 "${PIPE_DIR}/build_reliability_labels.py" \
            --sequence_dir "${ADM}" \
            --feature_csv "${FEATURE_CSV}" \
            --gt_csv "${GT_CSV}" \
            --out_dir "${LABEL_DIR}" \
            --label_mode gt_future \
            --horizon_steps "${HORIZON_STEPS}" \
            --trans_threshold_m "${TRANS_THRESHOLD_M}" \
            --rot_threshold_deg "${ROT_THRESHOLD_DEG}" \
            --trans_scale_m "${TRANS_SCALE_M}" \
            --rot_scale_deg "${ROT_SCALE_DEG}" \
            --risk_mode "${RISK_MODE}" \
            --rot_weight "${ROT_WEIGHT}" \
            --helpful_thr "${HELPFUL_THR}" \
            --harmful_thr "${HARMFUL_THR}" \
            --max_gt_time_gap "${MAX_GT_TIME_GAP}"
    else
        if [ "${ALLOW_PROXY}" -eq 1 ]; then
            echo "[WARN] ${seq}: 找不到 GT，改用 proxy_future，只建議 debug，不建議論文主結果"
            python3 "${PIPE_DIR}/build_reliability_labels.py" \
                --sequence_dir "${ADM}" \
                --feature_csv "${FEATURE_CSV}" \
                --out_dir "${LABEL_DIR}" \
                --label_mode proxy_future \
                --horizon_steps "${HORIZON_STEPS}" \
                --trans_threshold_m "${TRANS_THRESHOLD_M}" \
                --rot_threshold_deg "${ROT_THRESHOLD_DEG}" \
                --trans_scale_m "${TRANS_SCALE_M}" \
                --rot_scale_deg "${ROT_SCALE_DEG}" \
                --risk_mode "${RISK_MODE}" \
                --rot_weight "${ROT_WEIGHT}" \
                --helpful_thr "${HELPFUL_THR}" \
                --harmful_thr "${HARMFUL_THR}"
        else
            echo "[SKIP] ${seq}: 找不到 GT，且 ALLOW_PROXY=0"
            continue
        fi
    fi
done

echo
echo "========== [3] Combine all v4 feature/label csv =========="

export RESULT_ROOT
export COMBINED_DIR

python3 - <<'PY'
import csv
import json
import os
from pathlib import Path
import math

RESULT_ROOT = Path(os.environ["RESULT_ROOT"])
COMBINED_DIR = Path(os.environ["COMBINED_DIR"])
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

def to_int(v, default=-1):
    try:
        if v is None or str(v).strip() == "":
            return default
        return int(float(v))
    except Exception:
        return default

def to_float(v, default=math.nan):
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def sort_key(item):
    idx, row = item
    ts = to_float(row.get("timestamp", math.nan), math.nan)
    upd = to_int(row.get("update_id", idx), idx)
    if not math.isfinite(ts):
        ts = float(idx)
    return (ts, upd, idx)

def prepare_feature_rows(rows):
    indexed = list(enumerate(rows))
    indexed.sort(key=sort_key)
    out = []
    for sorted_idx, (orig_idx, row) in enumerate(indexed):
        local_pair_id = to_int(row.get("pair_id", -1), -1)
        if local_pair_id < 0:
            local_pair_id = sorted_idx

        r = dict(row)
        r["_source_row_index"] = str(orig_idx)
        r["_sorted_row_index"] = str(sorted_idx)
        r["_local_pair_id"] = str(local_pair_id)
        out.append((local_pair_id, r))
    return out

def write_csv(path, rows, priority_fields):
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")

    field_set = set()
    for r in rows:
        field_set.update(r.keys())

    fields = []
    for k in priority_fields:
        if k in field_set:
            fields.append(k)
            field_set.remove(k)
    fields += sorted(field_set)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

all_features = []
all_labels = []
summary = []
global_offset = 0

seq_dirs = sorted([p for p in RESULT_ROOT.iterdir() if p.is_dir() and p.name.startswith("urban")])

for seq_dir in seq_dirs:
    seq = seq_dir.name
    adm = seq_dir / "admission"
    feature_csv = adm / "features" / "reliability_features_vins.csv"
    label_csv = adm / "labels_v4" / "reliability_labels.csv"

    if not feature_csv.exists() or not label_csv.exists():
        print(f"[SKIP combine] {seq}: missing feature or labels_v4")
        continue

    feature_rows_raw = read_csv(feature_csv)
    label_rows_raw = read_csv(label_csv)

    prepared = prepare_feature_rows(feature_rows_raw)

    local_to_global = {}
    feature_rows_out = []

    for local_pair_id, row in prepared:
        global_pair_id = global_offset + len(feature_rows_out)
        local_to_global[local_pair_id] = global_pair_id

        r = dict(row)
        r["pair_id"] = str(global_pair_id)
        r["dataset_name"] = "KAIST"
        r["sequence_name"] = seq
        r["run_id"] = f"fog_{seq}"
        r["source_sequence"] = seq
        feature_rows_out.append(r)

    label_rows_out = []
    skipped_label = 0

    for lr0 in label_rows_raw:
        old_pair = to_int(lr0.get("pair_id", -1), -1)
        old_future = to_int(lr0.get("future_pair_id", -1), -1)

        if old_pair not in local_to_global:
            skipped_label += 1
            continue
        if old_future >= 0 and old_future not in local_to_global:
            skipped_label += 1
            continue

        lr = dict(lr0)
        lr["pair_id"] = str(local_to_global[old_pair])
        if old_future >= 0:
            lr["future_pair_id"] = str(local_to_global[old_future])

        lr["dataset_name"] = "KAIST"
        lr["sequence_name"] = seq
        lr["run_id"] = f"fog_{seq}"
        lr["source_sequence"] = seq
        label_rows_out.append(lr)

    all_features.extend(feature_rows_out)
    all_labels.extend(label_rows_out)

    summary.append({
        "sequence": seq,
        "feature_rows": len(feature_rows_out),
        "label_rows": len(label_rows_out),
        "skipped_label_rows": skipped_label,
        "global_pair_start": global_offset,
        "global_pair_end": global_offset + len(feature_rows_out) - 1,
    })

    global_offset += len(feature_rows_out)

if not all_features:
    raise RuntimeError("No combined features generated.")
if not all_labels:
    raise RuntimeError("No combined labels generated.")

feature_out = COMBINED_DIR / "features_all.csv"
label_out = COMBINED_DIR / "labels_all.csv"
summary_out = COMBINED_DIR / "combine_summary.json"

write_csv(
    feature_out,
    all_features,
    [
        "pair_id", "dataset_name", "sequence_name", "run_id",
        "timestamp", "update_id", "source_sequence",
    ],
)

write_csv(
    label_out,
    all_labels,
    [
        "pair_id", "future_pair_id",
        "dataset_name", "sequence_name", "run_id",
        "timestamp", "future_timestamp",
        "horizon_steps", "label_reg", "label_cls", "class_name", "y_fail",
        "label_source", "future_drift_trans_m", "future_drift_rot_deg",
        "future_drift_trans_norm", "future_drift_rot_norm",
        "future_drift_severity", "future_drift_risk", "future_proxy_risk",
    ],
)

with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("combined feature:", feature_out)
print("combined label  :", label_out)
print("summary         :", summary_out)
print("num features    :", len(all_features))
print("num labels      :", len(all_labels))
print("num sequences   :", len(summary))
PY

echo
echo "========== [4] Build combined v4 dataset =========="

python3 "${PIPE_DIR}/build_reliability_dataset.py" \
    --sequence_dir "${COMBINED_DIR}" \
    --feature_csv "${COMBINED_DIR}/features_all.csv" \
    --label_csv "${COMBINED_DIR}/labels_all.csv" \
    --out_dir "${COMBINED_DIR}/dataset" \
    --seq_len "${SEQ_LEN}" \
    --split_mode "${SPLIT_MODE}" \
    --split_seed "${SPLIT_SEED}"

echo
echo "========== [5] Train v4 combined model =========="

python3 "${PIPE_DIR}/train_reliability_model.py" \
    --dataset_dir "${COMBINED_DIR}/dataset" \
    --out_dir "${COMBINED_DIR}/model_${MODEL_TYPE}" \
    --model_type "${MODEL_TYPE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --dropout "${DROPOUT}" \
    --patience "${PATIENCE}" \
    --lambda_reg "${LAMBDA_REG}" \
    --lambda_fail "${LAMBDA_FAIL}" \
    --lambda_cls "${LAMBDA_CLS}"

echo
echo "========== Done v4 =========="
echo "Combined dataset: ${COMBINED_DIR}/dataset"
echo "Model output    : ${COMBINED_DIR}/model_${MODEL_TYPE}"
echo
echo "Check:"
echo "  cat ${COMBINED_DIR}/dataset/dataset_meta.json"
echo "  cat ${COMBINED_DIR}/model_${MODEL_TYPE}/best_info.json"
echo "  cat ${COMBINED_DIR}/model_${MODEL_TYPE}/final_summary.json"
