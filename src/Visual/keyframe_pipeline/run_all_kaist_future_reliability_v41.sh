#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_all_kaist_future_reliability_v41.sh
#
# v4.1 目標：
#   1. 不改三個 Python 主檔
#   2. 加入 sequence eligibility filter
#      - 太短 / label 太少 / duration 太短的 sequence 不進 combined dataset
#   3. 先用比較健康的 sequence 做 combined dataset + train
#   4. 輸出到 _combined_future_reliability_v41，不覆蓋 v4
#
# 放置位置：
#   ~/vins_ws/src/Visual/keyframe_pipeline/run_all_kaist_future_reliability_v41.sh
#
# 執行：
#   cd ~/vins_ws/src/Visual/keyframe_pipeline
#   chmod +x run_all_kaist_future_reliability_v41.sh
#   ./run_all_kaist_future_reliability_v41.sh
# ==============================================================================

PIPE_DIR="$HOME/vins_ws/src/Visual/keyframe_pipeline"

RESULT_ROOT="/mnt/sata4t/ivlab3_data/vins_project/results/kaist/fog"
GT_ROOT="/mnt/sata4t/datasets/kaist_complex_urban/extracted"

COMBINED_DIR="${RESULT_ROOT}/_combined_future_reliability_v41"

SEQ_LEN=16
HORIZON_STEPS=50

# v4 label scale
TRANS_THRESHOLD_M=2.0
ROT_THRESHOLD_DEG=10.0
TRANS_SCALE_M=2.0
ROT_SCALE_DEG=10.0
RISK_MODE="log_exp"

MAX_GT_TIME_GAP=0.05
ALLOW_PROXY=0

# v4.1 新增：sequence eligibility filter
# 小於這些條件的 sequence 大多是初始化就失敗 / 沒跑起來，不適合訓練 reliability
MIN_FEATURE_ROWS=300
MIN_LABELED_ROWS=200
MIN_DURATION_SEC=20.0

# 訓練設定
MODEL_TYPE="gru"
EPOCHS=40
BATCH_SIZE=64
HIDDEN_DIM=64
LR=3e-4
WEIGHT_DECAY=1e-3
DROPOUT=0.30
PATIENCE=6

# loss 權重：先以 regression / soft weighting 為主
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
echo "MIN_FEATURE_ROWS = ${MIN_FEATURE_ROWS}"
echo "MIN_LABELED_ROWS = ${MIN_LABELED_ROWS}"
echo "MIN_DURATION_SEC = ${MIN_DURATION_SEC}"

echo
echo "========== [1] Clean combined v4.1 output only =========="
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
    LABEL_DIR="${ADM}/labels_v41"

    if [ ! -f "${FEATURE_CSV}" ]; then
        echo "[SKIP label] ${seq}: no feature csv: ${FEATURE_CSV}"
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
            --max_gt_time_gap "${MAX_GT_TIME_GAP}"
    else
        if [ "${ALLOW_PROXY}" -eq 1 ]; then
            echo "[WARN] ${seq}: 找不到 GT，改用 proxy_future，只建議 debug"
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
                --risk_mode "${RISK_MODE}"
        else
            echo "[SKIP label] ${seq}: 找不到 GT，且 ALLOW_PROXY=0"
            continue
        fi
    fi
done

echo
echo "========== [3] Combine with sequence eligibility filter =========="

export RESULT_ROOT
export COMBINED_DIR
export MIN_FEATURE_ROWS
export MIN_LABELED_ROWS
export MIN_DURATION_SEC

python3 - <<'PY'
import csv
import json
import os
import math
from pathlib import Path

RESULT_ROOT = Path(os.environ["RESULT_ROOT"])
COMBINED_DIR = Path(os.environ["COMBINED_DIR"])
MIN_FEATURE_ROWS = int(os.environ.get("MIN_FEATURE_ROWS", "300"))
MIN_LABELED_ROWS = int(os.environ.get("MIN_LABELED_ROWS", "200"))
MIN_DURATION_SEC = float(os.environ.get("MIN_DURATION_SEC", "20.0"))

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

def duration_from_rows(rows):
    ts = []
    for r in rows:
        t = to_float(r.get("timestamp", math.nan), math.nan)
        if math.isfinite(t):
            ts.append(t)
    if len(ts) < 2:
        return 0.0
    return float(max(ts) - min(ts))

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

def class_counts(rows):
    out = {"harmful": 0, "neutral": 0, "helpful": 0, "unknown": 0}
    for r in rows:
        name = str(r.get("class_name", "")).strip().lower()
        if name in out:
            out[name] += 1
        else:
            cid = to_int(r.get("label_cls", -1), -1)
            if cid == 0:
                out["harmful"] += 1
            elif cid == 1:
                out["neutral"] += 1
            elif cid == 2:
                out["helpful"] += 1
            else:
                out["unknown"] += 1
    return out

def fail_counts(rows):
    valid = 0
    fail = 0
    for r in rows:
        y = to_int(r.get("y_fail", -1), -1)
        if y in [0, 1]:
            valid += 1
            fail += y
    return {"valid": valid, "fail": fail, "fail_rate": (fail / valid if valid > 0 else None)}

all_features = []
all_labels = []
summary = []
skipped = []
global_offset = 0

seq_dirs = sorted([p for p in RESULT_ROOT.iterdir() if p.is_dir() and p.name.startswith("urban")])

for seq_dir in seq_dirs:
    seq = seq_dir.name
    adm = seq_dir / "admission"
    feature_csv = adm / "features" / "reliability_features_vins.csv"
    label_csv = adm / "labels_v41" / "reliability_labels.csv"

    if not feature_csv.exists() or not label_csv.exists():
        skipped.append({
            "sequence": seq,
            "reason": "missing_feature_or_label",
            "feature_csv_exists": feature_csv.exists(),
            "label_csv_exists": label_csv.exists(),
        })
        print(f"[SKIP combine] {seq}: missing feature or label")
        continue

    feature_rows_raw = read_csv(feature_csv)
    label_rows_raw = read_csv(label_csv)
    duration_sec = duration_from_rows(feature_rows_raw)

    reason = None
    if len(feature_rows_raw) < MIN_FEATURE_ROWS:
        reason = f"too_few_feature_rows:{len(feature_rows_raw)}<{MIN_FEATURE_ROWS}"
    elif len(label_rows_raw) < MIN_LABELED_ROWS:
        reason = f"too_few_labeled_rows:{len(label_rows_raw)}<{MIN_LABELED_ROWS}"
    elif duration_sec < MIN_DURATION_SEC:
        reason = f"duration_too_short:{duration_sec:.3f}<{MIN_DURATION_SEC}"

    if reason is not None:
        skipped.append({
            "sequence": seq,
            "reason": reason,
            "feature_rows": len(feature_rows_raw),
            "label_rows": len(label_rows_raw),
            "duration_sec": duration_sec,
            "class_counts": class_counts(label_rows_raw),
            "fail_counts": fail_counts(label_rows_raw),
        })
        print(f"[SKIP combine] {seq}: {reason}")
        continue

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
        "used": True,
        "feature_rows": len(feature_rows_out),
        "label_rows": len(label_rows_out),
        "skipped_label_rows": skipped_label,
        "duration_sec": duration_sec,
        "class_counts": class_counts(label_rows_out),
        "fail_counts": fail_counts(label_rows_out),
        "global_pair_start": global_offset,
        "global_pair_end": global_offset + len(feature_rows_out) - 1,
    })

    print(f"[USE] {seq}: feature={len(feature_rows_out)} label={len(label_rows_out)} duration={duration_sec:.2f}s class={class_counts(label_rows_out)} fail={fail_counts(label_rows_out)}")
    global_offset += len(feature_rows_out)

if not all_features:
    raise RuntimeError("No combined features generated.")
if not all_labels:
    raise RuntimeError("No combined labels generated.")

feature_out = COMBINED_DIR / "features_all.csv"
label_out = COMBINED_DIR / "labels_all.csv"
summary_out = COMBINED_DIR / "combine_summary.json"
skipped_out = COMBINED_DIR / "skipped_sequences.json"

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
        "future_drift_risk", "future_proxy_risk",
    ],
)

with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

with open(skipped_out, "w", encoding="utf-8") as f:
    json.dump(skipped, f, ensure_ascii=False, indent=2)

print("combined feature:", feature_out)
print("combined label  :", label_out)
print("summary         :", summary_out)
print("skipped         :", skipped_out)
print("num features    :", len(all_features))
print("num labels      :", len(all_labels))
print("num used seq    :", len(summary))
print("num skipped seq :", len(skipped))
PY

echo
echo "========== [4] Build combined v4.1 dataset =========="
python3 "${PIPE_DIR}/build_reliability_dataset.py" \
    --sequence_dir "${COMBINED_DIR}" \
    --feature_csv "${COMBINED_DIR}/features_all.csv" \
    --label_csv "${COMBINED_DIR}/labels_all.csv" \
    --out_dir "${COMBINED_DIR}/dataset" \
    --seq_len "${SEQ_LEN}" \
    --split_mode sequence_stratified

echo
echo "========== [5] Train v4.1 combined model =========="
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
echo "========== [6] Debug v4.1 model if debug script exists =========="
if [ -f "${PIPE_DIR}/debug_reliability_model.py" ]; then
    python3 "${PIPE_DIR}/debug_reliability_model.py" \
        --dataset_dir "${COMBINED_DIR}/dataset" \
        --model_dir "${COMBINED_DIR}/model_${MODEL_TYPE}" \
        --out_dir "${COMBINED_DIR}/debug_model" \
        --splits train val test
else
    echo "[WARN] debug_reliability_model.py not found, skip debug."
fi

echo
echo "========== Done v4.1 =========="
echo "Combined dataset: ${COMBINED_DIR}/dataset"
echo "Model output    : ${COMBINED_DIR}/model_${MODEL_TYPE}"
echo "Debug output    : ${COMBINED_DIR}/debug_model"
echo
echo "Check:"
echo "  cat ${COMBINED_DIR}/skipped_sequences.json"
echo "  cat ${COMBINED_DIR}/combine_summary.json"
echo "  cat ${COMBINED_DIR}/dataset/dataset_meta.json"
echo "  cat ${COMBINED_DIR}/model_${MODEL_TYPE}/best_info.json"
echo "  cat ${COMBINED_DIR}/debug_model/debug_summary.json"
