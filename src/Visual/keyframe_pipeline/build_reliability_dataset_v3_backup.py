#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_reliability_dataset.py

目的：
  將 VINS-Fusion reliability feature csv + future drift label csv 合併，建立 train/val/test npz。

這版修正重點：
  1. feature 不再使用 target-like columns，例如 visual_w_pred / visual_soft_target_proxy / failure_detected_last。
  2. 不使用絕對 pose est_p_* / est_q_* 當模型輸入，避免模型記住路徑位置造成洩漏。
  3. 先依時間切 split，再各 split 內建立 sequence。
  4. 使用 label_csv 的 future_pair_id / horizon_steps 避免 train sample 的 label 偷看到 val/test 時段。
  5. fill value / mean / std 只從 train split 估計。
  6. 可輸出 y_reg、y_cls、y_fail、future drift debug targets。
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


CLASS_NAME_TO_ID = {
    "harmful": 0,
    "neutral": 1,
    "helpful": 2,
}

CLASS_ID_TO_NAME = {
    0: "harmful",
    1: "neutral",
    2: "helpful",
}


# 這些欄位是可線上取得、且不直接是舊模型輸出或未來 label 的候選 feature。
# 若 CSV 缺少某些欄位，builder 會自動忽略。
DEFAULT_BASE_FEATURE_NAMES = [
    "current_is_keyframe",
    "use_imu",
    "stereo",
    "estimate_extrinsic",
    "estimate_td",
    "td_current",

    "feature_tracker_time_ms",
    "tracked_feature_count_raw",
    "tracked_feature_count_mgr",
    "mean_track_vel_px",
    "median_track_vel_px",
    "min_track_vel_px",
    "max_track_vel_px",
    "std_track_vel_px",
    "p90_track_vel_px",

    "coverage_4x4",
    "coverage_8x8",
    "occupied_cells_4x4",
    "occupied_cells_8x8",
    "feature_entropy_4x4",
    "feature_entropy_8x8",
    "img_dt_sec",

    "solver_time_ms_last",
    "outlier_count_last",
    "inlier_count_last",
    "outlier_ratio_last",

    "vel_norm",
    "ba_norm",
    "bg_norm",
    "delta_p_norm",
    "delta_q_deg",

    "imu_sample_count",
    "acc_norm_mean",
    "acc_norm_std",
    "acc_norm_max",
    "gyr_norm_mean",
    "gyr_norm_std",
    "gyr_norm_max",

    "avg_track_length",
    "track_len_min",
    "track_len_max",
    "track_len_std",
    "track_len_p90",

    "good_depth_count",
    "bad_depth_count",
    "depth_mean",
    "depth_min",
    "depth_max",
    "depth_std",
]


# 不可當 feature 的欄位。這裡刻意保守，避免資料洩漏。
LEAKAGE_OR_ID_COLUMNS = {
    "schema_version",
    "pair_id",
    "source_row_index",
    "future_pair_id",
    "future_source_row_index",
    "run_id",
    "dataset_name",
    "sequence_name",
    "update_id",
    "future_update_id",
    "frame_count",
    "timestamp",
    "future_timestamp",
    "horizon_steps",
    "horizon_seconds",
    "solver_flag",

    # label / target
    "label_reg",
    "y_reg",
    "label_cls",
    "y_cls",
    "class_name",
    "y_fail",
    "label_source",
    "future_drift_trans_m",
    "future_drift_rot_deg",
    "future_drift_risk",
    "future_proxy_risk",

    # 舊模型輸出或會直接暗示 target 的欄位
    "visual_w_pred",
    "visual_gate_pass",
    "visual_alpha",
    "visual_has_prediction",
    "visual_soft_target_proxy",
    "failure_detected_last",
    "failure_reason_proxy",

    # absolute pose / calibration identity，容易讓模型記 sequence/location
    "est_p_x",
    "est_p_y",
    "est_p_z",
    "est_q_x",
    "est_q_y",
    "est_q_z",
    "est_q_w",
    "cam0_tic_x",
    "cam0_tic_y",
    "cam0_tic_z",
    "cam0_q_x",
    "cam0_q_y",
    "cam0_q_z",
    "cam0_q_w",
    "cam0_init_delta_t_norm",
    "cam0_init_delta_r_deg",
    "cam1_tic_x",
    "cam1_tic_y",
    "cam1_tic_z",
    "cam1_q_x",
    "cam1_q_y",
    "cam1_q_z",
    "cam1_q_w",
    "cam1_init_delta_t_norm",
    "cam1_init_delta_r_deg",
}


NONNEGATIVE_FEATURES = set(DEFAULT_BASE_FEATURE_NAMES)


# =========================================================
# utils
# =========================================================

def to_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value, default=-1) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def to_str(value, default="") -> str:
    if value is None:
        return default
    return str(value)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prepare_feature_rows(feature_csv: Path) -> List[Dict[str, str]]:
    rows = read_csv_rows(feature_csv)

    def sort_key(item):
        idx, row = item
        run_id = to_str(row.get("run_id", ""))
        dataset_name = to_str(row.get("dataset_name", ""))
        sequence_name = to_str(row.get("sequence_name", ""))
        ts = to_float(row.get("timestamp", np.nan), np.nan)
        upd = to_int(row.get("update_id", idx), idx)
        if not np.isfinite(ts):
            ts = float(idx)
        return (dataset_name, sequence_name, run_id, ts, upd, idx)

    indexed = list(enumerate(rows))
    indexed.sort(key=sort_key)

    out = []
    for global_idx, (orig_idx, row) in enumerate(indexed):
        r = dict(row)
        existing_pair_id = to_int(r.get("pair_id", -1), -1)
        if existing_pair_id < 0:
            r["pair_id"] = str(global_idx)
        else:
            r["pair_id"] = str(existing_pair_id)
        r["_source_row_index"] = str(orig_idx)
        r["_sorted_row_index"] = str(global_idx)
        out.append(r)
    return out


def numeric_columns_from_rows(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    keys = list(rows[0].keys())
    numeric = []
    sample = rows[: min(len(rows), 200)]
    for k in keys:
        if k in LEAKAGE_OR_ID_COLUMNS or k.startswith("_"):
            continue
        valid_count = 0
        for row in sample:
            v = to_float(row.get(k, np.nan), np.nan)
            if np.isfinite(v):
                valid_count += 1
        if valid_count >= max(3, int(0.5 * len(sample))):
            numeric.append(k)
    return numeric


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =========================================================
# builder
# =========================================================

class ReliabilityDatasetBuilder:
    def __init__(
        self,
        sequence_dir,
        feature_csv="",
        label_csv="",
        out_dir="",
        seq_len=16,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        min_rows=64,
        purge_gap=None,
        use_validity_mask=True,
        base_feature_names=None,
        auto_features=False,
        split_mode="chronological",
        require_label_cls=False,
        debug_csv_name="feature_label_debug.csv",
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()

        if feature_csv:
            self.feature_csv = Path(feature_csv).expanduser().resolve()
        else:
            candidates = [
                self.sequence_dir / "reliability_features_vins.csv",
                self.sequence_dir / "features" / "reliability_features_vins.csv",
                self.sequence_dir / "features" / "all_candidate_features.csv",
            ]
            self.feature_csv = next((p for p in candidates if p.exists()), candidates[0])

        self.label_csv = (
            Path(label_csv).expanduser().resolve()
            if label_csv
            else self.sequence_dir / "reliability_labels" / "reliability_labels.csv"
        )
        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.sequence_dir / "reliability_dataset"

        self.seq_len = int(seq_len)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.min_rows = int(min_rows)
        self.purge_gap = None if purge_gap is None else int(purge_gap)

        self.use_validity_mask = bool(use_validity_mask)
        self.require_label_cls = bool(require_label_cls)
        self.debug_csv_name = str(debug_csv_name)
        self.auto_features = bool(auto_features)
        self.split_mode = str(split_mode).strip().lower()

        if base_feature_names is None:
            self.base_feature_names = list(DEFAULT_BASE_FEATURE_NAMES)
        else:
            self.base_feature_names = list(base_feature_names)

    def _check_inputs(self):
        if not self.feature_csv.exists():
            raise FileNotFoundError(f"找不到 feature_csv: {self.feature_csv}")
        if not self.label_csv.exists():
            raise FileNotFoundError(f"找不到 label_csv: {self.label_csv}")

        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"train/val/test ratio 總和必須為 1.0，目前為 {total_ratio}")

        if self.seq_len < 1:
            raise ValueError("seq_len 必須 >= 1")

        if self.split_mode not in ["chronological", "sequence"]:
            raise ValueError("split_mode 只支援 chronological / sequence")

        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_valid_feature_value(name: str, value: float) -> bool:
        if not np.isfinite(value):
            return False
        if name in NONNEGATIVE_FEATURES and value < 0.0:
            return False
        return True

    @staticmethod
    def _median_ignore_invalid(arr: np.ndarray, feature_name: str, default_value: float = 0.0) -> float:
        valid_mask = np.isfinite(arr)
        if feature_name in NONNEGATIVE_FEATURES:
            valid_mask = valid_mask & (arr >= 0.0)
        valid = arr[valid_mask]
        if len(valid) == 0:
            return float(default_value)
        return float(np.median(valid))

    @staticmethod
    def _compute_mean_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(X, axis=0).astype(np.float32)
        std = np.std(X, axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std

    def _parse_label_row(self, row: Dict[str, str]) -> Optional[Dict[str, object]]:
        pair_id = to_int(row.get("pair_id", -1), -1)
        if pair_id < 0:
            return None

        label_reg = to_float(row.get("label_reg", np.nan), np.nan)
        if not np.isfinite(label_reg):
            label_reg = to_float(row.get("y_reg", np.nan), np.nan)

        label_cls = to_int(row.get("label_cls", -1), -1)
        if label_cls not in CLASS_ID_TO_NAME:
            name = to_str(row.get("class_name", "")).strip().lower()
            if name in CLASS_NAME_TO_ID:
                label_cls = CLASS_NAME_TO_ID[name]

        y_fail = to_int(row.get("y_fail", -1), -1)

        out = {
            "pair_id": int(pair_id),
            "future_pair_id": to_int(row.get("future_pair_id", -1), -1),
            "label_reg": float(label_reg) if np.isfinite(label_reg) else np.nan,
            "label_cls": int(label_cls) if label_cls in CLASS_ID_TO_NAME else -1,
            "class_name": CLASS_ID_TO_NAME.get(label_cls, ""),
            "y_fail": int(y_fail) if y_fail in [0, 1] else -1,
            "future_drift_trans_m": to_float(row.get("future_drift_trans_m", np.nan), np.nan),
            "future_drift_rot_deg": to_float(row.get("future_drift_rot_deg", np.nan), np.nan),
            "future_drift_risk": to_float(row.get("future_drift_risk", np.nan), np.nan),
            "future_proxy_risk": to_float(row.get("future_proxy_risk", np.nan), np.nan),
            "horizon_steps": to_int(row.get("horizon_steps", -1), -1),
            "label_source": to_str(row.get("label_source", "")),
        }
        return out

    def _load_label_map(self) -> Dict[int, Dict[str, object]]:
        rows = read_csv_rows(self.label_csv)
        label_map = {}

        for row in rows:
            parsed = self._parse_label_row(row)
            if parsed is None:
                continue
            label_map[int(parsed["pair_id"])] = parsed

        return label_map

    def _select_feature_names(self, feature_rows: List[Dict[str, str]]) -> List[str]:
        available = set(feature_rows[0].keys()) if feature_rows else set()

        if self.auto_features:
            names = numeric_columns_from_rows(feature_rows)
        else:
            names = [n for n in self.base_feature_names if n in available]

        names = [n for n in names if n not in LEAKAGE_OR_ID_COLUMNS and not n.startswith("_")]

        if len(names) == 0:
            raise ValueError("沒有可用 feature 欄位。請檢查 CSV 欄位或使用 --auto_features。")
        return names

    def _merge_feature_and_label_rows(
        self,
        feature_rows: List[Dict[str, str]],
        label_map: Dict[int, Dict[str, object]],
        feature_names: List[str],
    ) -> List[Dict[str, object]]:
        merged = []

        for row in feature_rows:
            pair_id = to_int(row.get("pair_id", -1), -1)
            if pair_id < 0:
                continue
            if pair_id not in label_map:
                continue

            label_info = label_map[pair_id]
            label_reg = float(label_info["label_reg"])
            label_cls = int(label_info["label_cls"])

            if not np.isfinite(label_reg):
                continue
            if self.require_label_cls and label_cls < 0:
                continue

            merged_row = {
                "pair_id": int(pair_id),
                "future_pair_id": int(label_info.get("future_pair_id", -1)),
                "source_row_index": to_int(row.get("_source_row_index", -1), -1),
                "sorted_row_index": to_int(row.get("_sorted_row_index", pair_id), pair_id),

                "dataset_name": to_str(row.get("dataset_name", "")),
                "sequence_name": to_str(row.get("sequence_name", "")),
                "run_id": to_str(row.get("run_id", "")),
                "update_id": to_int(row.get("update_id", pair_id), pair_id),
                "timestamp": to_float(row.get("timestamp", np.nan), np.nan),

                "label_reg": float(label_reg),
                "label_cls": int(label_cls),
                "class_name": to_str(label_info.get("class_name", "")),
                "y_fail": int(label_info.get("y_fail", -1)),
                "future_drift_trans_m": float(label_info.get("future_drift_trans_m", np.nan)),
                "future_drift_rot_deg": float(label_info.get("future_drift_rot_deg", np.nan)),
                "future_drift_risk": float(label_info.get("future_drift_risk", np.nan)),
                "future_proxy_risk": float(label_info.get("future_proxy_risk", np.nan)),
                "horizon_steps": int(label_info.get("horizon_steps", -1)),
                "label_source": to_str(label_info.get("label_source", "")),
            }

            for feat_name in feature_names:
                merged_row[feat_name] = to_float(row.get(feat_name, np.nan), np.nan)

            merged.append(merged_row)

        merged.sort(key=lambda x: (x["dataset_name"], x["sequence_name"], x["run_id"], x["timestamp"], x["update_id"], x["pair_id"]))
        return merged

    def _infer_purge_gap(self, rows: List[Dict[str, object]]) -> int:
        if self.purge_gap is not None:
            return int(self.purge_gap)

        hs = [int(r.get("horizon_steps", -1)) for r in rows]
        hs = [h for h in hs if h >= 0]
        max_horizon = max(hs) if hs else 0

        # seq_len-1 防止 sequence overlap，max_horizon 防止 label future 跨 split
        return int(max(self.seq_len - 1, 0) + max_horizon)

    def _filter_rows_future_inside_split(
        self,
        rows: List[Dict[str, object]],
        split_pair_set: set,
    ) -> List[Dict[str, object]]:
        out = []
        for r in rows:
            future_pair_id = int(r.get("future_pair_id", -1))
            if future_pair_id < 0:
                # 如果舊 label 沒有 future_pair_id，就只能依靠 purge_gap
                out.append(r)
            elif future_pair_id in split_pair_set:
                out.append(r)
        return out

    def _split_raw_rows_chronological(
        self,
        rows: List[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], int]:
        n = len(rows)
        if n < self.min_rows:
            raise ValueError(f"可用 labeled rows 太少，目前 {n} 筆，小於 min_rows={self.min_rows}")

        purge_gap = self._infer_purge_gap(rows)

        train_end = int(round(n * self.train_ratio))
        val_end = int(round(n * (self.train_ratio + self.val_ratio)))

        train_end = max(train_end, self.seq_len)
        val_end = max(val_end, train_end + self.seq_len)
        val_end = min(val_end, n)

        val_start = min(n, train_end + purge_gap)
        test_start = min(n, val_end + purge_gap)

        train_rows = rows[:train_end]
        val_rows = rows[val_start:val_end]
        test_rows = rows[test_start:]

        train_set = {int(r["pair_id"]) for r in train_rows}
        val_set = {int(r["pair_id"]) for r in val_rows}
        test_set = {int(r["pair_id"]) for r in test_rows}

        train_rows = self._filter_rows_future_inside_split(train_rows, train_set)
        val_rows = self._filter_rows_future_inside_split(val_rows, val_set)
        test_rows = self._filter_rows_future_inside_split(test_rows, test_set)

        self._validate_split_lengths(train_rows, val_rows, test_rows)
        return train_rows, val_rows, test_rows, purge_gap

    def _split_raw_rows_by_sequence(
        self,
        rows: List[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], int]:
        """
        如果之後你把多個 KAIST sequence 合在一起，建議用這個模式。
        它會以完整 sequence/run 為單位切 split，最能避免 sequence leakage。
        """
        groups = {}
        for r in rows:
            key = (r["dataset_name"], r["sequence_name"], r["run_id"])
            groups.setdefault(key, []).append(r)

        keys = sorted(groups.keys())
        if len(keys) < 3:
            raise ValueError("sequence split 至少需要 3 個 sequence/run；目前不足，請改用 chronological。")

        n_group = len(keys)
        n_train = max(1, int(round(n_group * self.train_ratio)))
        n_val = max(1, int(round(n_group * self.val_ratio)))
        if n_train + n_val >= n_group:
            n_train = max(1, n_group - 2)
            n_val = 1

        train_keys = set(keys[:n_train])
        val_keys = set(keys[n_train:n_train + n_val])
        test_keys = set(keys[n_train + n_val:])

        train_rows = [r for r in rows if (r["dataset_name"], r["sequence_name"], r["run_id"]) in train_keys]
        val_rows = [r for r in rows if (r["dataset_name"], r["sequence_name"], r["run_id"]) in val_keys]
        test_rows = [r for r in rows if (r["dataset_name"], r["sequence_name"], r["run_id"]) in test_keys]

        purge_gap = self._infer_purge_gap(rows)
        self._validate_split_lengths(train_rows, val_rows, test_rows)
        return train_rows, val_rows, test_rows, purge_gap

    def _validate_split_lengths(self, train_rows, val_rows, test_rows):
        if len(train_rows) < self.seq_len:
            raise ValueError(f"train split 太短，len={len(train_rows)}，seq_len={self.seq_len}")
        if len(val_rows) < self.seq_len:
            raise ValueError(f"val split 太短，len={len(val_rows)}，seq_len={self.seq_len}")
        if len(test_rows) < self.seq_len:
            raise ValueError(f"test split 太短，len={len(test_rows)}，seq_len={self.seq_len}")

    def _build_train_fill_values(self, train_rows: List[Dict[str, object]], feature_names: List[str]) -> Dict[str, float]:
        fill_values = {}

        for feat_name in feature_names:
            arr = np.array([to_float(r.get(feat_name, np.nan), np.nan) for r in train_rows], dtype=np.float64)
            fill_values[feat_name] = self._median_ignore_invalid(arr, feat_name, default_value=0.0)

        return fill_values

    def _feature_names_with_mask(self, feature_names: List[str]) -> List[str]:
        if not self.use_validity_mask:
            return list(feature_names)

        out = []
        for name in feature_names:
            out.append(name)
            out.append(f"valid_{name}")
        return out

    def _rows_to_feature_matrix(
        self,
        rows: List[Dict[str, object]],
        feature_names: List[str],
        fill_values: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        X_cols = []
        pair_ids = []
        y_reg = []
        y_cls = []
        y_fail = []
        y_drift_trans = []
        y_drift_rot = []
        y_drift_risk = []
        y_proxy_risk = []

        for row in rows:
            pair_ids.append(int(row["pair_id"]))
            y_reg.append(float(row["label_reg"]))
            y_cls.append(int(row["label_cls"]))
            y_fail.append(int(row.get("y_fail", -1)))
            y_drift_trans.append(float(row.get("future_drift_trans_m", np.nan)))
            y_drift_rot.append(float(row.get("future_drift_rot_deg", np.nan)))
            y_drift_risk.append(float(row.get("future_drift_risk", np.nan)))
            y_proxy_risk.append(float(row.get("future_proxy_risk", np.nan)))

        for feat_name in feature_names:
            raw_vals = np.array([to_float(r.get(feat_name, np.nan), np.nan) for r in rows], dtype=np.float64)

            valid_mask = np.array(
                [1.0 if self._is_valid_feature_value(feat_name, v) else 0.0 for v in raw_vals],
                dtype=np.float32,
            )

            filled_vals = raw_vals.copy()
            fill_v = float(fill_values.get(feat_name, 0.0))
            invalid = ~np.isfinite(filled_vals)
            if feat_name in NONNEGATIVE_FEATURES:
                invalid = invalid | (filled_vals < 0.0)
            filled_vals[invalid] = fill_v

            X_cols.append(filled_vals.reshape(-1, 1).astype(np.float32))
            if self.use_validity_mask:
                X_cols.append(valid_mask.reshape(-1, 1).astype(np.float32))

        X = np.concatenate(X_cols, axis=1).astype(np.float32)

        return {
            "X": X,
            "pair_ids": np.array(pair_ids, dtype=np.int32),
            "y_reg": np.array(y_reg, dtype=np.float32),
            "y_cls": np.array(y_cls, dtype=np.int64),
            "y_fail": np.array(y_fail, dtype=np.int64),
            "y_drift_trans": np.array(y_drift_trans, dtype=np.float32),
            "y_drift_rot": np.array(y_drift_rot, dtype=np.float32),
            "y_drift_risk": np.array(y_drift_risk, dtype=np.float32),
            "y_proxy_risk": np.array(y_proxy_risk, dtype=np.float32),
        }

    def _standardize_train_val_test(self, train_raw: np.ndarray, val_raw: np.ndarray, test_raw: np.ndarray):
        mean, std = self._compute_mean_std(train_raw)
        train = ((train_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
        val = ((val_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
        test = ((test_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
        return train, val, test, mean, std

    def _build_sequences_in_split(self, split_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X = split_data["X"]
        n = len(X)

        xs = []
        end_pair_ids = []
        y_reg = []
        y_cls = []
        y_fail = []
        y_drift_trans = []
        y_drift_rot = []
        y_drift_risk = []
        y_proxy_risk = []

        for end_idx in range(self.seq_len - 1, n):
            start_idx = end_idx - self.seq_len + 1
            xs.append(X[start_idx:end_idx + 1])
            end_pair_ids.append(split_data["pair_ids"][end_idx])
            y_reg.append(split_data["y_reg"][end_idx])
            y_cls.append(split_data["y_cls"][end_idx])
            y_fail.append(split_data["y_fail"][end_idx])
            y_drift_trans.append(split_data["y_drift_trans"][end_idx])
            y_drift_rot.append(split_data["y_drift_rot"][end_idx])
            y_drift_risk.append(split_data["y_drift_risk"][end_idx])
            y_proxy_risk.append(split_data["y_proxy_risk"][end_idx])

        return {
            "X": np.stack(xs).astype(np.float32),
            "pair_ids": np.array(end_pair_ids, dtype=np.int32),
            "y": np.array(y_reg, dtype=np.float32),  # backward compatible
            "y_reg": np.array(y_reg, dtype=np.float32),
            "y_cls": np.array(y_cls, dtype=np.int64),
            "y_fail": np.array(y_fail, dtype=np.int64),
            "y_drift_trans": np.array(y_drift_trans, dtype=np.float32),
            "y_drift_rot": np.array(y_drift_rot, dtype=np.float32),
            "y_drift_risk": np.array(y_drift_risk, dtype=np.float32),
            "y_proxy_risk": np.array(y_proxy_risk, dtype=np.float32),
        }

    def _write_npz(self, path: Path, seq_data: Dict[str, np.ndarray], feature_names: List[str], mean: np.ndarray, std: np.ndarray):
        np.savez_compressed(
            path,
            X=seq_data["X"],
            y=seq_data["y"],
            y_reg=seq_data["y_reg"],
            y_cls=seq_data["y_cls"],
            y_fail=seq_data["y_fail"],
            y_drift_trans=seq_data["y_drift_trans"],
            y_drift_rot=seq_data["y_drift_rot"],
            y_drift_risk=seq_data["y_drift_risk"],
            y_proxy_risk=seq_data["y_proxy_risk"],
            pair_ids=seq_data["pair_ids"],
            feature_names=np.array(feature_names, dtype=object),
            seq_len=np.array([self.seq_len], dtype=np.int32),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )

    def _write_debug_csv(self, train_rows, val_rows, test_rows, feature_names):
        debug_path = self.out_dir / self.debug_csv_name
        fieldnames = [
            "split",
            "pair_id",
            "future_pair_id",
            "dataset_name",
            "sequence_name",
            "run_id",
            "update_id",
            "timestamp",
            "label_reg",
            "label_cls",
            "class_name",
            "y_fail",
            "future_drift_trans_m",
            "future_drift_rot_deg",
            "future_drift_risk",
            "future_proxy_risk",
            "label_source",
        ] + list(feature_names)

        rows_out = []
        for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            for row in rows:
                out = {"split": split_name}
                for k in fieldnames:
                    if k == "split":
                        continue
                    out[k] = row.get(k, "")
                rows_out.append(out)

        write_csv_rows(debug_path, rows_out, fieldnames)

    def build(self):
        self._check_inputs()

        feature_rows = prepare_feature_rows(self.feature_csv)
        feature_names = self._select_feature_names(feature_rows)

        label_map = self._load_label_map()
        merged_rows = self._merge_feature_and_label_rows(feature_rows, label_map, feature_names)

        if len(merged_rows) < self.min_rows:
            raise ValueError(
                f"merged 後可用 rows 太少，只有 {len(merged_rows)} 筆；"
                f"請確認 label_csv 是否與 feature_csv 的 pair_id 對得上。"
            )

        if self.split_mode == "sequence":
            train_rows, val_rows, test_rows, purge_gap_used = self._split_raw_rows_by_sequence(merged_rows)
        else:
            train_rows, val_rows, test_rows, purge_gap_used = self._split_raw_rows_chronological(merged_rows)

        fill_values = self._build_train_fill_values(train_rows, feature_names)

        train_raw = self._rows_to_feature_matrix(train_rows, feature_names, fill_values)
        val_raw = self._rows_to_feature_matrix(val_rows, feature_names, fill_values)
        test_raw = self._rows_to_feature_matrix(test_rows, feature_names, fill_values)

        train_X_std, val_X_std, test_X_std, mean, std = self._standardize_train_val_test(
            train_raw["X"], val_raw["X"], test_raw["X"]
        )

        train_raw["X"] = train_X_std
        val_raw["X"] = val_X_std
        test_raw["X"] = test_X_std

        train_seq = self._build_sequences_in_split(train_raw)
        val_seq = self._build_sequences_in_split(val_raw)
        test_seq = self._build_sequences_in_split(test_raw)

        final_feature_names = self._feature_names_with_mask(feature_names)

        self._write_npz(self.out_dir / "train.npz", train_seq, final_feature_names, mean, std)
        self._write_npz(self.out_dir / "val.npz", val_seq, final_feature_names, mean, std)
        self._write_npz(self.out_dir / "test.npz", test_seq, final_feature_names, mean, std)

        self._write_debug_csv(train_rows, val_rows, test_rows, feature_names)

        meta = {
            "feature_csv": str(self.feature_csv),
            "label_csv": str(self.label_csv),
            "out_dir": str(self.out_dir),
            "seq_len": int(self.seq_len),
            "split_mode": self.split_mode,
            "train_ratio": float(self.train_ratio),
            "val_ratio": float(self.val_ratio),
            "test_ratio": float(self.test_ratio),
            "purge_gap_used": int(purge_gap_used),
            "use_validity_mask": bool(self.use_validity_mask),
            "auto_features": bool(self.auto_features),
            "base_feature_names": list(feature_names),
            "feature_names": list(final_feature_names),
            "input_dim": int(len(final_feature_names)),
            "num_merged_rows": int(len(merged_rows)),
            "num_train_rows": int(len(train_rows)),
            "num_val_rows": int(len(val_rows)),
            "num_test_rows": int(len(test_rows)),
            "num_train_seq": int(len(train_seq["X"])),
            "num_val_seq": int(len(val_seq["X"])),
            "num_test_seq": int(len(test_seq["X"])),
            "class_counts_train": {
                CLASS_ID_TO_NAME.get(int(c), str(c)): int(np.sum(train_seq["y_cls"] == int(c)))
                for c in [-1, 0, 1, 2]
            },
            "fail_counts_train": {
                "valid_fail_labels": int(np.sum((train_seq["y_fail"] == 0) | (train_seq["y_fail"] == 1))),
                "fail": int(np.sum(train_seq["y_fail"] == 1)),
            },
            "leakage_columns_excluded": sorted(list(LEAKAGE_OR_ID_COLUMNS)),
            "note": "mean/std/fill 只使用 train split；sequence 只在各 split 內建立。",
        }

        meta_path = self.out_dir / "dataset_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("========== Build Reliability Dataset Finished ==========")
        print(f"train npz: {self.out_dir / 'train.npz'}")
        print(f"val npz:   {self.out_dir / 'val.npz'}")
        print(f"test npz:  {self.out_dir / 'test.npz'}")
        print(f"debug csv: {self.out_dir / self.debug_csv_name}")
        print(f"meta json: {meta_path}")
        print(f"input_dim: {len(final_feature_names)}")
        print(f"purge_gap_used: {purge_gap_used}")
        print(f"num train/val/test seq: {len(train_seq['X'])} / {len(val_seq['X'])} / {len(test_seq['X'])}")

        return meta


# =========================================================
# CLI
# =========================================================

def parse_feature_name_list(s: str) -> Optional[List[str]]:
    if not s:
        return None
    p = Path(s).expanduser()
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "feature_names" in data:
            return list(data["feature_names"])
        if isinstance(data, list):
            return list(data)
        raise ValueError("feature_names_json 必須是 list 或包含 feature_names 的 dict")
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--feature_csv", type=str, default="")
    parser.add_argument("--label_csv", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--min_rows", type=int, default=64)
    parser.add_argument("--purge_gap", type=int, default=-1, help="-1 表示自動使用 seq_len + horizon_steps。")

    parser.add_argument("--use_validity_mask", action="store_true", default=True)
    parser.add_argument("--no_validity_mask", action="store_false", dest="use_validity_mask")

    parser.add_argument("--auto_features", action="store_true", help="自動使用 numeric columns，但仍會排除 leakage columns。")
    parser.add_argument("--feature_names", type=str, default="", help="逗號分隔欄位，或 JSON 檔路徑。")
    parser.add_argument("--split_mode", type=str, default="chronological", choices=["chronological", "sequence"])
    parser.add_argument("--require_label_cls", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    feature_names = parse_feature_name_list(args.feature_names)
    purge_gap = None if int(args.purge_gap) < 0 else int(args.purge_gap)

    builder = ReliabilityDatasetBuilder(
        sequence_dir=args.sequence_dir,
        feature_csv=args.feature_csv,
        label_csv=args.label_csv,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_rows=args.min_rows,
        purge_gap=purge_gap,
        use_validity_mask=args.use_validity_mask,
        base_feature_names=feature_names,
        auto_features=args.auto_features,
        split_mode=args.split_mode,
        require_label_cls=args.require_label_cls,
    )
    builder.build()


if __name__ == "__main__":
    main()
