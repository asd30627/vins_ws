#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_reliability_labels.py

目的：
  從 VINS-Fusion reliability feature csv 建立「未來漂移風險」label。
  這版刻意刪掉原本 LightGlue / image rematch / weak label 的部分，避免 label 直接由目前 feature 拼出來，
  改成兩種模式：

  1) gt_future:
     用 GT + VINS estimated pose 計算未來 horizon 的 Relative Pose Error (RPE)。
     這是論文建議使用的正式 label。

  2) proxy_future:
     沒有 GT 時的過渡版本，用未來 horizon 內的 VINS 內部不穩定訊號做 proxy label。
     這只能先讓 pipeline 跑起來，不建議當最終論文主結果。

輸出：
  reliability_labels.csv
  reliability_labels_debug.csv
  reliability_labels_summary.json

重要設計：
  - feature input 只應使用當下/過去資訊。
  - label 可以看未來 H 秒，因為這是 supervised training 的 target。
  - build dataset 時會使用 future_pair_id / horizon_steps 做 purge，避免 train label 偷看到 val/test 區間。
"""

import argparse
import csv
import json
import math
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


# =========================================================
# basic utils
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


def clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def safe_exp_good(error_value: float, scale: float) -> float:
    if not np.isfinite(error_value) or error_value < 0.0:
        return 0.0
    return clip01(math.exp(-float(error_value) / max(float(scale), 1e-9)))


def robust_percentile(values, q: float, default_value: float, min_value: float = 1e-9) -> float:
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[np.isfinite(arr) & (arr >= 0.0)]
    if len(valid) == 0:
        return float(default_value)
    return float(max(np.percentile(valid, q), default_value, min_value))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_first_existing_key(row: Dict[str, object], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in row:
            return k
    return None


# =========================================================
# pose math
# =========================================================

def quat_xyzw_to_rotmat(qx, qy, qz, qw) -> Optional[np.ndarray]:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return None
    qx, qy, qz, qw = q / n

    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def rotation_angle_deg(R: np.ndarray) -> float:
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def inverse_compose_relative(p0: np.ndarray, R0: np.ndarray, p1: np.ndarray, R1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    回傳 T_0^{-1} T_1 的 relative translation / rotation。
    translation 轉到 frame-0，可避免 global translation offset。
    若 est/gt 只差一個固定 global frame 旋轉，relative rotation 也可抵消。
    """
    R_rel = R0.T @ R1
    t_rel = R0.T @ (p1 - p0)
    return t_rel, R_rel


def relative_pose_error(
    est0: Dict[str, object],
    est1: Dict[str, object],
    gt0: Dict[str, object],
    gt1: Dict[str, object],
) -> Tuple[float, float]:
    p_e0, R_e0 = est0["p"], est0["R"]
    p_e1, R_e1 = est1["p"], est1["R"]
    p_g0, R_g0 = gt0["p"], gt0["R"]
    p_g1, R_g1 = gt1["p"], gt1["R"]

    t_e_rel, R_e_rel = inverse_compose_relative(p_e0, R_e0, p_e1, R_e1)
    t_g_rel, R_g_rel = inverse_compose_relative(p_g0, R_g0, p_g1, R_g1)

    trans_err = float(np.linalg.norm(t_e_rel - t_g_rel))
    rot_err = rotation_angle_deg(R_g_rel.T @ R_e_rel)
    return trans_err, rot_err


def row_to_est_pose(row: Dict[str, str]) -> Optional[Dict[str, object]]:
    p_keys = ["est_p_x", "est_p_y", "est_p_z"]
    q_keys = ["est_q_x", "est_q_y", "est_q_z", "est_q_w"]

    if not all(k in row for k in p_keys + q_keys):
        return None

    p = np.array([
        to_float(row.get("est_p_x"), np.nan),
        to_float(row.get("est_p_y"), np.nan),
        to_float(row.get("est_p_z"), np.nan),
    ], dtype=np.float64)

    R = quat_xyzw_to_rotmat(
        to_float(row.get("est_q_x"), np.nan),
        to_float(row.get("est_q_y"), np.nan),
        to_float(row.get("est_q_z"), np.nan),
        to_float(row.get("est_q_w"), np.nan),
    )

    if R is None or not np.all(np.isfinite(p)):
        return None

    return {"p": p, "R": R}


# =========================================================
# GT loading / association
# =========================================================

def normalize_header_name(s: str) -> str:
    return str(s).strip().lower().replace("#", "").replace(" ", "_")


def load_gt_poses(gt_csv: Path) -> List[Dict[str, object]]:
    """
    支援兩種格式：
    1) 有 header 的 csv：
       timestamp/time/t, tx/x/pos_x, ty/y/pos_y, tz/z/pos_z,
       qx/quat_x, qy/quat_y, qz/quat_z, qw/quat_w
    2) 無 header 的 TUM：
       timestamp tx ty tz qx qy qz qw
    """
    if not gt_csv.exists():
        raise FileNotFoundError(f"找不到 gt_csv: {gt_csv}")

    text = gt_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
    text = [ln for ln in text if ln.strip() and not ln.strip().startswith("//")]
    if len(text) == 0:
        raise ValueError(f"GT 檔案是空的: {gt_csv}")

    first = text[0].strip()
    has_header = any(ch.isalpha() for ch in first)

    poses = []

    if not has_header:
        # TUM format
        data = np.loadtxt(str(gt_csv), dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 8:
            raise ValueError("無 header GT 至少需要 8 欄: timestamp tx ty tz qx qy qz qw")

        for row in data:
            R = quat_xyzw_to_rotmat(row[4], row[5], row[6], row[7])
            if R is None:
                continue
            poses.append({
                "timestamp": float(row[0]),
                "p": np.asarray(row[1:4], dtype=np.float64),
                "R": R,
            })
    else:
        with open(gt_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_fieldnames = reader.fieldnames or []
            norm_to_raw = {normalize_header_name(k): k for k in raw_fieldnames}

            def pick(candidates):
                for c in candidates:
                    c_norm = normalize_header_name(c)
                    if c_norm in norm_to_raw:
                        return norm_to_raw[c_norm]
                return None

            t_key = pick(["timestamp", "time", "t", "stamp", "timestamp_sec", "time_sec"])
            x_key = pick(["tx", "x", "p_x", "pos_x", "position_x"])
            y_key = pick(["ty", "y", "p_y", "pos_y", "position_y"])
            z_key = pick(["tz", "z", "p_z", "pos_z", "position_z"])
            qx_key = pick(["qx", "q_x", "quat_x", "orientation_x"])
            qy_key = pick(["qy", "q_y", "quat_y", "orientation_y"])
            qz_key = pick(["qz", "q_z", "quat_z", "orientation_z"])
            qw_key = pick(["qw", "q_w", "quat_w", "orientation_w"])

            required = [t_key, x_key, y_key, z_key, qx_key, qy_key, qz_key, qw_key]
            if any(k is None for k in required):
                raise ValueError(
                    "GT CSV 欄位不足。需要 timestamp, position xyz, quaternion xyzw。"
                    f"目前欄位: {raw_fieldnames}"
                )

            for row in reader:
                ts = to_float(row.get(t_key), np.nan)
                p = np.array([
                    to_float(row.get(x_key), np.nan),
                    to_float(row.get(y_key), np.nan),
                    to_float(row.get(z_key), np.nan),
                ], dtype=np.float64)
                R = quat_xyzw_to_rotmat(
                    to_float(row.get(qx_key), np.nan),
                    to_float(row.get(qy_key), np.nan),
                    to_float(row.get(qz_key), np.nan),
                    to_float(row.get(qw_key), np.nan),
                )
                if not np.isfinite(ts) or R is None or not np.all(np.isfinite(p)):
                    continue
                poses.append({"timestamp": float(ts), "p": p, "R": R})

    poses.sort(key=lambda x: x["timestamp"])
    if len(poses) == 0:
        raise ValueError(f"GT 沒有可用 pose: {gt_csv}")
    return poses


def nearest_gt_pose(gt_poses: List[Dict[str, object]], ts: float, max_time_gap: float) -> Optional[Dict[str, object]]:
    if len(gt_poses) == 0 or not np.isfinite(ts):
        return None

    timestamps = np.array([p["timestamp"] for p in gt_poses], dtype=np.float64)
    idx = int(np.searchsorted(timestamps, ts))
    candidates = []
    if 0 <= idx < len(gt_poses):
        candidates.append(idx)
    if 0 <= idx - 1 < len(gt_poses):
        candidates.append(idx - 1)

    if not candidates:
        return None

    best = min(candidates, key=lambda i: abs(float(timestamps[i]) - float(ts)))
    if abs(float(timestamps[best]) - float(ts)) > float(max_time_gap):
        return None
    return gt_poses[best]


# =========================================================
# feature row preparation
# =========================================================

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


def group_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (
            to_str(row.get("dataset_name", "")),
            to_str(row.get("sequence_name", "")),
            to_str(row.get("run_id", "")),
        )
        groups.setdefault(key, []).append(row)
    return groups


def find_future_index_in_group(
    rows: List[Dict[str, str]],
    i: int,
    horizon_steps: int,
    horizon_seconds: float,
    max_time_gap: float,
) -> Optional[int]:
    if horizon_seconds > 0.0 and "timestamp" in rows[i]:
        timestamps = np.array([to_float(r.get("timestamp", np.nan), np.nan) for r in rows], dtype=np.float64)
        ts0 = timestamps[i]
        if not np.isfinite(ts0):
            return None
        target = ts0 + float(horizon_seconds)
        j = int(np.searchsorted(timestamps, target))
        if j >= len(rows):
            return None
        if abs(float(timestamps[j]) - target) > float(max_time_gap):
            return None
        return j

    j = i + int(horizon_steps)
    if j >= len(rows):
        return None
    return j


# =========================================================
# label builder
# =========================================================

class ReliabilityLabelBuilder:
    def __init__(
        self,
        sequence_dir: str,
        feature_csv: str = "",
        gt_csv: str = "",
        out_dir: str = "",
        label_mode: str = "auto",
        horizon_steps: int = 50,
        horizon_seconds: float = 0.0,
        max_time_gap: float = 0.05,
        max_gt_time_gap: float = 0.05,
        trans_threshold_m: float = 0.50,
        rot_threshold_deg: float = 5.0,
        trans_scale_m: float = 0.50,
        rot_scale_deg: float = 5.0,
        helpful_thr: float = 0.70,
        harmful_thr: float = 0.40,
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

        self.gt_csv = Path(gt_csv).expanduser().resolve() if gt_csv else None
        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.sequence_dir / "reliability_labels"

        self.label_mode = str(label_mode).strip().lower()
        self.horizon_steps = int(horizon_steps)
        self.horizon_seconds = float(horizon_seconds)
        self.max_time_gap = float(max_time_gap)
        self.max_gt_time_gap = float(max_gt_time_gap)

        self.trans_threshold_m = float(trans_threshold_m)
        self.rot_threshold_deg = float(rot_threshold_deg)
        self.trans_scale_m = float(trans_scale_m)
        self.rot_scale_deg = float(rot_scale_deg)

        self.helpful_thr = float(helpful_thr)
        self.harmful_thr = float(harmful_thr)

        self.proxy_scales = {}

    def _check_inputs(self):
        if not self.feature_csv.exists():
            raise FileNotFoundError(f"找不到 feature_csv: {self.feature_csv}")
        if self.horizon_steps < 1 and self.horizon_seconds <= 0.0:
            raise ValueError("horizon_steps 必須 >= 1，或設定 horizon_seconds > 0")
        if self.label_mode not in ["auto", "gt_future", "proxy_future"]:
            raise ValueError(f"label_mode 只支援 auto / gt_future / proxy_future，目前: {self.label_mode}")
        if not (0.0 <= self.harmful_thr <= self.helpful_thr <= 1.0):
            raise ValueError("threshold 必須滿足 0 <= harmful_thr <= helpful_thr <= 1")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _fit_proxy_scales(self, rows: List[Dict[str, str]]):
        def col_values(name):
            return [to_float(r.get(name, np.nan), np.nan) for r in rows]

        self.proxy_scales = {
            "delta_p_norm": robust_percentile(col_values("delta_p_norm"), 95, default_value=0.50),
            "delta_q_deg": robust_percentile(col_values("delta_q_deg"), 95, default_value=2.0),
            "outlier_ratio_last": robust_percentile(col_values("outlier_ratio_last"), 95, default_value=0.05),
            "solver_time_ms_last": robust_percentile(col_values("solver_time_ms_last"), 90, default_value=20.0),
            "inlier_count_last": robust_percentile(col_values("inlier_count_last"), 50, default_value=80.0),
        }

    def _score_to_class(self, reliability: float) -> Tuple[int, str]:
        if reliability < self.harmful_thr:
            return CLASS_NAME_TO_ID["harmful"], "harmful"
        if reliability > self.helpful_thr:
            return CLASS_NAME_TO_ID["helpful"], "helpful"
        return CLASS_NAME_TO_ID["neutral"], "neutral"

    def _compute_gt_label(
        self,
        row0: Dict[str, str],
        row1: Dict[str, str],
        gt_poses: List[Dict[str, object]],
    ) -> Tuple[Optional[Dict[str, object]], str]:
        est0 = row_to_est_pose(row0)
        est1 = row_to_est_pose(row1)
        if est0 is None or est1 is None:
            return None, "missing_est_pose"

        ts0 = to_float(row0.get("timestamp", np.nan), np.nan)
        ts1 = to_float(row1.get("timestamp", np.nan), np.nan)
        gt0 = nearest_gt_pose(gt_poses, ts0, self.max_gt_time_gap)
        gt1 = nearest_gt_pose(gt_poses, ts1, self.max_gt_time_gap)

        if gt0 is None or gt1 is None:
            return None, "gt_association_fail"

        trans_err, rot_err = relative_pose_error(est0, est1, gt0, gt1)

        trans_good = safe_exp_good(trans_err, self.trans_scale_m)
        rot_good = safe_exp_good(rot_err, self.rot_scale_deg)
        reliability = clip01(0.65 * trans_good + 0.35 * rot_good)

        y_fail = int((trans_err > self.trans_threshold_m) or (rot_err > self.rot_threshold_deg))

        out = {
            "label_source": "gt_future",
            "label_reg": float(reliability),
            "y_fail": int(y_fail),
            "future_drift_trans_m": float(trans_err),
            "future_drift_rot_deg": float(rot_err),
            "future_drift_risk": float(1.0 - reliability),
            "future_proxy_risk": np.nan,
        }
        return out, "gt_ok"

    def _compute_proxy_label(
        self,
        rows: List[Dict[str, str]],
        i: int,
        j: int,
    ) -> Tuple[Dict[str, object], str]:
        win = rows[i + 1:j + 1]
        if len(win) == 0:
            win = [rows[j]]

        def vals(name):
            return np.array([to_float(r.get(name, np.nan), np.nan) for r in win], dtype=np.float64)

        def nanmax_nonneg(name, default=0.0):
            v = vals(name)
            v = v[np.isfinite(v) & (v >= 0.0)]
            if len(v) == 0:
                return float(default)
            return float(np.max(v))

        def nanmin_nonneg(name, default=0.0):
            v = vals(name)
            v = v[np.isfinite(v) & (v >= 0.0)]
            if len(v) == 0:
                return float(default)
            return float(np.min(v))

        max_dp = nanmax_nonneg("delta_p_norm", 0.0)
        max_dq = nanmax_nonneg("delta_q_deg", 0.0)
        max_outlier = nanmax_nonneg("outlier_ratio_last", 0.0)
        max_solver = nanmax_nonneg("solver_time_ms_last", 0.0)
        min_inlier = nanmin_nonneg("inlier_count_last", self.proxy_scales.get("inlier_count_last", 80.0))
        min_coverage = nanmin_nonneg("coverage_8x8", 1.0)
        min_entropy = nanmin_nonneg("feature_entropy_8x8", 1.0)
        any_failure = int(nanmax_nonneg("failure_detected_last", 0.0) > 0.5)

        jump_risk = 0.5 * clip01(max_dp / self.proxy_scales["delta_p_norm"]) + 0.5 * clip01(max_dq / self.proxy_scales["delta_q_deg"])
        outlier_risk = clip01(max_outlier / self.proxy_scales["outlier_ratio_last"])
        solver_risk = clip01(max_solver / self.proxy_scales["solver_time_ms_last"])
        low_inlier_risk = clip01(1.0 - min_inlier / max(self.proxy_scales["inlier_count_last"], 1e-6))
        low_spread_risk = 0.5 * clip01(1.0 - min_coverage) + 0.5 * clip01(1.0 - min_entropy)

        proxy_risk = clip01(
            0.38 * jump_risk +
            0.20 * outlier_risk +
            0.14 * solver_risk +
            0.14 * low_inlier_risk +
            0.09 * low_spread_risk +
            0.05 * float(any_failure)
        )

        reliability = clip01(1.0 - proxy_risk)

        # proxy mode 沒有真正 GT drift，因此這兩欄用 horizon 內最大 pose update 當 debug proxy
        out = {
            "label_source": "proxy_future",
            "label_reg": float(reliability),
            "y_fail": int(proxy_risk > (1.0 - self.harmful_thr)),
            "future_drift_trans_m": float(max_dp),
            "future_drift_rot_deg": float(max_dq),
            "future_drift_risk": float(proxy_risk),
            "future_proxy_risk": float(proxy_risk),
            "proxy_max_delta_p_norm": float(max_dp),
            "proxy_max_delta_q_deg": float(max_dq),
            "proxy_max_outlier_ratio": float(max_outlier),
            "proxy_max_solver_time_ms": float(max_solver),
            "proxy_min_inlier_count": float(min_inlier),
            "proxy_min_coverage_8x8": float(min_coverage),
            "proxy_min_entropy_8x8": float(min_entropy),
        }
        return out, "proxy_ok"

    def build(self):
        self._check_inputs()

        rows = prepare_feature_rows(self.feature_csv)
        groups = group_rows(rows)
        self._fit_proxy_scales(rows)

        gt_poses = None
        actual_mode = self.label_mode
        if self.label_mode in ["auto", "gt_future"]:
            if self.gt_csv is not None and self.gt_csv.exists():
                gt_poses = load_gt_poses(self.gt_csv)
                actual_mode = "gt_future"
            elif self.label_mode == "gt_future":
                raise FileNotFoundError("label_mode=gt_future 但沒有提供有效 gt_csv")
            else:
                actual_mode = "proxy_future"

        label_rows: List[Dict[str, object]] = []
        debug_rows: List[Dict[str, object]] = []

        num_skipped = 0
        skip_reasons: Dict[str, int] = {}

        for group_key, group in groups.items():
            dataset_name, sequence_name, run_id = group_key

            for i, row0 in enumerate(group):
                j = find_future_index_in_group(
                    group,
                    i=i,
                    horizon_steps=self.horizon_steps,
                    horizon_seconds=self.horizon_seconds,
                    max_time_gap=self.max_time_gap,
                )
                if j is None:
                    num_skipped += 1
                    skip_reasons["no_future_row"] = skip_reasons.get("no_future_row", 0) + 1
                    continue

                row1 = group[j]
                reason = ""

                if actual_mode == "gt_future":
                    label_data, reason = self._compute_gt_label(row0, row1, gt_poses)
                    if label_data is None:
                        # GT 失敗時不自動退回 proxy，避免同一批 label 混雜標準。
                        num_skipped += 1
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        debug_rows.append({
                            "pair_id": to_int(row0.get("pair_id"), -1),
                            "future_pair_id": to_int(row1.get("pair_id"), -1),
                            "dataset_name": dataset_name,
                            "sequence_name": sequence_name,
                            "run_id": run_id,
                            "status": "skipped",
                            "reason": reason,
                        })
                        continue
                else:
                    label_data, reason = self._compute_proxy_label(group, i, j)

                pair_id = to_int(row0.get("pair_id"), -1)
                future_pair_id = to_int(row1.get("pair_id"), -1)
                label_cls, class_name = self._score_to_class(float(label_data["label_reg"]))

                horizon_dt = to_float(row1.get("timestamp", np.nan), np.nan) - to_float(row0.get("timestamp", np.nan), np.nan)
                if not np.isfinite(horizon_dt):
                    horizon_dt = np.nan

                out = {
                    "pair_id": int(pair_id),
                    "future_pair_id": int(future_pair_id),
                    "source_row_index": to_int(row0.get("_source_row_index"), -1),
                    "future_source_row_index": to_int(row1.get("_source_row_index"), -1),

                    "dataset_name": dataset_name,
                    "sequence_name": sequence_name,
                    "run_id": run_id,

                    "update_id": to_int(row0.get("update_id"), pair_id),
                    "future_update_id": to_int(row1.get("update_id"), future_pair_id),
                    "timestamp": to_float(row0.get("timestamp", np.nan), np.nan),
                    "future_timestamp": to_float(row1.get("timestamp", np.nan), np.nan),
                    "horizon_steps": int(j - i),
                    "horizon_seconds": float(horizon_dt) if np.isfinite(horizon_dt) else "",

                    "label_reg": float(label_data["label_reg"]),
                    "label_cls": int(label_cls),
                    "class_name": class_name,
                    "y_fail": int(label_data["y_fail"]),

                    "label_source": label_data["label_source"],
                    "future_drift_trans_m": label_data["future_drift_trans_m"],
                    "future_drift_rot_deg": label_data["future_drift_rot_deg"],
                    "future_drift_risk": label_data["future_drift_risk"],
                    "future_proxy_risk": label_data["future_proxy_risk"],

                    "reason": reason,
                }

                # proxy debug 欄位，gt mode 會留空
                for k in [
                    "proxy_max_delta_p_norm",
                    "proxy_max_delta_q_deg",
                    "proxy_max_outlier_ratio",
                    "proxy_max_solver_time_ms",
                    "proxy_min_inlier_count",
                    "proxy_min_coverage_8x8",
                    "proxy_min_entropy_8x8",
                ]:
                    out[k] = label_data.get(k, "")

                label_rows.append(out)
                debug_rows.append(dict(out, status="ok"))

        label_rows.sort(key=lambda r: int(r["pair_id"]))
        debug_rows.sort(key=lambda r: int(r.get("pair_id", -1)))

        label_csv_path = self.out_dir / "reliability_labels.csv"
        debug_csv_path = self.out_dir / "reliability_labels_debug.csv"
        summary_json_path = self.out_dir / "reliability_labels_summary.json"

        fieldnames = [
            "pair_id",
            "future_pair_id",
            "source_row_index",
            "future_source_row_index",
            "dataset_name",
            "sequence_name",
            "run_id",
            "update_id",
            "future_update_id",
            "timestamp",
            "future_timestamp",
            "horizon_steps",
            "horizon_seconds",
            "label_reg",
            "label_cls",
            "class_name",
            "y_fail",
            "label_source",
            "future_drift_trans_m",
            "future_drift_rot_deg",
            "future_drift_risk",
            "future_proxy_risk",
            "reason",
            "proxy_max_delta_p_norm",
            "proxy_max_delta_q_deg",
            "proxy_max_outlier_ratio",
            "proxy_max_solver_time_ms",
            "proxy_min_inlier_count",
            "proxy_min_coverage_8x8",
            "proxy_min_entropy_8x8",
        ]

        write_csv_rows(label_csv_path, label_rows, fieldnames)
        write_csv_rows(debug_csv_path, debug_rows, ["status"] + fieldnames)

        class_counts = {name: 0 for name in CLASS_NAME_TO_ID.keys()}
        fail_count = 0
        label_values = []
        for r in label_rows:
            class_counts[to_str(r.get("class_name", "neutral"))] = class_counts.get(to_str(r.get("class_name", "neutral")), 0) + 1
            fail_count += int(to_int(r.get("y_fail", 0), 0))
            label_values.append(float(r["label_reg"]))

        label_arr = np.asarray(label_values, dtype=np.float64)
        summary = {
            "feature_csv": str(self.feature_csv),
            "gt_csv": str(self.gt_csv) if self.gt_csv else "",
            "out_dir": str(self.out_dir),
            "requested_label_mode": self.label_mode,
            "actual_label_mode": actual_mode,
            "horizon_steps_requested": int(self.horizon_steps),
            "horizon_seconds_requested": float(self.horizon_seconds),
            "max_time_gap": float(self.max_time_gap),
            "max_gt_time_gap": float(self.max_gt_time_gap),
            "trans_threshold_m": float(self.trans_threshold_m),
            "rot_threshold_deg": float(self.rot_threshold_deg),
            "trans_scale_m": float(self.trans_scale_m),
            "rot_scale_deg": float(self.rot_scale_deg),
            "helpful_thr": float(self.helpful_thr),
            "harmful_thr": float(self.harmful_thr),
            "num_total_feature_rows": int(len(rows)),
            "num_labeled_rows": int(len(label_rows)),
            "num_skipped": int(num_skipped),
            "skip_reasons": skip_reasons,
            "class_counts": class_counts,
            "y_fail_count": int(fail_count),
            "label_reg_mean": float(np.mean(label_arr)) if len(label_arr) else None,
            "label_reg_std": float(np.std(label_arr)) if len(label_arr) else None,
            "label_reg_min": float(np.min(label_arr)) if len(label_arr) else None,
            "label_reg_max": float(np.max(label_arr)) if len(label_arr) else None,
            "proxy_scales": self.proxy_scales,
            "note": (
                "gt_future 是正式 label；proxy_future 只適合 pipeline/debug，"
                "不可直接當作最終論文結論。"
            ),
        }

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("========== Build Reliability Labels Finished ==========")
        print(f"label csv: {label_csv_path}")
        print(f"debug csv: {debug_csv_path}")
        print(f"summary json: {summary_json_path}")
        print(f"actual_label_mode: {actual_mode}")
        print(f"num_total_feature_rows: {len(rows)}")
        print(f"num_labeled_rows: {len(label_rows)}")
        print(f"class_counts: {class_counts}")
        print(f"y_fail_count: {fail_count}")
        if actual_mode == "proxy_future":
            print("[WARN] 目前使用 proxy_future，正式論文建議改用 --gt_csv 做 gt_future label。")

        return {
            "label_csv": str(label_csv_path),
            "debug_csv": str(debug_csv_path),
            "summary_json": str(summary_json_path),
            "num_labeled_rows": int(len(label_rows)),
            "class_counts": class_counts,
            "y_fail_count": int(fail_count),
            "actual_label_mode": actual_mode,
        }


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--feature_csv", type=str, default="")
    parser.add_argument("--gt_csv", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")

    parser.add_argument(
        "--label_mode",
        type=str,
        default="auto",
        choices=["auto", "gt_future", "proxy_future"],
        help="auto: 有 gt_csv 就用 gt_future，否則 proxy_future；gt_future: 正式 label；proxy_future: 沒 GT 的暫時 label。",
    )

    parser.add_argument("--horizon_steps", type=int, default=50, help="預設 50 steps；若你的 feature 是 10Hz 約等於 5 秒。")
    parser.add_argument("--horizon_seconds", type=float, default=0.0, help=">0 時改用 timestamp 找未來 H 秒的 row。")
    parser.add_argument("--max_time_gap", type=float, default=0.05)
    parser.add_argument("--max_gt_time_gap", type=float, default=0.05)

    parser.add_argument("--trans_threshold_m", type=float, default=0.50)
    parser.add_argument("--rot_threshold_deg", type=float, default=5.0)
    parser.add_argument("--trans_scale_m", type=float, default=0.50)
    parser.add_argument("--rot_scale_deg", type=float, default=5.0)

    parser.add_argument("--helpful_thr", type=float, default=0.70)
    parser.add_argument("--harmful_thr", type=float, default=0.40)

    return parser.parse_args()


def main():
    args = parse_args()
    builder = ReliabilityLabelBuilder(
        sequence_dir=args.sequence_dir,
        feature_csv=args.feature_csv,
        gt_csv=args.gt_csv,
        out_dir=args.out_dir,
        label_mode=args.label_mode,
        horizon_steps=args.horizon_steps,
        horizon_seconds=args.horizon_seconds,
        max_time_gap=args.max_time_gap,
        max_gt_time_gap=args.max_gt_time_gap,
        trans_threshold_m=args.trans_threshold_m,
        rot_threshold_deg=args.rot_threshold_deg,
        trans_scale_m=args.trans_scale_m,
        rot_scale_deg=args.rot_scale_deg,
        helpful_thr=args.helpful_thr,
        harmful_thr=args.harmful_thr,
    )
    builder.build()


if __name__ == "__main__":
    main()
