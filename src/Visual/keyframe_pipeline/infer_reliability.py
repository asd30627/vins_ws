#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_reliability.py

Offline inference for the future-drift-aware VINS-Fusion reliability model.

This version matches the new training pipeline:
  - reads feature_names / fill_values / mean / std / seq_len from dataset_stats.json
  - reads model_type / hidden_dim / num_layers / dropout / input_dim from model_config.json
  - loads best.pt saved as a raw PyTorch state_dict
  - predicts all three heads: reg, fail, cls
  - writes reliability_predictions.csv + summary.json

Prediction columns:
  has_prediction : 0 for warmup rows where seq_len history is not yet available
  w_pred         : visual reliability score, 0~1, larger means safer/more useful visual correction
  p_fail         : predicted future drift/failure probability
  p_harmful      : predicted probability of harmful visual update
  p_neutral      : predicted probability of neutral visual update
  p_helpful      : predicted probability of helpful visual update
  visual_weight  : suggested soft visual correction weight
  gate_pass      : 1 means visual correction is allowed; 0 means visual-safe / reject
  mode_code      : -1 warmup, 0 normal, 1 degraded, 2 visual_safe
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from build_reliability_dataset import NONNEGATIVE_FEATURES, CLASS_ID_TO_NAME
from train_reliability_model import build_model


# =========================================================
# basic utils
# =========================================================

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


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    s = np.sum(e, axis=axis, keepdims=True)
    s = np.clip(s, 1e-12, None)
    return e / s


def is_valid_feature_value(name: str, value: float) -> bool:
    if not np.isfinite(value):
        return False
    if name in NONNEGATIVE_FEATURES and value < 0.0:
        return False
    return True


def safe_torch_load(path: Path, map_location):
    """Compatible torch.load wrapper for both old and new PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_state_dict(ckpt):
    """Accept both raw state_dict and dict checkpoint with model_state_dict."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def median_ignore_invalid(values, feature_name: str, default_value: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    valid = []
    for v in arr:
        if is_valid_feature_value(feature_name, float(v)):
            valid.append(float(v))
    if len(valid) == 0:
        return float(default_value)
    return float(np.median(np.asarray(valid, dtype=np.float64)))


def load_stats_from_train_npz(dataset_dir: Path, meta_json: Path) -> Dict[str, object]:
    """
    v4.1 dataset 沒有 dataset_stats.json。
    所以從 train.npz 讀 feature_names / seq_len / mean / std，
    從 dataset_meta.json 讀 base_feature_names。
    """
    train_npz = dataset_dir / "train.npz"
    if not train_npz.exists():
        raise FileNotFoundError(f"找不到 train.npz: {train_npz}")

    data = np.load(str(train_npz), allow_pickle=True)

    stats = {
        "feature_names": [str(x) for x in data["feature_names"].tolist()],
        "seq_len": int(data["seq_len"][0]),
        "mean": data["mean"].astype(np.float32).tolist(),
        "std": data["std"].astype(np.float32).tolist(),
        "fill_values": {},
    }

    if meta_json.exists():
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        stats["base_feature_names"] = [str(x) for x in meta.get("base_feature_names", [])]
        if not stats["base_feature_names"]:
            stats["base_feature_names"] = dedupe_keep_order([
                n[len("valid_"):] if n.startswith("valid_") else n
                for n in stats["feature_names"]
            ])
    else:
        stats["base_feature_names"] = dedupe_keep_order([
            n[len("valid_"):] if n.startswith("valid_") else n
            for n in stats["feature_names"]
        ])

    return stats


def rebuild_fill_values_from_debug_csv(dataset_dir: Path, base_feature_names: List[str]) -> Dict[str, float]:
    """
    build_reliability_dataset.py 沒有把 fill_values 寫進 dataset_meta.json。
    但 debug csv 裡有 split=train 的 raw feature，可以重建 train median fill。
    如果找不到 feature_label_debug.csv，就退回 0.0。
    """
    debug_csv = dataset_dir / "feature_label_debug.csv"
    fill_values = {name: 0.0 for name in base_feature_names}

    if not debug_csv.exists():
        print(f"[WARN] 找不到 {debug_csv}，missing feature 將用 0.0 補。")
        return fill_values

    rows = read_csv_rows(debug_csv)
    train_rows = [r for r in rows if str(r.get("split", "")).strip() == "train"]

    if len(train_rows) == 0:
        print(f"[WARN] {debug_csv} 裡沒有 split=train，missing feature 將用 0.0 補。")
        return fill_values

    for name in base_feature_names:
        vals = [to_float(r.get(name, np.nan), np.nan) for r in train_rows]
        fill_values[name] = median_ignore_invalid(vals, name, default_value=0.0)

    return fill_values

# =========================================================
# Inferencer
# =========================================================

class ReliabilityInferencer:
    def __init__(
        self,
        sequence_dir: str,
        feature_csv: str = "",
        dataset_dir: str = "",
        checkpoint: str = "",
        model_config: str = "",
        stats_json: str = "",
        out_dir: str = "",
        label_csv: str = "",
        batch_size: int = 256,
        device: Optional[str] = None,

        # gate / soft-weight thresholds
        tau_reg: float = 0.45,
        tau_fail: float = 0.65,
        tau_harmful: float = 0.50,
        hard_fail_thr: float = 0.80,
        hard_harmful_thr: float = 0.70,
        hard_reg_min: float = 0.20,
        degraded_weight_thr: float = 0.60,
        safe_weight: float = 0.0,
        min_degraded_weight: float = 0.05,

        # fallback model config if model_config.json does not exist
        model_type: str = "gru",
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.10,
        num_classes: int = 3,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.feature_csv = (
            Path(feature_csv).expanduser().resolve()
            if feature_csv
            else self.sequence_dir / "features" / "all_candidate_features.csv"
        )
        self.dataset_dir = (
            Path(dataset_dir).expanduser().resolve()
            if dataset_dir
            else self.sequence_dir / "reliability_dataset"
        )
        self.out_dir = (
            Path(out_dir).expanduser().resolve()
            if out_dir
            else self.sequence_dir / "reliability_inference"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint
            else self.dataset_dir / "model_runs" / model_type / "best.pt"
        )
        self.model_config_json = (
            Path(model_config).expanduser().resolve()
            if model_config
            else self.checkpoint.parent / "model_config.json"
        )
        self.stats_json = (
            Path(stats_json).expanduser().resolve()
            if stats_json
            else self.dataset_dir / "dataset_stats.json"
        )
        self.label_csv = (
            Path(label_csv).expanduser().resolve()
            if label_csv
            else self.sequence_dir / "reliability_labels" / "reliability_labels.csv"
        )

        self.batch_size = int(batch_size)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tau_reg = float(tau_reg)
        self.tau_fail = float(tau_fail)
        self.tau_harmful = float(tau_harmful)
        self.hard_fail_thr = float(hard_fail_thr)
        self.hard_harmful_thr = float(hard_harmful_thr)
        self.hard_reg_min = float(hard_reg_min)
        self.degraded_weight_thr = float(degraded_weight_thr)
        self.safe_weight = float(safe_weight)
        self.min_degraded_weight = float(min_degraded_weight)

        self.fallback_model_type = str(model_type)
        self.fallback_hidden_dim = int(hidden_dim)
        self.fallback_num_layers = int(num_layers)
        self.fallback_dropout = float(dropout)
        self.fallback_num_classes = int(num_classes)

        self.feature_names: List[str] = []
        self.base_feature_names: List[str] = []
        self.fill_values: Dict[str, float] = {}
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.seq_len: Optional[int] = None

        self.model_type: Optional[str] = None
        self.hidden_dim: Optional[int] = None
        self.num_layers: Optional[int] = None
        self.dropout: Optional[float] = None
        self.num_classes: Optional[int] = None
        self.input_dim: Optional[int] = None

    # -----------------------------------------------------
    # loading metadata
    # -----------------------------------------------------
    def check_required_files(self):
        if not self.feature_csv.exists():
            raise FileNotFoundError(f"找不到 feature_csv: {self.feature_csv}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {self.checkpoint}")

        # v4.1 可能沒有 dataset_stats.json，但至少要有 train.npz
        if not self.stats_json.exists():
            train_npz = self.dataset_dir / "train.npz"
            meta_json = self.dataset_dir / "dataset_meta.json"
            if not train_npz.exists():
                raise FileNotFoundError(
                    f"找不到 dataset_stats.json，也找不到 train.npz。\n"
                    f"stats_json: {self.stats_json}\n"
                    f"train_npz : {train_npz}\n"
                    f"meta_json : {meta_json}"
            )

    def load_stats_and_model_config(self):
        meta_json = self.dataset_dir / "dataset_meta.json"

        if self.stats_json.exists() and self.stats_json.name == "dataset_stats.json":
            with open(self.stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)

            # 保險檢查：真的要有 mean/std 才能當 stats 檔
            if "mean" not in stats or "std" not in stats:
                print(f"[WARN] {self.stats_json} 沒有 mean/std，改從 train.npz 載入。")
                stats = load_stats_from_train_npz(self.dataset_dir, meta_json)
            else:
                print(f"[INFO] loaded dataset_stats.json: {self.stats_json}")
        else:
            stats = load_stats_from_train_npz(self.dataset_dir, meta_json)
            print(f"[INFO] dataset_stats.json 不存在或你傳的是 dataset_meta.json，改從 train.npz + dataset_meta.json 載入統計。")
            print(f"[INFO] train_npz : {self.dataset_dir / 'train.npz'}")
            print(f"[INFO] meta_json : {meta_json}")

        self.feature_names = [str(x) for x in stats["feature_names"]]
        
        if "base_feature_names" in stats:
            self.base_feature_names = [str(x) for x in stats["base_feature_names"]]
        else:
            self.base_feature_names = dedupe_keep_order([
                n[len("valid_"):] if n.startswith("valid_") else n
                for n in self.feature_names
            ])

        self.fill_values = {str(k): float(v) for k, v in stats.get("fill_values", {}).items()}

        # v4.1 沒有保存 fill_values，從 feature_label_debug.csv 的 train split 重建
        if len(self.fill_values) == 0:
            self.fill_values = rebuild_fill_values_from_debug_csv(self.dataset_dir, self.base_feature_names)

        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        self.seq_len = int(stats["seq_len"])
        self.input_dim = int(len(self.feature_names))

        if len(self.mean) != self.input_dim or len(self.std) != self.input_dim:
            raise ValueError(
                f"stats 維度不一致: len(feature_names)={self.input_dim}, "
                f"len(mean)={len(self.mean)}, len(std)={len(self.std)}"
            )

        if self.model_config_json.exists():
            with open(self.model_config_json, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.model_type = str(cfg.get("model_type", self.fallback_model_type))
            self.hidden_dim = int(cfg.get("hidden_dim", self.fallback_hidden_dim))
            self.num_layers = int(cfg.get("num_layers", self.fallback_num_layers))
            self.dropout = float(cfg.get("dropout", self.fallback_dropout))
            self.num_classes = int(cfg.get("num_classes", self.fallback_num_classes))

            cfg_input_dim = int(cfg.get("input_dim", self.input_dim))
            cfg_seq_len = int(cfg.get("seq_len", self.seq_len))
            if cfg_input_dim != self.input_dim:
                raise ValueError(f"model_config input_dim={cfg_input_dim} 但 stats input_dim={self.input_dim}")
            if cfg_seq_len != self.seq_len:
                raise ValueError(f"model_config seq_len={cfg_seq_len} 但 stats seq_len={self.seq_len}")
        else:
            self.model_type = self.fallback_model_type
            self.hidden_dim = self.fallback_hidden_dim
            self.num_layers = self.fallback_num_layers
            self.dropout = self.fallback_dropout
            self.num_classes = self.fallback_num_classes

        print(f"[INFO] seq_len={self.seq_len}, input_dim={self.input_dim}, num_features={len(self.feature_names)}")

    def load_label_map(self) -> Dict[int, Dict[str, object]]:
        if not self.label_csv.exists():
            return {}

        rows = read_csv_rows(self.label_csv)
        label_map = {}
        for row in rows:
            pair_id = to_int(row.get("pair_id", -1), -1)
            if pair_id < 0:
                continue
            label_map[pair_id] = {
                "label_reg": to_float(row.get("label_reg", np.nan), np.nan),
                "label_cls": to_int(row.get("label_cls", -1), -1),
                "class_name": to_str(row.get("class_name", "")),
                "y_fail": to_int(row.get("y_fail", -1), -1),
                "future_drift_trans_m": to_float(row.get("future_drift_trans_m", np.nan), np.nan),
                "future_drift_rot_deg": to_float(row.get("future_drift_rot_deg", np.nan), np.nan),
                "label_source": to_str(row.get("label_source", "")),
            }
        return label_map

    # -----------------------------------------------------
    # feature matrix
    # -----------------------------------------------------
    @staticmethod
    def row_sort_key(item):
        idx, row = item
        dataset_name = to_str(row.get("dataset_name", ""))
        sequence_name = to_str(row.get("sequence_name", ""))
        run_id = to_str(row.get("run_id", ""))
        ts = to_float(row.get("timestamp", np.nan), np.nan)
        upd = to_int(row.get("update_id", idx), idx)
        pid = to_int(row.get("pair_id", idx), idx)
        if not np.isfinite(ts):
            ts = float(idx)
        return (dataset_name, sequence_name, run_id, ts, upd, pid, idx)

    def prepare_rows(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        indexed = list(enumerate(rows))
        indexed.sort(key=self.row_sort_key)
        out = []
        for global_idx, (orig_idx, row) in enumerate(indexed):
            r = dict(row)
            pid = to_int(r.get("pair_id", -1), -1)
            if pid < 0:
                pid = global_idx
            r["pair_id"] = str(pid)
            r["_source_row_index"] = str(orig_idx)
            r["_sorted_row_index"] = str(global_idx)
            out.append(r)
        return out

    def build_feature_table(self, rows: List[Dict[str, str]]):
        pair_ids = []
        group_keys = []
        meta = {}
        cols = {name: [] for name in self.feature_names}

        for row in rows:
            pid = to_int(row.get("pair_id", -1), -1)
            if pid < 0:
                continue
            pair_ids.append(pid)

            group_key = (
                to_str(row.get("dataset_name", "")),
                to_str(row.get("sequence_name", "")),
                to_str(row.get("run_id", "")),
            )
            group_keys.append(group_key)

            meta[pid] = {
                "pair_id": int(pid),
                "dataset_name": group_key[0],
                "sequence_name": group_key[1],
                "run_id": group_key[2],
                "update_id": to_int(row.get("update_id", pid), pid),
                "frame_count": to_int(row.get("frame_count", -1), -1),
                "timestamp": to_float(row.get("timestamp", np.nan), np.nan),
                "source_row_index": to_int(row.get("_source_row_index", -1), -1),
                "sorted_row_index": to_int(row.get("_sorted_row_index", -1), -1),
            }

            for name in self.feature_names:
                if name.startswith("valid_"):
                    base_name = name[len("valid_"):]
                    raw_v = to_float(row.get(base_name, np.nan), np.nan)
                    cols[name].append(1.0 if is_valid_feature_value(base_name, raw_v) else 0.0)
                else:
                    raw_v = to_float(row.get(name, np.nan), np.nan)
                    cols[name].append(raw_v if is_valid_feature_value(name, raw_v) else np.nan)

        table = {
            "pair_ids": np.asarray(pair_ids, dtype=np.int32),
            "group_keys": group_keys,
            "meta": meta,
        }
        for name in self.feature_names:
            table[name] = np.asarray(cols[name], dtype=np.float64)
        return table

    def apply_fill_values(self, table) -> np.ndarray:
        X_cols = []
        for name in self.feature_names:
            arr = table[name].copy()
            if name.startswith("valid_"):
                fill_v = 0.0
            else:
                fill_v = float(self.fill_values.get(name, 0.0))
            arr[~np.isfinite(arr)] = fill_v
            X_cols.append(arr.reshape(-1, 1))
        return np.concatenate(X_cols, axis=1).astype(np.float32)

    def standardize_features(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)).astype(np.float32)

    def build_sequences(self, X: np.ndarray, pair_ids: np.ndarray, group_keys: List[Tuple[str, str, str]]):
        xs = []
        seq_meta = []
        n = len(X)
        if n == 0:
            return np.empty((0, int(self.seq_len), X.shape[1]), dtype=np.float32), []

        for end_idx in range(int(self.seq_len) - 1, n):
            start_idx = end_idx - int(self.seq_len) + 1
            # 不允許 sequence 跨 dataset/sequence/run
            if len(set(group_keys[start_idx:end_idx + 1])) != 1:
                continue
            xs.append(X[start_idx:end_idx + 1])
            seq_meta.append({
                "start_pair_id": int(pair_ids[start_idx]),
                "end_pair_id": int(pair_ids[end_idx]),
                "start_index": int(start_idx),
                "end_index": int(end_idx),
                "group_key": group_keys[end_idx],
            })

        if len(xs) == 0:
            return np.empty((0, int(self.seq_len), X.shape[1]), dtype=np.float32), []
        return np.stack(xs).astype(np.float32), seq_meta

    # -----------------------------------------------------
    # model inference
    # -----------------------------------------------------
    def build_model_instance(self):
        model = build_model(
            model_type=self.model_type,
            seq_len=int(self.seq_len),
            input_dim=int(self.input_dim),
            hidden_dim=int(self.hidden_dim),
            num_layers=int(self.num_layers),
            dropout=float(self.dropout),
            num_classes=int(self.num_classes),
        ).to(self.device)
        ckpt = safe_torch_load(self.checkpoint, map_location=self.device)
        model.load_state_dict(normalize_state_dict(ckpt))
        model.eval()
        return model

    @torch.no_grad()
    def infer_in_batches(self, model, X_seq: np.ndarray):
        regs = []
        fail_logits_list = []
        cls_logits_list = []

        for start in range(0, len(X_seq), self.batch_size):
            end = min(start + self.batch_size, len(X_seq))
            xb = torch.from_numpy(X_seq[start:end]).float().to(self.device)
            outputs = model(xb, return_dict=True)
            regs.append(outputs["reg"].detach().cpu().numpy().astype(np.float32).reshape(-1))
            fail_logits_list.append(outputs["fail_logits"].detach().cpu().numpy().astype(np.float32).reshape(-1))
            cls_logits_list.append(outputs["cls_logits"].detach().cpu().numpy().astype(np.float32))

        if len(regs) == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.empty((0, int(self.num_classes)), dtype=np.float32),
            )

        pred_reg = np.clip(np.concatenate(regs, axis=0).astype(np.float32), 0.0, 1.0)
        p_fail = sigmoid_np(np.concatenate(fail_logits_list, axis=0).astype(np.float32)).reshape(-1)
        cls_logits = np.concatenate(cls_logits_list, axis=0).astype(np.float32)
        cls_probs = softmax_np(cls_logits, axis=1)
        return pred_reg, p_fail, cls_probs

    # -----------------------------------------------------
    # decision logic
    # -----------------------------------------------------
    def decision_from_prediction(self, w_pred: float, p_fail: float, p_harmful: float, p_helpful: float) -> Dict[str, object]:
        """
        v4.1 第一版 inference 只做 observer / soft-weight 建議。
        不直接 hard reject，因為目前 fail_head calibration 還不夠穩。
        """
        w_pred = clip01(w_pred)
        p_fail = clip01(p_fail)
        p_harmful = clip01(p_harmful) if np.isfinite(p_harmful) else 0.0
        p_helpful = clip01(p_helpful) if np.isfinite(p_helpful) else 0.0

        # 第一版只信 reg_head，保守 clamp
        visual_weight = float(np.clip(w_pred, 0.30, 1.00))

        if visual_weight < 0.45:
            mode_code = 1
            mode_name = "degraded"
        else:
            mode_code = 0
            mode_name = "normal"

        return {
            "gate_pass": 1,
            "mode_code": int(mode_code),
            "mode_name": str(mode_name),
            "visual_weight": float(visual_weight),
            "gate_reason": (
                f"observer_soft: w={w_pred:.3f}, weight={visual_weight:.3f}, "
                f"p_fail_log_only={p_fail:.3f}, p_harmful_log_only={p_harmful:.3f}"
            ),
        }

    # -----------------------------------------------------
    # output
    # -----------------------------------------------------
    def run(self):
        self.check_required_files()
        self.load_stats_and_model_config()

        rows = self.prepare_rows(read_csv_rows(self.feature_csv))
        table = self.build_feature_table(rows)
        X = self.apply_fill_values(table)
        X = self.standardize_features(X)

        pair_ids = table["pair_ids"]
        group_keys = table["group_keys"]
        X_seq, seq_meta = self.build_sequences(X, pair_ids, group_keys)

        model = self.build_model_instance()
        pred_reg, p_fail, cls_probs = self.infer_in_batches(model, X_seq)
        pred_cls = np.argmax(cls_probs, axis=1).astype(np.int64) if len(cls_probs) > 0 else np.array([], dtype=np.int64)

        label_map = self.load_label_map()

        pred_map: Dict[int, Dict[str, object]] = {}
        for i, meta in enumerate(seq_meta):
            end_pair_id = int(meta["end_pair_id"])
            p_harmful = float(cls_probs[i, 0]) if int(self.num_classes) >= 1 else np.nan
            p_neutral = float(cls_probs[i, 1]) if int(self.num_classes) >= 2 else np.nan
            p_helpful = float(cls_probs[i, 2]) if int(self.num_classes) >= 3 else np.nan
            pred_class_id = int(pred_cls[i])
            pred_class_name = CLASS_ID_TO_NAME.get(pred_class_id, f"class_{pred_class_id}")
            decision = self.decision_from_prediction(
                w_pred=float(pred_reg[i]),
                p_fail=float(p_fail[i]),
                p_harmful=p_harmful,
                p_helpful=p_helpful,
            )
            pred_map[end_pair_id] = {
                "has_prediction": 1,
                "start_pair_id": int(meta["start_pair_id"]),
                "end_pair_id": end_pair_id,
                "w_pred": float(pred_reg[i]),
                "p_fail": float(p_fail[i]),
                "pred_class_id": pred_class_id,
                "pred_class": pred_class_name,
                "p_harmful": p_harmful,
                "p_neutral": p_neutral,
                "p_helpful": p_helpful,
                "visual_weight": float(decision["visual_weight"]),
                "gate_pass": int(decision["gate_pass"]),
                "mode_code": int(decision["mode_code"]),
                "mode_name": str(decision["mode_name"]),
                "gate_reason": str(decision["gate_reason"]),
                "soft_target_proxy": float(label_map.get(end_pair_id, {}).get("label_reg", np.nan)),
            }

        out_rows = []
        for pid in pair_ids.tolist():
            meta = table["meta"][int(pid)]
            base = {
                "pair_id": int(pid),
                "start_pair_id": -1,
                "end_pair_id": int(pid),
                "dataset_name": meta["dataset_name"],
                "sequence_name": meta["sequence_name"],
                "run_id": meta["run_id"],
                "update_id": int(meta["update_id"]),
                "frame_count": int(meta["frame_count"]),
                "timestamp": meta["timestamp"],
                "source_row_index": int(meta["source_row_index"]),
                "sorted_row_index": int(meta["sorted_row_index"]),
                "has_prediction": 0,
                "w_pred": np.nan,
                "p_fail": np.nan,
                "pred_class_id": -1,
                "pred_class": "",
                "p_harmful": np.nan,
                "p_neutral": np.nan,
                "p_helpful": np.nan,
                "visual_weight": 1.0,
                "gate_pass": 1,
                "mode_code": -1,
                "mode_name": "warmup",
                "gate_reason": "warmup: seq_len history not ready",
                "soft_target_proxy": np.nan,
                "label_reg_gt": np.nan,
                "label_cls_gt": -1,
                "label_class_name_gt": "",
                "y_fail_gt": -1,
                "future_drift_trans_m_gt": np.nan,
                "future_drift_rot_deg_gt": np.nan,
                "label_source_gt": "",
            }

            if int(pid) in pred_map:
                base.update(pred_map[int(pid)])

            if int(pid) in label_map:
                gt = label_map[int(pid)]
                base["label_reg_gt"] = float(gt.get("label_reg", np.nan))
                base["label_cls_gt"] = int(gt.get("label_cls", -1))
                base["label_class_name_gt"] = str(gt.get("class_name", ""))
                base["y_fail_gt"] = int(gt.get("y_fail", -1))
                base["future_drift_trans_m_gt"] = float(gt.get("future_drift_trans_m", np.nan))
                base["future_drift_rot_deg_gt"] = float(gt.get("future_drift_rot_deg", np.nan))
                base["label_source_gt"] = str(gt.get("label_source", ""))
                if np.isfinite(base["label_reg_gt"]):
                    base["soft_target_proxy"] = float(base["label_reg_gt"])

            out_rows.append(base)

        pred_csv = self.out_dir / "reliability_predictions.csv"
        summary_json = self.out_dir / "summary.json"

        fieldnames = [
            "pair_id", "start_pair_id", "end_pair_id",
            "dataset_name", "sequence_name", "run_id", "update_id", "frame_count", "timestamp",
            "source_row_index", "sorted_row_index",
            "has_prediction", "w_pred", "p_fail",
            "pred_class_id", "pred_class", "p_harmful", "p_neutral", "p_helpful",
            "visual_weight", "gate_pass", "mode_code", "mode_name", "gate_reason",
            "soft_target_proxy",
            "label_reg_gt", "label_cls_gt", "label_class_name_gt", "y_fail_gt",
            "future_drift_trans_m_gt", "future_drift_rot_deg_gt", "label_source_gt",
        ]
        write_csv_rows(pred_csv, out_rows, fieldnames)

        num_total = len(out_rows)
        num_pred = int(sum(int(r["has_prediction"]) for r in out_rows))
        mode_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {}
        for r in out_rows:
            if int(r["has_prediction"]) == 1:
                mode_counts[str(r["mode_name"])] = mode_counts.get(str(r["mode_name"]), 0) + 1
                class_counts[str(r["pred_class"])] = class_counts.get(str(r["pred_class"]), 0) + 1

        summary = {
            "sequence_dir": str(self.sequence_dir),
            "feature_csv": str(self.feature_csv),
            "dataset_dir": str(self.dataset_dir),
            "checkpoint": str(self.checkpoint),
            "model_config_json": str(self.model_config_json),
            "stats_json": str(self.stats_json),
            "label_csv": str(self.label_csv),
            "pred_csv": str(pred_csv),
            "device": str(self.device),
            "model_type": str(self.model_type),
            "seq_len": int(self.seq_len),
            "input_dim": int(self.input_dim),
            "num_classes": int(self.num_classes),
            "feature_names": list(self.feature_names),
            "tau_reg": float(self.tau_reg),
            "tau_fail": float(self.tau_fail),
            "tau_harmful": float(self.tau_harmful),
            "hard_fail_thr": float(self.hard_fail_thr),
            "hard_harmful_thr": float(self.hard_harmful_thr),
            "hard_reg_min": float(self.hard_reg_min),
            "degraded_weight_thr": float(self.degraded_weight_thr),
            "safe_weight": float(self.safe_weight),
            "min_degraded_weight": float(self.min_degraded_weight),
            "num_total_pairs": int(num_total),
            "num_with_prediction": int(num_pred),
            "num_without_prediction": int(num_total - num_pred),
            "mode_counts": mode_counts,
            "pred_class_counts": class_counts,
            "num_gate_reject": int(sum(1 for r in out_rows if int(r["has_prediction"]) == 1 and int(r["gate_pass"]) == 0)),
            "num_gate_pass": int(sum(1 for r in out_rows if int(r["has_prediction"]) == 1 and int(r["gate_pass"]) == 1)),
            "gt_overlap_rows": int(sum(1 for r in out_rows if np.isfinite(to_float(r["label_reg_gt"], np.nan)))),
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("========== Infer Reliability Finished ==========")
        print(f"pred_csv     : {pred_csv}")
        print(f"summary_json : {summary_json}")
        print(f"num_total_pairs      : {summary['num_total_pairs']}")
        print(f"num_with_prediction  : {summary['num_with_prediction']}")
        print(f"mode_counts          : {summary['mode_counts']}")
        print(f"pred_class_counts    : {summary['pred_class_counts']}")
        print(f"num_gate_pass/reject : {summary['num_gate_pass']} / {summary['num_gate_reject']}")

        return {
            "pred_csv": str(pred_csv),
            "summary_json": str(summary_json),
            "summary": summary,
        }


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--feature_csv", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_config", type=str, default="")
    parser.add_argument("--stats_json", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--label_csv", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="")

    parser.add_argument("--tau_reg", type=float, default=0.45)
    parser.add_argument("--tau_fail", type=float, default=0.65)
    parser.add_argument("--tau_harmful", type=float, default=0.50)
    parser.add_argument("--hard_fail_thr", type=float, default=0.80)
    parser.add_argument("--hard_harmful_thr", type=float, default=0.70)
    parser.add_argument("--hard_reg_min", type=float, default=0.20)
    parser.add_argument("--degraded_weight_thr", type=float, default=0.60)
    parser.add_argument("--safe_weight", type=float, default=0.0)
    parser.add_argument("--min_degraded_weight", type=float, default=0.05)

    # fallback config when model_config.json does not exist
    parser.add_argument("--model_type", type=str, default="gru", choices=["mlp", "gru", "tcn"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--num_classes", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = ReliabilityInferencer(
        sequence_dir=args.sequence_dir,
        feature_csv=args.feature_csv,
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        stats_json=args.stats_json,
        out_dir=args.out_dir,
        label_csv=args.label_csv,
        batch_size=args.batch_size,
        device=args.device.strip() if args.device.strip() else None,
        tau_reg=args.tau_reg,
        tau_fail=args.tau_fail,
        tau_harmful=args.tau_harmful,
        hard_fail_thr=args.hard_fail_thr,
        hard_harmful_thr=args.hard_harmful_thr,
        hard_reg_min=args.hard_reg_min,
        degraded_weight_thr=args.degraded_weight_thr,
        safe_weight=args.safe_weight,
        min_degraded_weight=args.min_degraded_weight,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=args.num_classes,
    )
    inferencer.run()


if __name__ == "__main__":
    main()
