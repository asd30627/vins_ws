#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reliability_infer_node.py

ROS2 online inference node for the future-drift-aware VINS-Fusion reliability model.

Recommended runtime input:
  /vins_admission/features_json  std_msgs/String

The String payload is a JSON object. Example:
{
  "timestamp": 123.456,
  "tracked_feature_count_raw": 180,
  "tracked_feature_count_mgr": 145,
  "current_is_keyframe": 1,
  "outlier_ratio_last": 0.08,
  "solver_time_ms_last": 12.3,
  "acc_norm_mean": 9.81,
  "gyr_norm_mean": 0.03
}

The node does NOT hard-code the training feature schema. It reads:
  dataset_stats.json: feature_names / fill_values / mean / std / seq_len
  model_config.json : model_type / hidden_dim / num_layers / dropout / input_dim
  best.pt           : raw state_dict saved by train_reliability_model.py

Prediction Float64MultiArray format:
  data = [
    stamp_sec,
    has_prediction,    # 0 during warmup, 1 after seq_len history is available
    w_pred,            # reliability score, 0~1
    p_fail,            # future drift/failure probability
    p_harmful,
    p_neutral,
    p_helpful,
    gate_pass,         # 1 allow visual correction, 0 visual-safe/reject
    mode_code,         # -1 warmup, 0 normal, 1 degraded, 2 visual_safe
    visual_weight      # suggested soft visual weight
  ]

Optional prediction JSON is also published for debugging.
"""

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import torch

from build_reliability_dataset import NONNEGATIVE_FEATURES, CLASS_ID_TO_NAME
from train_reliability_model import build_model


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


def clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def sigmoid_float(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(float(x), -80.0, 80.0))))


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / max(float(s), 1e-12)


def is_valid_feature_value(name: str, value: float) -> bool:
    if not np.isfinite(value):
        return False
    if name in NONNEGATIVE_FEATURES and value < 0.0:
        return False
    return True


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def parse_csv_names(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


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


def read_csv_rows(path: Path):
    import csv
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_stats_from_train_npz(dataset_dir: Path, meta_json: Path) -> Dict[str, object]:
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
    debug_csv = dataset_dir / "feature_label_debug.csv"
    fill_values = {name: 0.0 for name in base_feature_names}

    if not debug_csv.exists():
        return fill_values

    rows = read_csv_rows(debug_csv)
    train_rows = [r for r in rows if str(r.get("split", "")).strip() == "train"]

    if len(train_rows) == 0:
        return fill_values

    for name in base_feature_names:
        vals = [to_float(r.get(name, np.nan), np.nan) for r in train_rows]
        fill_values[name] = median_ignore_invalid(vals, name, default_value=0.0)

    return fill_values

# =========================================================
# ROS2 node
# =========================================================

class ReliabilityInferNode(Node):
    def __init__(self):
        super().__init__("reliability_infer_node")

        # I/O
        self.declare_parameter("feature_topic", "/vins_admission/features_json")
        self.declare_parameter("feature_format", "json")  # json or array
        self.declare_parameter("array_feature_names", "")
        self.declare_parameter("prediction_topic", "/vins_admission/prediction")
        self.declare_parameter("prediction_json_topic", "/vins_admission/prediction_json")
        self.declare_parameter("publish_prediction_json", True)

        # model paths
        self.declare_parameter("dataset_dir", "")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("stats_json", "")
        self.declare_parameter("model_config", "")

        # runtime
        self.declare_parameter("device", "cpu")
        self.declare_parameter("warmup_pass", True)
        self.declare_parameter("drop_old_stamp", True)
        self.declare_parameter("log_missing_features", False)

        # gate / soft weighting thresholds
        self.declare_parameter("tau_reg", 0.45)
        self.declare_parameter("tau_fail", 0.65)
        self.declare_parameter("tau_harmful", 0.50)
        self.declare_parameter("hard_fail_thr", 0.80)
        self.declare_parameter("hard_harmful_thr", 0.70)
        self.declare_parameter("hard_reg_min", 0.20)
        self.declare_parameter("degraded_weight_thr", 0.60)
        self.declare_parameter("safe_weight", 0.0)
        self.declare_parameter("min_degraded_weight", 0.05)

        # fallback model config if model_config.json is unavailable
        self.declare_parameter("model_type", "gru")
        self.declare_parameter("hidden_dim", 64)
        self.declare_parameter("num_layers", 1)
        self.declare_parameter("dropout", 0.10)
        self.declare_parameter("num_classes", 3)
        self.declare_parameter("observer_only", True)
        self.declare_parameter("min_soft_weight", 0.30)

        self.observer_only = bool(self.get_parameter("observer_only").value)
        self.min_soft_weight = float(self.get_parameter("min_soft_weight").value)
        self.feature_topic = self.get_parameter("feature_topic").value
        self.feature_format = str(self.get_parameter("feature_format").value).strip().lower()
        self.array_feature_names = parse_csv_names(self.get_parameter("array_feature_names").value)
        self.prediction_topic = self.get_parameter("prediction_topic").value
        self.prediction_json_topic = self.get_parameter("prediction_json_topic").value
        self.publish_prediction_json = bool(self.get_parameter("publish_prediction_json").value)

        self.dataset_dir = str(self.get_parameter("dataset_dir").value).strip()
        self.checkpoint_path = str(self.get_parameter("checkpoint").value).strip()
        self.stats_json_path = str(self.get_parameter("stats_json").value).strip()
        self.model_config_path = str(self.get_parameter("model_config").value).strip()

        self.device = torch.device(str(self.get_parameter("device").value).strip())
        self.warmup_pass = bool(self.get_parameter("warmup_pass").value)
        self.drop_old_stamp = bool(self.get_parameter("drop_old_stamp").value)
        self.log_missing_features = bool(self.get_parameter("log_missing_features").value)

        self.tau_reg = float(self.get_parameter("tau_reg").value)
        self.tau_fail = float(self.get_parameter("tau_fail").value)
        self.tau_harmful = float(self.get_parameter("tau_harmful").value)
        self.hard_fail_thr = float(self.get_parameter("hard_fail_thr").value)
        self.hard_harmful_thr = float(self.get_parameter("hard_harmful_thr").value)
        self.hard_reg_min = float(self.get_parameter("hard_reg_min").value)
        self.degraded_weight_thr = float(self.get_parameter("degraded_weight_thr").value)
        self.safe_weight = float(self.get_parameter("safe_weight").value)
        self.min_degraded_weight = float(self.get_parameter("min_degraded_weight").value)

        self.fallback_model_type = str(self.get_parameter("model_type").value)
        self.fallback_hidden_dim = int(self.get_parameter("hidden_dim").value)
        self.fallback_num_layers = int(self.get_parameter("num_layers").value)
        self.fallback_dropout = float(self.get_parameter("dropout").value)
        self.fallback_num_classes = int(self.get_parameter("num_classes").value)

        self.feature_names: List[str] = []
        self.base_feature_names: List[str] = []
        self.fill_values: Dict[str, float] = {}
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.seq_len: int = 1
        self.input_dim: int = 0
        self.num_classes: int = 3

        self.model = None
        self.feature_buffer = deque(maxlen=1)
        self.last_stamp = -float("inf")
        self.num_received = 0
        self.num_predicted = 0
        self.missing_feature_warned = set()

        self.publisher = self.create_publisher(Float64MultiArray, self.prediction_topic, 10)
        self.publisher_json = None
        if self.publish_prediction_json:
            self.publisher_json = self.create_publisher(String, self.prediction_json_topic, 10)

        self._load_model_and_stats()

        if self.feature_format == "json":
            self.subscription = self.create_subscription(String, self.feature_topic, self.feature_json_callback, 10)
        elif self.feature_format == "array":
            self.subscription = self.create_subscription(Float64MultiArray, self.feature_topic, self.feature_array_callback, 10)
        else:
            raise RuntimeError("feature_format 只支援 json 或 array")

        self.get_logger().info(
            f"ReliabilityInferNode ready. format={self.feature_format}, feature_topic={self.feature_topic}, "
            f"prediction_topic={self.prediction_topic}, seq_len={self.seq_len}, input_dim={self.input_dim}, "
            f"device={self.device}"
        )

    # -----------------------------------------------------
    # loading
    # -----------------------------------------------------
    def _resolve_paths(self):
        dataset_dir = Path(self.dataset_dir).expanduser().resolve() if self.dataset_dir else None

        if not self.checkpoint_path:
            if dataset_dir is None:
                raise RuntimeError("必須提供 checkpoint，或提供 dataset_dir 讓 node 自動找 dataset_dir/model_runs/<model_type>/best.pt")
            self.checkpoint_path = str(dataset_dir / "model_runs" / self.fallback_model_type / "best.pt")

        checkpoint = Path(self.checkpoint_path).expanduser().resolve()

        if not self.stats_json_path:
            if dataset_dir is None:
                # checkpoint = dataset_dir/model_runs/<model_type>/best.pt
                dataset_dir = checkpoint.parent.parent.parent

            stats_candidate = dataset_dir / "dataset_stats.json"
            meta_candidate = dataset_dir / "dataset_meta.json"

            if stats_candidate.exists():
                self.stats_json_path = str(stats_candidate)
            else:
                self.stats_json_path = str(meta_candidate)

        if not self.model_config_path:
            self.model_config_path = str(checkpoint.parent / "model_config.json")

        return (
            checkpoint,
            Path(self.stats_json_path).expanduser().resolve(),
            Path(self.model_config_path).expanduser().resolve(),
        )

    def _load_model_and_stats(self):
        checkpoint, stats_json, model_config = self._resolve_paths()

        if not stats_json.exists():
            raise FileNotFoundError(f"找不到 stats/meta json: {stats_json}")

        # v4.1：stats_json_path 可能是 dataset_meta.json；mean/std 要從 train.npz 讀
        if stats_json.name == "dataset_stats.json":
            with open(stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)
            self.get_logger().info(f"loaded dataset_stats.json: {stats_json}")
        else:
            dataset_dir = stats_json.parent
            stats = load_stats_from_train_npz(dataset_dir, stats_json)
            self.get_logger().info("dataset_stats.json 不存在，改從 train.npz + dataset_meta.json 載入統計")
            self.get_logger().info(f"train_npz : {dataset_dir / 'train.npz'}")
            self.get_logger().info(f"meta_json : {stats_json}")

        self.feature_names = [str(x) for x in stats["feature_names"]]

        if "base_feature_names" in stats:
            self.base_feature_names = [str(x) for x in stats["base_feature_names"]]
        else:
            self.base_feature_names = dedupe_keep_order([
                n[len("valid_"):] if n.startswith("valid_") else n
                for n in self.feature_names
            ])

        self.fill_values = {str(k): float(v) for k, v in stats.get("fill_values", {}).items()}

        if len(self.fill_values) == 0:
            self.fill_values = rebuild_fill_values_from_debug_csv(stats_json.parent, self.base_feature_names)

        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        self.seq_len = int(stats["seq_len"])
        self.input_dim = int(len(self.feature_names))

        if len(self.mean) != self.input_dim or len(self.std) != self.input_dim:
            raise RuntimeError(
                f"stats 維度不一致: len(feature_names)={self.input_dim}, len(mean)={len(self.mean)}, len(std)={len(self.std)}"
            )

        model_type = self.fallback_model_type
        hidden_dim = self.fallback_hidden_dim
        num_layers = self.fallback_num_layers
        dropout = self.fallback_dropout
        self.num_classes = self.fallback_num_classes

        if model_config.exists():
            with open(model_config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_type = str(cfg.get("model_type", model_type))
            hidden_dim = int(cfg.get("hidden_dim", hidden_dim))
            num_layers = int(cfg.get("num_layers", num_layers))
            dropout = float(cfg.get("dropout", dropout))
            self.num_classes = int(cfg.get("num_classes", self.num_classes))
            cfg_input_dim = int(cfg.get("input_dim", self.input_dim))
            cfg_seq_len = int(cfg.get("seq_len", self.seq_len))
            if cfg_input_dim != self.input_dim:
                raise RuntimeError(f"model_config input_dim={cfg_input_dim} 但 stats input_dim={self.input_dim}")
            if cfg_seq_len != self.seq_len:
                raise RuntimeError(f"model_config seq_len={cfg_seq_len} 但 stats seq_len={self.seq_len}")
        else:
            self.get_logger().warn(f"找不到 model_config.json，使用 fallback model config: {model_config}")

        self.model = build_model(
            model_type=model_type,
            seq_len=self.seq_len,
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=self.num_classes,
        ).to(self.device)
        ckpt = safe_torch_load(checkpoint, map_location=self.device)
        self.model.load_state_dict(normalize_state_dict(ckpt))
        self.model.eval()

        self.feature_buffer = deque(maxlen=max(self.seq_len, 1))

        self.get_logger().info(f"loaded checkpoint: {checkpoint}")
        self.get_logger().info(f"loaded stats_json : {stats_json}")
        self.get_logger().info(f"model_config      : {model_config if model_config.exists() else 'fallback'}")
        self.get_logger().info(f"model_type={model_type}, seq_len={self.seq_len}, input_dim={self.input_dim}, num_classes={self.num_classes}")

    # -----------------------------------------------------
    # feature conversion
    # -----------------------------------------------------
    def _stamp_from_dict(self, d: Dict[str, object]) -> float:
        for key in ["timestamp", "stamp", "stamp_sec", "time", "t"]:
            v = to_float(d.get(key, np.nan), np.nan)
            if np.isfinite(v):
                return float(v)
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _feature_dict_to_x(self, feature_dict: Dict[str, object]) -> np.ndarray:
        vals = []
        missing_this_msg = []

        for name in self.feature_names:
            if name.startswith("valid_"):
                base_name = name[len("valid_"):]
                raw_v = to_float(feature_dict.get(base_name, np.nan), np.nan)
                vals.append(1.0 if is_valid_feature_value(base_name, raw_v) else 0.0)
                continue

            raw_v = to_float(feature_dict.get(name, np.nan), np.nan)
            if not is_valid_feature_value(name, raw_v):
                if name not in feature_dict:
                    missing_this_msg.append(name)
                raw_v = float(self.fill_values.get(name, 0.0))
            vals.append(float(raw_v))

        if self.log_missing_features and missing_this_msg:
            # 避免每一幀都洗版：每個欄位只提醒一次
            new_missing = [m for m in missing_this_msg if m not in self.missing_feature_warned]
            if new_missing:
                self.missing_feature_warned.update(new_missing)
                self.get_logger().warn(
                    "runtime feature 缺少以下欄位，已使用 train fill value 補上: " + ", ".join(new_missing)
                )

        x = np.asarray(vals, dtype=np.float32)
        x = (x - self.mean) / self.std
        return x.astype(np.float32)

    def _array_to_feature_dict(self, msg: Float64MultiArray) -> Dict[str, object]:
        data = list(msg.data)
        if len(data) == 0:
            raise ValueError("Float64MultiArray data 是空的")

        if self.array_feature_names:
            names = self.array_feature_names
        else:
            # 預設 array layout: [stamp, base_feature_0, base_feature_1, ...]
            names = self.base_feature_names

        expected = 1 + len(names)
        if len(data) < expected:
            raise ValueError(f"array feature 長度不足: got={len(data)}, expected>={expected}")

        d = {"timestamp": float(data[0])}
        for i, name in enumerate(names):
            d[name] = float(data[i + 1])
        return d

    # -----------------------------------------------------
    # decision and publish
    # -----------------------------------------------------
    def _decision_from_prediction(self, w_pred: float, p_fail: float, p_harmful: float, p_helpful: float) -> Dict[str, object]:
        w_pred = clip01(w_pred)
        p_fail = clip01(p_fail)
        p_harmful = clip01(p_harmful) if np.isfinite(p_harmful) else 0.0
        p_helpful = clip01(p_helpful) if np.isfinite(p_helpful) else 0.0

        # 第一版：observer / soft 建議，不 hard reject
        if self.observer_only:
            visual_weight = float(np.clip(w_pred, self.min_soft_weight, 1.0))
            mode_code = 1 if visual_weight < 0.45 else 0
            mode_name = "degraded" if mode_code == 1 else "normal"

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

        # 下面保留原本 gate 邏輯，但不建議現在啟用
        raw_weight = clip01(w_pred * (1.0 - p_fail) * (1.0 - 0.5 * p_harmful))

        hard_reject = (
            (w_pred < self.hard_reg_min) or
            (p_fail >= self.hard_fail_thr) or
            (p_harmful >= self.hard_harmful_thr)
        )
        soft_pass = (
            (w_pred >= self.tau_reg) and
            (p_fail <= self.tau_fail) and
            (p_harmful <= self.tau_harmful)
        )

        if hard_reject or not soft_pass:
            return {
                "gate_pass": 0,
                "mode_code": 2,
                "mode_name": "visual_safe",
                "visual_weight": float(self.safe_weight),
                "gate_reason": (
                    f"reject: w={w_pred:.3f}, p_fail={p_fail:.3f}, "
                    f"p_harmful={p_harmful:.3f}, p_helpful={p_helpful:.3f}"
                ),
            }

        if raw_weight < self.degraded_weight_thr:
            return {
                "gate_pass": 1,
                "mode_code": 1,
                "mode_name": "degraded",
                "visual_weight": float(max(raw_weight, self.min_degraded_weight)),
                "gate_reason": f"degraded: visual_weight={raw_weight:.3f}",
            }

        return {
            "gate_pass": 1,
            "mode_code": 0,
            "mode_name": "normal",
            "visual_weight": float(raw_weight),
            "gate_reason": f"normal: visual_weight={raw_weight:.3f}",
        }

    def _publish_prediction(self, payload: Dict[str, object]):
        msg = Float64MultiArray()
        msg.data = [
            float(payload["stamp_sec"]),
            float(payload["has_prediction"]),
            float(payload["w_pred"]),
            float(payload["p_fail"]),
            float(payload["p_harmful"]),
            float(payload["p_neutral"]),
            float(payload["p_helpful"]),
            float(payload["gate_pass"]),
            float(payload["mode_code"]),
            float(payload["visual_weight"]),
        ]
        self.publisher.publish(msg)

        if self.publisher_json is not None:
            jmsg = String()
            jmsg.data = json.dumps(payload, ensure_ascii=False)
            self.publisher_json.publish(jmsg)

    def _publish_warmup(self, stamp_sec: float, buffer_size: int):
        if not self.warmup_pass:
            return
        self._publish_prediction({
            "stamp_sec": float(stamp_sec),
            "has_prediction": 0,
            "w_pred": 1.0,
            "p_fail": 0.0,
            "p_harmful": 0.0,
            "p_neutral": 0.0,
            "p_helpful": 1.0,
            "pred_class_id": 2,
            "pred_class": "helpful",
            "gate_pass": 1,
            "mode_code": -1,
            "mode_name": "warmup",
            "visual_weight": 1.0,
            "gate_reason": f"warmup: buffer={buffer_size}/{self.seq_len}",
            "seq_len": int(self.seq_len),
            "input_dim": int(self.input_dim),
        })

    @torch.no_grad()
    def _run_model_once(self, stamp_sec: float):
        x_seq = np.stack(list(self.feature_buffer), axis=0).astype(np.float32)
        xb = torch.from_numpy(x_seq.reshape(1, self.seq_len, self.input_dim)).float().to(self.device)
        outputs = self.model(xb, return_dict=True)

        w_pred = float(outputs["reg"].detach().cpu().numpy().reshape(-1)[0])
        p_fail = sigmoid_float(float(outputs["fail_logits"].detach().cpu().numpy().reshape(-1)[0]))
        cls_logits = outputs["cls_logits"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        probs = softmax_np(cls_logits)

        p_harmful = float(probs[0]) if self.num_classes >= 1 else 0.0
        p_neutral = float(probs[1]) if self.num_classes >= 2 else 0.0
        p_helpful = float(probs[2]) if self.num_classes >= 3 else 0.0
        pred_class_id = int(np.argmax(probs)) if probs.size > 0 else -1
        pred_class = CLASS_ID_TO_NAME.get(pred_class_id, f"class_{pred_class_id}")

        decision = self._decision_from_prediction(w_pred, p_fail, p_harmful, p_helpful)
        payload = {
            "stamp_sec": float(stamp_sec),
            "has_prediction": 1,
            "w_pred": clip01(w_pred),
            "p_fail": clip01(p_fail),
            "p_harmful": clip01(p_harmful),
            "p_neutral": clip01(p_neutral),
            "p_helpful": clip01(p_helpful),
            "pred_class_id": int(pred_class_id),
            "pred_class": str(pred_class),
            "gate_pass": int(decision["gate_pass"]),
            "mode_code": int(decision["mode_code"]),
            "mode_name": str(decision["mode_name"]),
            "visual_weight": float(decision["visual_weight"]),
            "gate_reason": str(decision["gate_reason"]),
            "seq_len": int(self.seq_len),
            "input_dim": int(self.input_dim),
            "num_received": int(self.num_received),
            "num_predicted": int(self.num_predicted + 1),
        }
        self.num_predicted += 1
        self._publish_prediction(payload)

    def _handle_feature_dict(self, feature_dict: Dict[str, object]):
        stamp_sec = self._stamp_from_dict(feature_dict)
        if self.drop_old_stamp and stamp_sec <= self.last_stamp:
            return
        self.last_stamp = stamp_sec
        self.num_received += 1

        x = self._feature_dict_to_x(feature_dict)
        self.feature_buffer.append(x)

        if len(self.feature_buffer) < self.seq_len:
            self._publish_warmup(stamp_sec, len(self.feature_buffer))
            return

        self._run_model_once(stamp_sec)

    # -----------------------------------------------------
    # callbacks
    # -----------------------------------------------------
    def feature_json_callback(self, msg: String):
        try:
            feature_dict = json.loads(msg.data)
            if not isinstance(feature_dict, dict):
                raise ValueError("JSON payload 必須是 object/dict")
            self._handle_feature_dict(feature_dict)
        except Exception as e:
            self.get_logger().warn(f"解析 JSON feature 失敗: {e}")

    def feature_array_callback(self, msg: Float64MultiArray):
        try:
            feature_dict = self._array_to_feature_dict(msg)
            self._handle_feature_dict(feature_dict)
        except Exception as e:
            self.get_logger().warn(f"解析 array feature 失敗: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ReliabilityInferNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
