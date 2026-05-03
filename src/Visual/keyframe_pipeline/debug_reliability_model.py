#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_reliability_model.py

分析 reliability model 的預測品質，不改模型、不重新訓練。
輸出：
  - predictions_<split>.csv
  - sequence_metrics_<split>.csv
  - reg_bins_<split>.csv
  - fail_calibration_<split>.csv
  - class_confusion_<split>.csv
  - debug_summary.json

建議放在：
  ~/vins_ws/src/Visual/keyframe_pipeline/debug_reliability_model.py
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

CLASS_ID_TO_NAME = {0: "harmful", 1: "neutral", 2: "helpful"}


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(x, default=-1):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_true - y_pred) ** 2))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def corrcoef_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    if float(np.std(y_true)) < 1e-12 or float(np.std(y_pred)) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def binary_metrics_np(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    valid = np.isfinite(y_prob) & (y_true >= 0) & (y_true <= 1)
    if int(valid.sum()) == 0:
        return {
            "fail_num_valid": 0,
            "fail_pos_rate": 0.0,
            "fail_pred_pos_rate": 0.0,
            "fail_acc": 0.0,
            "fail_precision": 0.0,
            "fail_recall": 0.0,
            "fail_f1": 0.0,
            "fail_brier": 0.0,
        }

    yt = y_true[valid].astype(np.int64)
    yp_prob = y_prob[valid].astype(np.float64)
    yp = (yp_prob >= threshold).astype(np.int64)

    tp = float(np.sum((yp == 1) & (yt == 1)))
    fp = float(np.sum((yp == 1) & (yt == 0)))
    tn = float(np.sum((yp == 0) & (yt == 0)))
    fn = float(np.sum((yp == 0) & (yt == 1)))

    acc = (tp + tn) / max(tp + fp + tn + fn, 1.0)
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    brier = float(np.mean((yp_prob - yt.astype(np.float64)) ** 2))

    return {
        "fail_num_valid": int(valid.sum()),
        "fail_pos_rate": float(np.mean(yt)),
        "fail_pred_pos_rate": float(np.mean(yp)),
        "fail_acc": float(acc),
        "fail_precision": float(precision),
        "fail_recall": float(recall),
        "fail_f1": float(f1),
        "fail_brier": float(brier),
    }


def multiclass_metrics_np(y_true: np.ndarray, prob: np.ndarray, num_classes: int = 3) -> Dict[str, float]:
    valid = (y_true >= 0) & (y_true < num_classes) & np.isfinite(prob).all(axis=1)
    if int(valid.sum()) == 0:
        return {
            "cls_num_valid": 0,
            "cls_acc": 0.0,
            "cls_macro_f1": 0.0,
            "cls_pred_harmful_rate": 0.0,
            "cls_pred_neutral_rate": 0.0,
            "cls_pred_helpful_rate": 0.0,
        }

    yt = y_true[valid].astype(np.int64)
    pred = np.argmax(prob[valid], axis=1).astype(np.int64)

    acc = float(np.mean(pred == yt))

    f1s = []
    for c in range(num_classes):
        tp = float(np.sum((pred == c) & (yt == c)))
        fp = float(np.sum((pred == c) & (yt != c)))
        fn = float(np.sum((pred != c) & (yt == c)))
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(float(f1))

    return {
        "cls_num_valid": int(valid.sum()),
        "cls_acc": float(acc),
        "cls_macro_f1": float(np.mean(f1s)),
        "cls_pred_harmful_rate": float(np.mean(pred == 0)),
        "cls_pred_neutral_rate": float(np.mean(pred == 1)),
        "cls_pred_helpful_rate": float(np.mean(pred == 2)),
    }


def regression_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(valid.sum()) == 0:
        return {
            "reg_num_valid": 0,
            "reg_mse": 0.0,
            "reg_mae": 0.0,
            "reg_corr": 0.0,
            "label_reg_mean": 0.0,
            "pred_reg_mean": 0.0,
            "label_reg_std": 0.0,
            "pred_reg_std": 0.0,
        }

    yt = y_true[valid].astype(np.float64)
    yp = y_pred[valid].astype(np.float64)
    return {
        "reg_num_valid": int(valid.sum()),
        "reg_mse": mse_np(yt, yp),
        "reg_mae": mae_np(yt, yp),
        "reg_corr": corrcoef_np(yt, yp),
        "label_reg_mean": float(np.mean(yt)),
        "pred_reg_mean": float(np.mean(yp)),
        "label_reg_std": float(np.std(yt)),
        "pred_reg_std": float(np.std(yp)),
    }


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, object]], preferred_fields: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if preferred_fields is None:
        preferred_fields = []

    field_set = set()
    for r in rows:
        field_set.update(r.keys())

    fields = []
    for k in preferred_fields:
        if k in field_set:
            fields.append(k)
            field_set.remove(k)
    fields.extend(sorted(field_set))

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def load_npz_split(dataset_dir: Path, split: str) -> Dict[str, np.ndarray]:
    path = dataset_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"找不到 split npz: {path}")

    data = np.load(path, allow_pickle=True)
    out = {
        "X": data["X"].astype(np.float32),
        "pair_ids": data["pair_ids"].astype(np.int64),
        "feature_names": [str(x) for x in data["feature_names"].tolist()],
        "seq_len": int(data["seq_len"][0]),
    }

    if "y_reg" in data:
        out["y_reg"] = data["y_reg"].astype(np.float32)
    elif "y" in data:
        out["y_reg"] = data["y"].astype(np.float32)
    else:
        raise KeyError(f"{path} 缺少 y_reg 或 y")

    out["y_fail"] = data["y_fail"].astype(np.int64) if "y_fail" in data else np.full((len(out["X"]),), -1, dtype=np.int64)
    out["y_cls"] = data["y_cls"].astype(np.int64) if "y_cls" in data else np.full((len(out["X"]),), -1, dtype=np.int64)
    return out


def load_debug_map(dataset_dir: Path) -> Dict[int, Dict[str, str]]:
    debug_csv = dataset_dir / "feature_label_debug.csv"
    rows = read_csv_rows(debug_csv)
    pair_to_row = {}
    for r in rows:
        pid = safe_int(r.get("pair_id", -1), -1)
        if pid >= 0:
            pair_to_row[pid] = r
    return pair_to_row


def import_train_module(script_dir: Path):
    sys.path.insert(0, str(script_dir))
    try:
        import train_reliability_model as train_mod
    except Exception as e:
        raise RuntimeError(
            "無法 import train_reliability_model.py。請把 debug_reliability_model.py "
            "放在 ~/vins_ws/src/Visual/keyframe_pipeline，或用 --script_dir 指到該資料夾。"
        ) from e
    return train_mod


def load_model_from_dir(model_dir: Path, dataset_info: Dict[str, np.ndarray], script_dir: Path, device: torch.device):
    config_path = model_dir / "model_config.json"
    best_path = model_dir / "best.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"找不到 model_config.json: {config_path}")
    if not best_path.exists():
        raise FileNotFoundError(f"找不到 best.pt: {best_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_mod = import_train_module(script_dir)

    model = train_mod.build_model(
        model_type=cfg.get("model_type", "gru"),
        seq_len=int(cfg.get("seq_len", dataset_info["seq_len"])),
        input_dim=int(cfg.get("input_dim", dataset_info["X"].shape[-1])),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
        num_classes=int(cfg.get("num_classes", 3)),
    ).to(device)

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def extract_outputs(outputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(outputs, dict):
        if "reg" in outputs:
            reg = outputs["reg"]
        elif "pred_reg" in outputs:
            reg = outputs["pred_reg"]
        else:
            raise KeyError(f"model output dict 找不到 reg key，可用 keys={list(outputs.keys())}")

        fail_prob_t = None
        if "fail_prob" in outputs:
            fail_prob_t = outputs["fail_prob"]
        elif "fail" in outputs:
            fail_prob_t = outputs["fail"]
        elif "fail_logits" in outputs:
            x = outputs["fail_logits"]
            if x.ndim == 2 and x.shape[-1] == 2:
                fail_prob_t = torch.softmax(x, dim=-1)[:, 1]
            else:
                fail_prob_t = torch.sigmoid(x.squeeze(-1))
        elif "fail_logit" in outputs:
            fail_prob_t = torch.sigmoid(outputs["fail_logit"].squeeze(-1))

        cls_prob_t = None
        if "cls_prob" in outputs:
            cls_prob_t = outputs["cls_prob"]
        elif "cls_logits" in outputs:
            cls_prob_t = torch.softmax(outputs["cls_logits"], dim=-1)
        elif "class_logits" in outputs:
            cls_prob_t = torch.softmax(outputs["class_logits"], dim=-1)
    else:
        reg = outputs
        fail_prob_t = None
        cls_prob_t = None

    pred_reg = reg.detach().cpu().numpy().reshape(-1).astype(np.float32)
    fail_prob = np.full((len(pred_reg),), np.nan, dtype=np.float32) if fail_prob_t is None else fail_prob_t.detach().cpu().numpy().reshape(-1).astype(np.float32)

    if cls_prob_t is None:
        cls_prob = np.full((len(pred_reg), 3), np.nan, dtype=np.float32)
    else:
        cls_prob = cls_prob_t.detach().cpu().numpy().astype(np.float32)
        if cls_prob.ndim == 1:
            cls_prob = np.stack([1.0 - cls_prob, cls_prob, np.zeros_like(cls_prob)], axis=1)
    return pred_reg, fail_prob, cls_prob


@torch.no_grad()
def run_inference(model, split_data: Dict[str, np.ndarray], device: torch.device, batch_size: int = 1024):
    X = split_data["X"]
    n = len(X)
    pred_regs, fail_probs, cls_probs = [], [], []

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.from_numpy(X[s:e]).float().to(device)
        outputs = model(xb, return_dict=True)
        pred_reg, fail_prob, cls_prob = extract_outputs(outputs)
        pred_regs.append(pred_reg)
        fail_probs.append(fail_prob)
        cls_probs.append(cls_prob)

    return {
        "pred_reg": np.concatenate(pred_regs, axis=0),
        "fail_prob": np.concatenate(fail_probs, axis=0),
        "cls_prob": np.concatenate(cls_probs, axis=0),
    }


def build_prediction_rows(split: str, split_data: Dict[str, np.ndarray], pred: Dict[str, np.ndarray], debug_map: Dict[int, Dict[str, str]]) -> List[Dict[str, object]]:
    rows = []
    y_reg, y_fail, y_cls = split_data["y_reg"], split_data["y_fail"], split_data["y_cls"]
    pair_ids = split_data["pair_ids"]
    pred_reg, fail_prob, cls_prob = pred["pred_reg"], pred["fail_prob"], pred["cls_prob"]
    cls_pred = np.argmax(cls_prob, axis=1) if np.isfinite(cls_prob).any() else np.full(len(pair_ids), -1)

    for i, pid in enumerate(pair_ids):
        info = debug_map.get(int(pid), {})
        seq = info.get("sequence_name", "") or info.get("source_sequence", "") or info.get("run_id", "") or "unknown"
        rows.append({
            "split": split,
            "sample_index": int(i),
            "pair_id": int(pid),
            "sequence_name": seq,
            "timestamp": info.get("timestamp", ""),
            "update_id": info.get("update_id", ""),
            "label_reg": float(y_reg[i]),
            "pred_reg": float(pred_reg[i]),
            "abs_reg_error": float(abs(float(y_reg[i]) - float(pred_reg[i]))),
            "y_fail": int(y_fail[i]),
            "fail_prob": float(fail_prob[i]) if np.isfinite(fail_prob[i]) else "",
            "fail_pred": int(fail_prob[i] >= 0.5) if np.isfinite(fail_prob[i]) else -1,
            "y_cls": int(y_cls[i]),
            "y_cls_name": CLASS_ID_TO_NAME.get(int(y_cls[i]), ""),
            "cls_pred": int(cls_pred[i]) if int(cls_pred[i]) in CLASS_ID_TO_NAME else -1,
            "cls_pred_name": CLASS_ID_TO_NAME.get(int(cls_pred[i]), ""),
            "cls_prob_harmful": float(cls_prob[i, 0]) if np.isfinite(cls_prob[i, 0]) else "",
            "cls_prob_neutral": float(cls_prob[i, 1]) if np.isfinite(cls_prob[i, 1]) else "",
            "cls_prob_helpful": float(cls_prob[i, 2]) if np.isfinite(cls_prob[i, 2]) else "",
            "future_drift_trans_m": info.get("future_drift_trans_m", ""),
            "future_drift_rot_deg": info.get("future_drift_rot_deg", ""),
            "future_drift_risk": info.get("future_drift_risk", ""),
            "future_proxy_risk": info.get("future_proxy_risk", ""),
            "class_name": info.get("class_name", ""),
            "label_source": info.get("label_source", ""),
        })
    return rows


def metrics_for_rows(rows: List[Dict[str, object]]) -> Dict[str, float]:
    y_reg = np.array([safe_float(r.get("label_reg", np.nan), np.nan) for r in rows], dtype=np.float64)
    pred_reg = np.array([safe_float(r.get("pred_reg", np.nan), np.nan) for r in rows], dtype=np.float64)
    y_fail = np.array([safe_int(r.get("y_fail", -1), -1) for r in rows], dtype=np.int64)
    fail_prob = np.array([safe_float(r.get("fail_prob", np.nan), np.nan) for r in rows], dtype=np.float64)
    y_cls = np.array([safe_int(r.get("y_cls", -1), -1) for r in rows], dtype=np.int64)
    cls_prob = np.zeros((len(rows), 3), dtype=np.float64)
    for i, r in enumerate(rows):
        cls_prob[i, 0] = safe_float(r.get("cls_prob_harmful", np.nan), np.nan)
        cls_prob[i, 1] = safe_float(r.get("cls_prob_neutral", np.nan), np.nan)
        cls_prob[i, 2] = safe_float(r.get("cls_prob_helpful", np.nan), np.nan)
    out = {}
    out.update(regression_metrics_np(y_reg, pred_reg))
    out.update(binary_metrics_np(y_fail, fail_prob))
    out.update(multiclass_metrics_np(y_cls, cls_prob))
    out["num_samples"] = int(len(rows))
    return out


def build_sequence_metrics(pred_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_seq: Dict[str, List[Dict[str, object]]] = {}
    for r in pred_rows:
        by_seq.setdefault(str(r.get("sequence_name", "unknown")), []).append(r)
    out = []
    for seq, rows in sorted(by_seq.items()):
        m = metrics_for_rows(rows)
        m["sequence_name"] = seq
        out.append(m)
    out.sort(key=lambda r: (-float(r.get("reg_mse", 0.0)), float(r.get("reg_corr", 0.0))))
    return out


def build_reg_bins(pred_rows: List[Dict[str, object]], num_bins: int = 10) -> List[Dict[str, object]]:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    out = []
    label = np.array([safe_float(r.get("label_reg", np.nan), np.nan) for r in pred_rows], dtype=np.float64)
    pred = np.array([safe_float(r.get("pred_reg", np.nan), np.nan) for r in pred_rows], dtype=np.float64)
    y_fail = np.array([safe_int(r.get("y_fail", -1), -1) for r in pred_rows], dtype=np.int64)
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred >= lo) & (pred <= hi) if i == num_bins - 1 else (pred >= lo) & (pred < hi)
        n = int(mask.sum())
        row = {"bin_id": i, "pred_reg_lo": float(lo), "pred_reg_hi": float(hi), "count": n}
        if n > 0:
            valid_fail = mask & (y_fail >= 0)
            row.update({
                "label_reg_mean": float(np.nanmean(label[mask])),
                "pred_reg_mean": float(np.nanmean(pred[mask])),
                "abs_error_mean": float(np.nanmean(np.abs(label[mask] - pred[mask]))),
                "fail_rate": float(np.mean(y_fail[valid_fail])) if int(valid_fail.sum()) > 0 else "",
            })
        out.append(row)
    return out


def build_fail_calibration(pred_rows: List[Dict[str, object]], num_bins: int = 10) -> List[Dict[str, object]]:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    out = []
    y_fail = np.array([safe_int(r.get("y_fail", -1), -1) for r in pred_rows], dtype=np.int64)
    prob = np.array([safe_float(r.get("fail_prob", np.nan), np.nan) for r in pred_rows], dtype=np.float64)
    total_valid = int(np.sum(np.isfinite(prob) & (y_fail >= 0)))
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = np.isfinite(prob) & (y_fail >= 0) & ((prob >= lo) & (prob <= hi) if i == num_bins - 1 else (prob >= lo) & (prob < hi))
        n = int(mask.sum())
        row = {"bin_id": i, "prob_lo": float(lo), "prob_hi": float(hi), "count": n}
        if n > 0:
            mean_prob = float(np.mean(prob[mask]))
            fail_rate = float(np.mean(y_fail[mask]))
            gap = abs(mean_prob - fail_rate)
            ece += (n / max(total_valid, 1)) * gap
            row.update({"mean_fail_prob": mean_prob, "actual_fail_rate": fail_rate, "calibration_gap": float(gap)})
        out.append(row)
    if out:
        out[0]["ECE"] = float(ece)
        out[0]["total_valid"] = int(total_valid)
    return out


def build_class_confusion(pred_rows: List[Dict[str, object]], num_classes: int = 3) -> List[Dict[str, object]]:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for r in pred_rows:
        yt = safe_int(r.get("y_cls", -1), -1)
        yp = safe_int(r.get("cls_pred", -1), -1)
        if 0 <= yt < num_classes and 0 <= yp < num_classes:
            mat[yt, yp] += 1
    rows = []
    for yt in range(num_classes):
        row = {"true_class_id": yt, "true_class_name": CLASS_ID_TO_NAME.get(yt, str(yt))}
        for yp in range(num_classes):
            row[f"pred_{CLASS_ID_TO_NAME.get(yp, str(yp))}"] = int(mat[yt, yp])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--script_dir", default="")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", default="")
    parser.add_argument("--num_bins", type=int, default=10)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else model_dir / "debug_model"
    script_dir = Path(args.script_dir).expanduser().resolve() if args.script_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    first_split_data = load_npz_split(dataset_dir, args.splits[0])
    model, cfg = load_model_from_dir(model_dir, first_split_data, script_dir, device)
    debug_map = load_debug_map(dataset_dir)

    print("========== Debug Reliability Model ==========")
    print(f"dataset_dir: {dataset_dir}")
    print(f"model_dir  : {model_dir}")
    print(f"out_dir    : {out_dir}")
    print(f"script_dir : {script_dir}")
    print(f"device     : {device}")
    print(f"splits     : {args.splits}")

    summary = {"dataset_dir": str(dataset_dir), "model_dir": str(model_dir), "out_dir": str(out_dir), "device": str(device), "model_config": cfg, "splits": {}}

    for split in args.splits:
        print(f"\n========== Split: {split} ==========")
        data = load_npz_split(dataset_dir, split)
        pred = run_inference(model, data, device=device, batch_size=args.batch_size)
        pred_rows = build_prediction_rows(split, data, pred, debug_map)

        pred_csv = out_dir / f"predictions_{split}.csv"
        seq_csv = out_dir / f"sequence_metrics_{split}.csv"
        reg_bins_csv = out_dir / f"reg_bins_{split}.csv"
        fail_cal_csv = out_dir / f"fail_calibration_{split}.csv"
        cls_conf_csv = out_dir / f"class_confusion_{split}.csv"

        write_csv_rows(pred_csv, pred_rows, ["split", "sample_index", "pair_id", "sequence_name", "timestamp", "update_id", "label_reg", "pred_reg", "abs_reg_error", "y_fail", "fail_prob", "fail_pred", "y_cls", "y_cls_name", "cls_pred", "cls_pred_name", "cls_prob_harmful", "cls_prob_neutral", "cls_prob_helpful", "future_drift_trans_m", "future_drift_rot_deg", "future_drift_risk"])
        write_csv_rows(seq_csv, build_sequence_metrics(pred_rows), ["sequence_name", "num_samples", "reg_mse", "reg_mae", "reg_corr", "label_reg_mean", "pred_reg_mean", "label_reg_std", "pred_reg_std", "fail_pos_rate", "fail_pred_pos_rate", "fail_acc", "fail_precision", "fail_recall", "fail_f1", "fail_brier", "cls_acc", "cls_macro_f1", "cls_pred_harmful_rate", "cls_pred_neutral_rate", "cls_pred_helpful_rate"])
        write_csv_rows(reg_bins_csv, build_reg_bins(pred_rows, num_bins=args.num_bins))
        write_csv_rows(fail_cal_csv, build_fail_calibration(pred_rows, num_bins=args.num_bins))
        write_csv_rows(cls_conf_csv, build_class_confusion(pred_rows))

        split_metrics = metrics_for_rows(pred_rows)
        summary["splits"][split] = split_metrics
        print(json.dumps(split_metrics, ensure_ascii=False, indent=2))
        print(f"[WRITE] {pred_csv}")
        print(f"[WRITE] {seq_csv}")
        print(f"[WRITE] {reg_bins_csv}")
        print(f"[WRITE] {fail_cal_csv}")
        print(f"[WRITE] {cls_conf_csv}")

    summary_json = out_dir / "debug_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n========== Done ==========")
    print(f"summary: {summary_json}")


if __name__ == "__main__":
    main()
