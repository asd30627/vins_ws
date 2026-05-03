#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_reliability_model.py

訓練 future-drift-aware reliability model。

輸入 dataset：
  build_reliability_dataset.py 產生的 train.npz / val.npz / test.npz

模型輸出：
  reg  : visual reliability score, 0~1，越大越可信，可用於 soft weighting
  fail : future drift failure probability，可用於 predictive visual-safe mode
  cls  : harmful / neutral / helpful，可用於 hard gating 的 debug/輔助指標
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_npz(path: Path):
    data = np.load(path, allow_pickle=True)

    out = {
        "X": data["X"].astype(np.float32),
        "pair_ids": data["pair_ids"].astype(np.int32),
        "feature_names": [str(x) for x in data["feature_names"].tolist()],
        "seq_len": int(data["seq_len"][0]),
    }

    if "y_reg" in data:
        out["y_reg"] = data["y_reg"].astype(np.float32)
    elif "y" in data:
        out["y_reg"] = data["y"].astype(np.float32)
    else:
        raise KeyError(f"{path} 缺少 y_reg 或 y")

    if "y_cls" in data:
        out["y_cls"] = data["y_cls"].astype(np.int64)
    else:
        out["y_cls"] = np.full((len(out["X"]),), -1, dtype=np.int64)

    if "y_fail" in data:
        out["y_fail"] = data["y_fail"].astype(np.int64)
    else:
        out["y_fail"] = np.full((len(out["X"]),), -1, dtype=np.int64)

    for k in ["y_drift_trans", "y_drift_rot", "y_drift_risk", "y_proxy_risk"]:
        if k in data:
            out[k] = data[k].astype(np.float32)

    return out


def mse_np(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def corrcoef_np(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return float(a / b)


def macro_f1_from_classes(y_true: np.ndarray, y_pred: np.ndarray, classes) -> float:
    f1s = []
    for c in classes:
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision + recall < 1e-12:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(float(f1))
    return float(np.mean(f1s)) if f1s else 0.0


def classification_metrics_np(y_true: np.ndarray, logits: np.ndarray, num_classes: int = 3) -> Dict[str, float]:
    valid = (y_true >= 0) & (y_true < num_classes)
    if valid.sum() == 0:
        return {"cls_acc": 0.0, "cls_macro_f1": 0.0, "cls_num_valid": 0}

    yv = y_true[valid].astype(np.int64)
    pred = np.argmax(logits[valid], axis=1).astype(np.int64)

    return {
        "cls_acc": float((pred == yv).mean()),
        "cls_macro_f1": macro_f1_from_classes(yv, pred, list(range(num_classes))),
        "cls_num_valid": int(valid.sum()),
    }


def binary_metrics_np(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    valid = (y_true == 0) | (y_true == 1)
    if valid.sum() == 0:
        return {"fail_acc": 0.0, "fail_f1": 0.0, "fail_num_valid": 0, "fail_pos_rate": 0.0}

    yv = y_true[valid].astype(np.int64)
    prob = 1.0 / (1.0 + np.exp(-logits[valid].reshape(-1)))
    pred = (prob >= 0.5).astype(np.int64)

    return {
        "fail_acc": float((pred == yv).mean()),
        "fail_f1": macro_f1_from_classes(yv, pred, [0, 1]),
        "fail_num_valid": int(valid.sum()),
        "fail_pos_rate": float(yv.mean()),
    }


def compute_class_weights(y_cls: np.ndarray, num_classes: int = 3) -> Optional[np.ndarray]:
    valid = (y_cls >= 0) & (y_cls < num_classes)
    if valid.sum() == 0:
        return None

    counts = np.bincount(y_cls[valid], minlength=num_classes).astype(np.float64)
    counts = np.where(counts <= 0.0, 1.0, counts)
    inv = 1.0 / counts
    weights = inv / max(inv.mean(), 1e-12)
    return weights.astype(np.float32)


def compute_binary_pos_weight(y_fail: np.ndarray) -> Optional[float]:
    valid = (y_fail == 0) | (y_fail == 1)
    if valid.sum() == 0:
        return None
    pos = float(np.sum(y_fail[valid] == 1))
    neg = float(np.sum(y_fail[valid] == 0))
    if pos <= 0.0:
        return None
    return float(max(neg / pos, 1e-3))


# =========================================================
# Dataset
# =========================================================

class SequenceDataset(Dataset):
    def __init__(self, X, y_reg, y_cls, y_fail, pair_ids):
        self.X = torch.from_numpy(X).float()
        self.y_reg = torch.from_numpy(y_reg).float()
        self.y_cls = torch.from_numpy(y_cls).long()
        self.y_fail = torch.from_numpy(y_fail).long()
        self.pair_ids = torch.from_numpy(pair_ids).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_cls[idx], self.y_fail[idx], self.pair_ids[idx]


# =========================================================
# encoders
# =========================================================

class MLPEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        flat_dim = seq_len * input_dim
        mid_dim = max(hidden_dim // 2, 16)

        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.net(x)


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        _, h = self.gru(x)
        return self.proj(h[-1])


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class TCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.block1 = TemporalConvBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.block2 = TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
        self.out_dim = hidden_dim

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


class MultiTaskReliabilityModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.fail_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, return_dict: bool = False):
        feat = self.encoder(x)
        reg = self.reg_head(feat).squeeze(-1)
        fail_logits = self.fail_head(feat).squeeze(-1)
        cls_logits = self.cls_head(feat)

        if return_dict:
            return {
                "reg": reg,
                "fail_logits": fail_logits,
                "cls_logits": cls_logits,
                "feat": feat,
            }
        return reg


def build_model(model_type, seq_len, input_dim, hidden_dim, num_layers, dropout, num_classes: int = 3):
    model_type = str(model_type).strip().lower()

    if model_type == "mlp":
        encoder = MLPEncoder(seq_len=seq_len, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == "gru":
        encoder = GRUEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    elif model_type == "tcn":
        encoder = TCNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"未知 model_type: {model_type}")

    return MultiTaskReliabilityModel(
        encoder=encoder,
        hidden_dim=encoder.out_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


# =========================================================
# Trainer
# =========================================================

class ReliabilityModelTrainer:
    def __init__(
        self,
        dataset_dir,
        out_dir="",
        model_type="gru",
        epochs=80,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=64,
        num_layers=1,
        dropout=0.10,
        seed=42,
        device=None,
        lambda_reg=1.0,
        lambda_cls=0.5,
        lambda_fail=1.0,
        label_smoothing=0.0,
        grad_clip_norm=1.0,
        patience=12,
        min_delta=1e-5,
        num_workers=0,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.model_type = str(model_type)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.lambda_reg = float(lambda_reg)
        self.lambda_cls = float(lambda_cls)
        self.lambda_fail = float(lambda_fail)
        self.label_smoothing = float(label_smoothing)
        self.grad_clip_norm = float(grad_clip_norm)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.num_workers = int(num_workers)

        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.dataset_dir / "model_runs" / self.model_type
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.best_path = self.out_dir / "best.pt"
        self.last_path = self.out_dir / "last.pt"
        self.best_info_json = self.out_dir / "best_info.json"
        self.history_json = self.out_dir / "history.json"
        self.history_csv = self.out_dir / "history.csv"
        self.model_config_json = self.out_dir / "model_config.json"

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.seq_len = None
        self.input_dim = None
        self.feature_names = None
        self.num_classes = 3

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.reg_criterion = nn.MSELoss()
        self.cls_criterion = None
        self.fail_criterion = None

        self.history = []
        self.best_val_joint = float("inf")
        self.best_epoch = -1
        self.best_metrics = None
        self.epochs_no_improve = 0

    def load_data(self):
        train_path = self.dataset_dir / "train.npz"
        val_path = self.dataset_dir / "val.npz"
        test_path = self.dataset_dir / "test.npz"

        for p in [train_path, val_path, test_path]:
            if not p.exists():
                raise FileNotFoundError(f"找不到 {p}")

        self.train_data = load_split_npz(train_path)
        self.val_data = load_split_npz(val_path)
        self.test_data = load_split_npz(test_path)

        self.seq_len = int(self.train_data["seq_len"])
        self.input_dim = int(self.train_data["X"].shape[-1])
        self.feature_names = list(self.train_data["feature_names"])

        self.train_loader = DataLoader(
            SequenceDataset(
                self.train_data["X"],
                self.train_data["y_reg"],
                self.train_data["y_cls"],
                self.train_data["y_fail"],
                self.train_data["pair_ids"],
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        self.val_loader = DataLoader(
            SequenceDataset(
                self.val_data["X"],
                self.val_data["y_reg"],
                self.val_data["y_cls"],
                self.val_data["y_fail"],
                self.val_data["pair_ids"],
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        self.test_loader = DataLoader(
            SequenceDataset(
                self.test_data["X"],
                self.test_data["y_reg"],
                self.test_data["y_cls"],
                self.test_data["y_fail"],
                self.test_data["pair_ids"],
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def build_model_and_optimizer(self):
        self.model = build_model(
            model_type=self.model_type,
            seq_len=self.seq_len,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=self.num_classes,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=4,
        )

        class_weights = compute_class_weights(self.train_data["y_cls"], num_classes=self.num_classes)
        if class_weights is None:
            self.cls_criterion = None
        else:
            self.cls_criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(class_weights).float().to(self.device),
                ignore_index=-1,
                label_smoothing=self.label_smoothing,
            )

        pos_weight = compute_binary_pos_weight(self.train_data["y_fail"])
        if pos_weight is None:
            self.fail_criterion = None
        else:
            self.fail_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
            )

    def _compute_losses(self, outputs, y_reg, y_cls, y_fail):
        pred_reg = outputs["reg"]
        cls_logits = outputs["cls_logits"]
        fail_logits = outputs["fail_logits"]

        reg_loss = self.reg_criterion(pred_reg, y_reg)

        if self.cls_criterion is None:
            cls_loss = torch.tensor(0.0, device=pred_reg.device)
        else:
            valid_cls = (y_cls >= 0) & (y_cls < self.num_classes)
            if int(valid_cls.sum().item()) == 0:
                cls_loss = torch.tensor(0.0, device=pred_reg.device)
            else:
                cls_loss = self.cls_criterion(cls_logits, y_cls)

        if self.fail_criterion is None:
            fail_loss = torch.tensor(0.0, device=pred_reg.device)
        else:
            valid_fail = (y_fail == 0) | (y_fail == 1)
            if int(valid_fail.sum().item()) == 0:
                fail_loss = torch.tensor(0.0, device=pred_reg.device)
            else:
                fail_loss = self.fail_criterion(fail_logits[valid_fail], y_fail[valid_fail].float())

        joint_loss = self.lambda_reg * reg_loss + self.lambda_cls * cls_loss + self.lambda_fail * fail_loss
        return reg_loss, cls_loss, fail_loss, joint_loss

    def _run_epoch(self, loader, split_name: str, epoch_idx: int, train: bool):
        self.model.train(mode=train)

        total_reg_loss = 0.0
        total_cls_loss = 0.0
        total_fail_loss = 0.0
        total_joint_loss = 0.0
        total_samples = 0

        all_y_reg = []
        all_pred_reg = []
        all_y_cls = []
        all_cls_logits = []
        all_y_fail = []
        all_fail_logits = []

        for xb, y_reg, y_cls, y_fail, _ in loader:
            xb = xb.to(self.device, non_blocking=True)
            y_reg = y_reg.to(self.device, non_blocking=True)
            y_cls = y_cls.to(self.device, non_blocking=True)
            y_fail = y_fail.to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(xb, return_dict=True)
            reg_loss, cls_loss, fail_loss, joint_loss = self._compute_losses(outputs, y_reg, y_cls, y_fail)

            if train:
                joint_loss.backward()
                if self.grad_clip_norm > 0.0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            bs = int(xb.shape[0])
            total_samples += bs
            total_reg_loss += float(reg_loss.item()) * bs
            total_cls_loss += float(cls_loss.item()) * bs
            total_fail_loss += float(fail_loss.item()) * bs
            total_joint_loss += float(joint_loss.item()) * bs

            all_y_reg.append(y_reg.detach().cpu().numpy())
            all_pred_reg.append(outputs["reg"].detach().cpu().numpy())
            all_y_cls.append(y_cls.detach().cpu().numpy())
            all_cls_logits.append(outputs["cls_logits"].detach().cpu().numpy())
            all_y_fail.append(y_fail.detach().cpu().numpy())
            all_fail_logits.append(outputs["fail_logits"].detach().cpu().numpy())

        y_reg_np = np.concatenate(all_y_reg, axis=0).astype(np.float32)
        pred_reg_np = np.concatenate(all_pred_reg, axis=0).astype(np.float32)
        y_cls_np = np.concatenate(all_y_cls, axis=0).astype(np.int64)
        cls_logits_np = np.concatenate(all_cls_logits, axis=0).astype(np.float32)
        y_fail_np = np.concatenate(all_y_fail, axis=0).astype(np.int64)
        fail_logits_np = np.concatenate(all_fail_logits, axis=0).astype(np.float32)

        cls_metrics = classification_metrics_np(y_cls_np, cls_logits_np, num_classes=self.num_classes)
        fail_metrics = binary_metrics_np(y_fail_np, fail_logits_np)

        prefix = split_name.strip().lower()
        metrics = {
            "epoch": int(epoch_idx),
            f"{prefix}_reg_loss": float(total_reg_loss / max(total_samples, 1)),
            f"{prefix}_cls_loss": float(total_cls_loss / max(total_samples, 1)),
            f"{prefix}_fail_loss": float(total_fail_loss / max(total_samples, 1)),
            f"{prefix}_joint_loss": float(total_joint_loss / max(total_samples, 1)),

            f"{prefix}_mse": mse_np(y_reg_np, pred_reg_np),
            f"{prefix}_mae": mae_np(y_reg_np, pred_reg_np),
            f"{prefix}_corr": corrcoef_np(y_reg_np, pred_reg_np),

            f"{prefix}_cls_acc": float(cls_metrics["cls_acc"]),
            f"{prefix}_cls_macro_f1": float(cls_metrics["cls_macro_f1"]),
            f"{prefix}_cls_num_valid": int(cls_metrics["cls_num_valid"]),

            f"{prefix}_fail_acc": float(fail_metrics["fail_acc"]),
            f"{prefix}_fail_f1": float(fail_metrics["fail_f1"]),
            f"{prefix}_fail_num_valid": int(fail_metrics["fail_num_valid"]),
            f"{prefix}_fail_pos_rate": float(fail_metrics["fail_pos_rate"]),
        }
        return metrics

    def train_one_epoch(self, epoch_idx: int):
        return self._run_epoch(self.train_loader, "train", epoch_idx, train=True)

    @torch.no_grad()
    def eval_one_epoch(self, loader, split_name: str, epoch_idx: int):
        return self._run_epoch(loader, split_name, epoch_idx, train=False)

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _save_raw_state_dict(self, path: Path):
        torch.save(self.model.state_dict(), path)

    def _write_history_csv(self):
        if len(self.history) == 0:
            return
        all_keys = sorted(set().union(*[set(r.keys()) for r in self.history]))
        with open(self.history_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def _write_history_json(self):
        with open(self.history_json, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def _write_model_config_json(self):
        cfg = {
            "dataset_dir": str(self.dataset_dir),
            "out_dir": str(self.out_dir),
            "model_type": str(self.model_type),
            "seq_len": int(self.seq_len),
            "input_dim": int(self.input_dim),
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
            "num_classes": int(self.num_classes),
            "feature_names": list(self.feature_names),
            "lambda_reg": float(self.lambda_reg),
            "lambda_cls": float(self.lambda_cls),
            "lambda_fail": float(self.lambda_fail),
        }
        with open(self.model_config_json, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def _write_best_info_json(self):
        payload = {
            "best_epoch": int(self.best_epoch),
            "best_val_joint": float(self.best_val_joint),
            "best_metrics": self.best_metrics,
            "model_type": str(self.model_type),
            "seq_len": int(self.seq_len),
            "input_dim": int(self.input_dim),
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
            "feature_names": list(self.feature_names),
        }
        with open(self.best_info_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def run(self):
        set_seed(self.seed)
        self.load_data()
        self.build_model_and_optimizer()
        self._write_model_config_json()

        print("========== Train Reliability Model ==========")
        print(f"dataset_dir: {self.dataset_dir}")
        print(f"out_dir: {self.out_dir}")
        print(f"device: {self.device}")
        print(f"model_type: {self.model_type}")
        print(f"seq_len: {self.seq_len}, input_dim: {self.input_dim}")
        print(f"train/val/test: {len(self.train_data['X'])}/{len(self.val_data['X'])}/{len(self.test_data['X'])}")

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.eval_one_epoch(self.val_loader, "val", epoch)

            val_joint = float(val_metrics["val_joint_loss"])
            self.scheduler.step(val_joint)

            merged = {}
            merged.update(train_metrics)
            merged.update(val_metrics)
            merged["lr"] = self._current_lr()
            self.history.append(merged)

            improved = (self.best_val_joint - val_joint) > self.min_delta
            if improved:
                self.best_val_joint = val_joint
                self.best_epoch = epoch
                self.best_metrics = dict(merged)
                self.epochs_no_improve = 0
                self._save_raw_state_dict(self.best_path)
                self._write_best_info_json()
            else:
                self.epochs_no_improve += 1

            self._save_raw_state_dict(self.last_path)

            print(
                f"[Epoch {epoch:03d}] "
                f"train_joint={train_metrics['train_joint_loss']:.6f} "
                f"val_joint={val_metrics['val_joint_loss']:.6f} "
                f"val_mse={val_metrics['val_mse']:.6f} "
                f"val_corr={val_metrics['val_corr']:.4f} "
                f"val_fail_f1={val_metrics['val_fail_f1']:.4f} "
                f"val_cls_f1={val_metrics['val_cls_macro_f1']:.4f} "
                f"lr={merged['lr']:.6e}"
            )

            self._write_history_json()
            self._write_history_csv()

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        if not self.best_path.exists():
            raise FileNotFoundError(f"找不到 best checkpoint: {self.best_path}")

        self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        test_metrics = self.eval_one_epoch(self.test_loader, "test", self.best_epoch)

        final_summary = {
            "best_epoch": int(self.best_epoch),
            "best_val_joint": float(self.best_val_joint),
            "train_best": self.best_metrics,
            "test": test_metrics,
            "best_path": str(self.best_path),
            "last_path": str(self.last_path),
            "best_info_json": str(self.best_info_json),
            "history_json": str(self.history_json),
            "history_csv": str(self.history_csv),
            "model_config_json": str(self.model_config_json),
        }

        self._write_history_json()
        self._write_history_csv()
        self._write_best_info_json()

        final_json = self.out_dir / "final_summary.json"
        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)

        print("================ Final Metrics ================")
        print(f"Best epoch: {self.best_epoch}")
        print(
            f"Val  | joint={self.best_metrics['val_joint_loss']:.6f} "
            f"MSE={self.best_metrics['val_mse']:.6f} "
            f"MAE={self.best_metrics['val_mae']:.6f} "
            f"Corr={self.best_metrics['val_corr']:.4f} "
            f"FailF1={self.best_metrics['val_fail_f1']:.4f} "
            f"ClsF1={self.best_metrics['val_cls_macro_f1']:.4f}"
        )
        print(
            f"Test | joint={test_metrics['test_joint_loss']:.6f} "
            f"MSE={test_metrics['test_mse']:.6f} "
            f"MAE={test_metrics['test_mae']:.6f} "
            f"Corr={test_metrics['test_corr']:.4f} "
            f"FailF1={test_metrics['test_fail_f1']:.4f} "
            f"ClsF1={test_metrics['test_cls_macro_f1']:.4f}"
        )
        print(f"模型與結果已輸出到: {self.out_dir}")

        return final_summary


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--model_type", type=str, default="gru", choices=["mlp", "gru", "tcn"])

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")

    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--lambda_fail", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    trainer = ReliabilityModelTrainer(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        lambda_reg=args.lambda_reg,
        lambda_cls=args.lambda_cls,
        lambda_fail=args.lambda_fail,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
    )
    trainer.run()


if __name__ == "__main__":
    main()
