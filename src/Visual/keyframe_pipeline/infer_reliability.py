import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from build_reliability_dataset import NONNEGATIVE_FEATURES, CLASS_ID_TO_NAME
from train_reliability_model import build_model


# =========================================================
# 基本工具
# =========================================================

def read_csv_rows(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def to_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def to_int(value, default=-1):
    try:
        return int(float(value))
    except Exception:
        return default


def to_str(value, default=''):
    if value is None:
        return default
    return str(value)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
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


# =========================================================
# Inferencer
# =========================================================

class ReliabilityInferencer:
    def __init__(
        self,
        sequence_dir,
        feature_csv='',
        dataset_dir='',
        checkpoint='',
        out_dir='',
        label_csv='',
        tau=0.5,
        helpful_prob_thr=0.5,
        batch_size=256,
        device=None,

        # fallback model config if model_config.json 不存在
        model_type='gru',
        hidden_dim=64,
        num_layers=1,
        dropout=0.10,
        num_classes=3,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.feature_csv = (
            Path(feature_csv).expanduser().resolve()
            if feature_csv
            else self.sequence_dir / 'features' / 'all_candidate_features.csv'
        )
        self.dataset_dir = (
            Path(dataset_dir).expanduser().resolve()
            if dataset_dir
            else self.sequence_dir / 'reliability_dataset'
        )
        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.sequence_dir / 'reliability_inference'
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint
            else self.dataset_dir / 'model_runs' / model_type / 'best.pt'
        )
        self.model_config_json = self.checkpoint.parent / 'model_config.json'
        self.stats_json = self.dataset_dir / 'dataset_stats.json'

        self.label_csv = (
            Path(label_csv).expanduser().resolve()
            if label_csv
            else self.sequence_dir / 'reliability_labels' / 'reliability_labels.csv'
        )

        self.tau = float(tau)
        self.helpful_prob_thr = float(helpful_prob_thr)
        self.batch_size = int(batch_size)
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.fallback_model_type = str(model_type)
        self.fallback_hidden_dim = int(hidden_dim)
        self.fallback_num_layers = int(num_layers)
        self.fallback_dropout = float(dropout)
        self.fallback_num_classes = int(num_classes)

        self.feature_names = None
        self.fill_values = None
        self.mean = None
        self.std = None
        self.seq_len = None

        self.model_type = None
        self.hidden_dim = None
        self.num_layers = None
        self.dropout = None
        self.num_classes = None

    # -----------------------------------------------------
    # loading metadata
    # -----------------------------------------------------
    def check_required_files(self):
        if not self.feature_csv.exists():
            raise FileNotFoundError(f'找不到 feature_csv: {self.feature_csv}')
        if not self.stats_json.exists():
            raise FileNotFoundError(f'找不到 dataset_stats.json: {self.stats_json}')
        if not self.checkpoint.exists():
            raise FileNotFoundError(f'找不到 checkpoint: {self.checkpoint}')

    def load_stats_and_model_config(self):
        with open(self.stats_json, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        self.feature_names = list(stats['feature_names'])
        self.fill_values = dict(stats['fill_values'])
        self.mean = np.asarray(stats['mean'], dtype=np.float32)
        self.std = np.asarray(stats['std'], dtype=np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        self.seq_len = int(stats['seq_len'])

        if self.model_config_json.exists():
            with open(self.model_config_json, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            self.model_type = str(cfg.get('model_type', self.fallback_model_type))
            self.hidden_dim = int(cfg.get('hidden_dim', self.fallback_hidden_dim))
            self.num_layers = int(cfg.get('num_layers', self.fallback_num_layers))
            self.dropout = float(cfg.get('dropout', self.fallback_dropout))
            self.num_classes = int(cfg.get('num_classes', self.fallback_num_classes))
        else:
            self.model_type = self.fallback_model_type
            self.hidden_dim = self.fallback_hidden_dim
            self.num_layers = self.fallback_num_layers
            self.dropout = self.fallback_dropout
            self.num_classes = self.fallback_num_classes

    def load_label_map(self) -> Dict[int, Dict[str, object]]:
        if not self.label_csv.exists():
            return {}

        rows = read_csv_rows(self.label_csv)
        label_map = {}

        for row in rows:
            pair_id = to_int(row.get('pair_id', -1), -1)
            if pair_id < 0:
                continue

            label_map[pair_id] = {
                'label_reg': to_float(row.get('label_reg', np.nan), np.nan),
                'label_cls': to_int(row.get('label_cls', -1), -1),
                'class_name': to_str(row.get('class_name', '')),
                'label_source': to_str(row.get('label_source', '')),
            }

        return label_map

    # -----------------------------------------------------
    # feature table
    # -----------------------------------------------------
    def build_feature_table(self, rows, feature_names):
        rows_sorted = sorted(rows, key=lambda r: int(to_float(r.get('pair_id', 0), 0)))

        pair_ids = []
        meta = {}
        cols = {name: [] for name in feature_names}

        for row in rows_sorted:
            pid = int(to_float(row.get('pair_id', 0), 0))
            pair_ids.append(pid)

            meta[pid] = {
                'pair_id': pid,
                'kf_prev_id': int(to_float(row.get('kf_prev_id', -1), -1)),
                'kf_curr_id': int(to_float(row.get('kf_curr_id', -1), -1)),
                'src_prev_id': int(to_float(row.get('src_prev_id', -1), -1)),
                'src_curr_id': int(to_float(row.get('src_curr_id', -1), -1)),
                'image_prev': row.get('image_prev', ''),
                'image_curr': row.get('image_curr', ''),
                'timestamp_prev': row.get('timestamp_prev', ''),
                'timestamp_curr': row.get('timestamp_curr', ''),
                'accepted': int(to_float(row.get('accepted', -1), -1)),
                'reason': row.get('reason', ''),
            }

            for name in feature_names:
                if name.startswith('valid_'):
                    base_name = name[len('valid_'):]
                    v = to_float(row.get(base_name, np.nan), np.nan)
                    cols[name].append(1.0 if is_valid_feature_value(base_name, v) else 0.0)
                else:
                    v = to_float(row.get(name, np.nan), np.nan)
                    cols[name].append(v if is_valid_feature_value(name, v) else np.nan)

        table = {
            'pair_ids': np.array(pair_ids, dtype=np.int32),
            'meta': meta,
        }

        for name in feature_names:
            table[name] = np.array(cols[name], dtype=np.float64)

        return table

    def apply_fill_values(self, table, feature_names, fill_values):
        X_cols = []

        for name in feature_names:
            arr = table[name].copy()

            if name.startswith('valid_'):
                fill_v = 0.0
            else:
                fill_v = float(fill_values.get(name, 0.0))

            arr[~np.isfinite(arr)] = fill_v
            X_cols.append(arr.reshape(-1, 1))

        return np.concatenate(X_cols, axis=1).astype(np.float32)

    def standardize_features(self, X):
        mean = self.mean.reshape(1, -1)
        std = self.std.reshape(1, -1)
        return ((X - mean) / std).astype(np.float32)

    def build_sequences(self, X, pair_ids, seq_len):
        xs = []
        seq_meta = []

        for end_idx in range(seq_len - 1, len(X)):
            start_idx = end_idx - seq_len + 1
            xs.append(X[start_idx:end_idx + 1])

            seq_meta.append({
                'start_pair_id': int(pair_ids[start_idx]),
                'end_pair_id': int(pair_ids[end_idx]),
                'start_index': int(start_idx),
                'end_index': int(end_idx),
            })

        if len(xs) == 0:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), []

        return np.stack(xs).astype(np.float32), seq_meta

    # -----------------------------------------------------
    # model inference
    # -----------------------------------------------------
    def build_model(self, input_dim):
        model = build_model(
            model_type=self.model_type,
            seq_len=self.seq_len,
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=self.num_classes,
        ).to(self.device)
        return model

    @torch.no_grad()
    def infer_in_batches(self, model, X_seq):
        model.eval()

        regs = []
        logits_list = []

        for start in range(0, len(X_seq), self.batch_size):
            end = min(start + self.batch_size, len(X_seq))
            xb = torch.from_numpy(X_seq[start:end]).float().to(self.device)

            outputs = model(xb, return_dict=True)
            pred_reg = outputs['reg'].detach().cpu().numpy().astype(np.float32).reshape(-1)
            cls_logits = outputs['cls_logits'].detach().cpu().numpy().astype(np.float32)

            regs.append(pred_reg)
            logits_list.append(cls_logits)

        if len(regs) == 0:
            return (
                np.array([], dtype=np.float32),
                np.empty((0, self.num_classes), dtype=np.float32),
            )

        regs = np.concatenate(regs, axis=0).astype(np.float32).reshape(-1)
        regs = np.clip(regs, 0.0, 1.0)

        logits = np.concatenate(logits_list, axis=0).astype(np.float32)
        return regs, logits

    # -----------------------------------------------------
    # output
    # -----------------------------------------------------
    def run(self):
        self.check_required_files()
        self.load_stats_and_model_config()

        feature_rows = read_csv_rows(self.feature_csv)
        table = self.build_feature_table(feature_rows, self.feature_names)

        X = self.apply_fill_values(table, self.feature_names, self.fill_values)
        X = self.standardize_features(X)

        pair_ids = table['pair_ids']
        X_seq, seq_meta = self.build_sequences(X, pair_ids, self.seq_len)

        model = self.build_model(input_dim=X.shape[1])
        state_dict = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(state_dict)

        pred_reg, cls_logits = self.infer_in_batches(model, X_seq)
        probs = softmax_np(cls_logits, axis=1) if len(cls_logits) > 0 else np.empty((0, self.num_classes), dtype=np.float32)
        pred_cls = np.argmax(probs, axis=1).astype(np.int64) if len(probs) > 0 else np.array([], dtype=np.int64)

        label_map = self.load_label_map()

        # 先建立一個 pair_id -> prediction 的 map
        pred_map = {}
        for i, meta in enumerate(seq_meta):
            end_pair_id = int(meta['end_pair_id'])

            p_harmful = float(probs[i, 0]) if self.num_classes >= 1 else np.nan
            p_neutral = float(probs[i, 1]) if self.num_classes >= 2 else np.nan
            p_helpful = float(probs[i, 2]) if self.num_classes >= 3 else np.nan

            gate_pass_tau = int(float(pred_reg[i]) >= self.tau)
            gate_pass_helpful_prob = int(np.isfinite(p_helpful) and p_helpful >= self.helpful_prob_thr)
            pred_class_id = int(pred_cls[i])
            pred_class_name = CLASS_ID_TO_NAME.get(pred_class_id, f'class_{pred_class_id}')
            gate_pass_by_class = int(pred_class_name == 'helpful')

            pred_map[end_pair_id] = {
                'has_prediction': 1,
                'start_pair_id': int(meta['start_pair_id']),
                'end_pair_id': int(end_pair_id),
                'w_pred': float(pred_reg[i]),

                'pred_class_id': int(pred_class_id),
                'pred_class': str(pred_class_name),

                'p_harmful': float(p_harmful),
                'p_neutral': float(p_neutral),
                'p_helpful': float(p_helpful),

                # backward compatibility
                'gate_pass': int(gate_pass_tau),
                'soft_target_proxy': float(label_map.get(end_pair_id, {}).get('label_reg', np.nan)),

                # extra gates
                'gate_pass_tau': int(gate_pass_tau),
                'gate_pass_helpful_prob': int(gate_pass_helpful_prob),
                'gate_pass_by_class': int(gate_pass_by_class),
            }

        # 對所有 pair 都輸出一列；前 seq_len-1 筆沒有 prediction
        out_rows = []
        for pid in pair_ids.tolist():
            meta = table['meta'][int(pid)]
            base = {
                'pair_id': int(pid),
                'start_pair_id': -1,
                'end_pair_id': int(pid),

                'kf_prev_id': int(meta['kf_prev_id']),
                'kf_curr_id': int(meta['kf_curr_id']),
                'src_prev_id': int(meta['src_prev_id']),
                'src_curr_id': int(meta['src_curr_id']),
                'image_prev': meta['image_prev'],
                'image_curr': meta['image_curr'],
                'timestamp_prev': meta['timestamp_prev'],
                'timestamp_curr': meta['timestamp_curr'],
                'accepted': int(meta['accepted']),
                'reason': meta['reason'],

                'has_prediction': 0,
                'w_pred': np.nan,
                'pred_class_id': -1,
                'pred_class': '',

                'p_harmful': np.nan,
                'p_neutral': np.nan,
                'p_helpful': np.nan,

                # backward compatibility
                'gate_pass': 0,
                'soft_target_proxy': np.nan,

                'gate_pass_tau': 0,
                'gate_pass_helpful_prob': 0,
                'gate_pass_by_class': 0,

                'label_reg_gt': np.nan,
                'label_cls_gt': -1,
                'label_class_name_gt': '',
                'label_source_gt': '',
            }

            if int(pid) in pred_map:
                base.update(pred_map[int(pid)])

            if int(pid) in label_map:
                gt = label_map[int(pid)]
                base['label_reg_gt'] = float(gt.get('label_reg', np.nan))
                base['label_cls_gt'] = int(gt.get('label_cls', -1))
                base['label_class_name_gt'] = str(gt.get('class_name', ''))
                base['label_source_gt'] = str(gt.get('label_source', ''))

                # backward compatibility: 舊 backend 讀這欄
                if np.isfinite(base['label_reg_gt']):
                    base['soft_target_proxy'] = float(base['label_reg_gt'])

            out_rows.append(base)

        pred_csv = self.out_dir / 'reliability_predictions.csv'
        summary_json = self.out_dir / 'summary.json'

        fieldnames = [
            'pair_id',
            'start_pair_id',
            'end_pair_id',

            'kf_prev_id',
            'kf_curr_id',
            'src_prev_id',
            'src_curr_id',
            'image_prev',
            'image_curr',
            'timestamp_prev',
            'timestamp_curr',
            'accepted',
            'reason',

            'has_prediction',
            'w_pred',

            'pred_class_id',
            'pred_class',
            'p_harmful',
            'p_neutral',
            'p_helpful',

            # backward compatibility
            'gate_pass',
            'soft_target_proxy',

            'gate_pass_tau',
            'gate_pass_helpful_prob',
            'gate_pass_by_class',

            'label_reg_gt',
            'label_cls_gt',
            'label_class_name_gt',
            'label_source_gt',
        ]

        with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)

        num_total_pairs = int(len(out_rows))
        num_with_prediction = int(sum(int(r['has_prediction']) for r in out_rows))
        num_gate_pass_tau = int(sum(int(r['gate_pass_tau']) for r in out_rows))
        num_gate_pass_helpful_prob = int(sum(int(r['gate_pass_helpful_prob']) for r in out_rows))
        num_gate_pass_by_class = int(sum(int(r['gate_pass_by_class']) for r in out_rows))

        pred_class_counts = {}
        for row in out_rows:
            cname = row['pred_class']
            if cname == '':
                continue
            pred_class_counts[cname] = pred_class_counts.get(cname, 0) + 1

        gt_overlap = 0
        if len(label_map) > 0:
            gt_overlap = int(sum(1 for r in out_rows if np.isfinite(to_float(r['label_reg_gt'], np.nan),)))

        summary = {
            'sequence_dir': str(self.sequence_dir),
            'feature_csv': str(self.feature_csv),
            'dataset_dir': str(self.dataset_dir),
            'checkpoint': str(self.checkpoint),
            'model_config_json': str(self.model_config_json),
            'stats_json': str(self.stats_json),
            'label_csv': str(self.label_csv),
            'pred_csv': str(pred_csv),

            'model_type': str(self.model_type),
            'seq_len': int(self.seq_len),
            'input_dim': int(X.shape[1]),
            'num_classes': int(self.num_classes),

            'tau': float(self.tau),
            'helpful_prob_thr': float(self.helpful_prob_thr),
            'batch_size': int(self.batch_size),

            'num_total_pairs': int(num_total_pairs),
            'num_with_prediction': int(num_with_prediction),
            'num_without_prediction': int(num_total_pairs - num_with_prediction),
            'num_gate_pass_tau': int(num_gate_pass_tau),
            'num_gate_pass_helpful_prob': int(num_gate_pass_helpful_prob),
            'num_gate_pass_by_class': int(num_gate_pass_by_class),
            'pred_class_counts': pred_class_counts,
            'gt_overlap_rows': int(gt_overlap),
        }

        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print('========== Infer Reliability v2 Finished ==========')
        print(f'pred_csv: {pred_csv}')
        print(f'summary_json: {summary_json}')
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f'{k}: {v}')
            elif k not in ['sequence_dir', 'feature_csv', 'dataset_dir', 'checkpoint', 'model_config_json', 'stats_json', 'label_csv', 'pred_csv']:
                print(f'{k}: {v}')

        return {
            'pred_csv': str(pred_csv),
            'summary_json': str(summary_json),
            'num_total_pairs': int(num_total_pairs),
            'num_with_prediction': int(num_with_prediction),
            'num_gate_pass_tau': int(num_gate_pass_tau),
            'num_gate_pass_helpful_prob': int(num_gate_pass_helpful_prob),
            'num_gate_pass_by_class': int(num_gate_pass_by_class),
            'pred_class_counts': pred_class_counts,
        }


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sequence_dir',
        type=str,
        required=True,
        help='例如: /mnt/sata4t/dataset/sequence_001'
    )
    parser.add_argument(
        '--feature_csv',
        type=str,
        default='',
        help='若留空，預設 sequence_dir/features/all_candidate_features.csv'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='',
        help='若留空，預設 sequence_dir/reliability_dataset'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='若留空，預設 dataset_dir/model_runs/model_type/best.pt'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='若留空，預設 sequence_dir/reliability_inference'
    )
    parser.add_argument(
        '--label_csv',
        type=str,
        default='',
        help='若留空，預設 sequence_dir/reliability_labels/reliability_labels.csv'
    )

    parser.add_argument('--tau', type=float, default=0.5, help='regression gate threshold')
    parser.add_argument('--helpful_prob_thr', type=float, default=0.5, help='classification helpful probability threshold')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='')

    # fallback config when model_config.json 不存在
    parser.add_argument('--model_type', type=str, default='gru', choices=['mlp', 'gru', 'tcn'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.10)
    parser.add_argument('--num_classes', type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device.strip() if args.device.strip() != '' else None

    inferencer = ReliabilityInferencer(
        sequence_dir=args.sequence_dir,
        feature_csv=args.feature_csv,
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        label_csv=args.label_csv,
        tau=args.tau,
        helpful_prob_thr=args.helpful_prob_thr,
        batch_size=args.batch_size,
        device=device,

        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=args.num_classes,
    )
    inferencer.run()


if __name__ == '__main__':
    main()