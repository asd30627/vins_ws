import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_VINS_FEATURE_NAMES = [
    'current_is_keyframe',
    'feature_tracker_time_ms',
    'tracked_feature_count_raw',
    'tracked_feature_count_mgr',
    'mean_track_vel_px',
    'median_track_vel_px',
    'coverage_4x4',
    'img_dt_sec',
    'imu_sample_count',
    'acc_norm_mean',
    'gyr_norm_mean',
    'avg_track_length',
]

NONNEGATIVE_FEATURES = set(DEFAULT_VINS_FEATURE_NAMES)
CLASS_NAME_TO_ID = {'harmful': 0, 'neutral': 1, 'helpful': 2}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}
SPLITS = ['train', 'val', 'test']


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


def read_csv_rows(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def bincount3(y: np.ndarray) -> np.ndarray:
    valid = y[y >= 0]
    if len(valid) == 0:
        return np.zeros((3,), dtype=np.int64)
    return np.bincount(valid.astype(np.int64), minlength=3).astype(np.int64)


class VinsReliabilityDatasetBuilder:
    def __init__(
        self,
        feature_csv: str,
        label_csv: str,
        out_dir: str,
        seq_len: int = 8,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_gap: Optional[int] = None,
        min_rows: int = 32,
        feature_names: Optional[List[str]] = None,
        drop_invalid_labels: bool = True,
        split_mode: str = 'chronological',
        block_size: int = 512,
        seed: int = 42,
        search_trials: int = 300,
    ):
        self.feature_csv = Path(feature_csv).expanduser().resolve()
        self.label_csv = Path(label_csv).expanduser().resolve()
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.seq_len = int(seq_len)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.purge_gap = int(purge_gap) if purge_gap is not None else max(self.seq_len - 1, 0)
        self.min_rows = int(min_rows)
        self.feature_names = list(feature_names) if feature_names is not None else list(DEFAULT_VINS_FEATURE_NAMES)
        self.drop_invalid_labels = bool(drop_invalid_labels)
        self.split_mode = str(split_mode)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.search_trials = int(search_trials)

        self.feature_table_csv = self.out_dir / 'feature_table_vins.csv'
        self.debug_csv = self.out_dir / 'feature_label_debug_vins.csv'
        self.stats_json = self.out_dir / 'dataset_stats.json'

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

    def _read_feature_rows(self) -> List[Dict[str, object]]:
        rows = read_csv_rows(self.feature_csv)
        parsed = []
        for row in rows:
            pair_id = to_int(row.get('pair_id', row.get('update_id', -1)), -1)
            if pair_id < 0:
                continue
            item = {
                'pair_id': pair_id,
                'update_id': to_int(row.get('update_id', pair_id), pair_id),
                'timestamp': to_float(row.get('timestamp', np.nan), np.nan),
            }
            for name in self.feature_names:
                item[name] = to_float(row.get(name, np.nan), np.nan)
            parsed.append(item)
        parsed.sort(key=lambda x: x['pair_id'])
        return parsed

    def _read_label_map(self) -> Dict[int, Dict[str, object]]:
        rows = read_csv_rows(self.label_csv)
        label_map = {}
        for row in rows:
            pair_id = to_int(row.get('pair_id', row.get('update_id', -1)), -1)
            if pair_id < 0:
                continue
            label_map[pair_id] = {
                'label_reg': to_float(row.get('label_reg', np.nan), np.nan),
                'label_cls': to_int(row.get('label_cls', -1), -1),
                'class_name': row.get('class_name', ''),
                'label_source': row.get('label_source', ''),
            }
        return label_map

    def _merge_rows(self, feature_rows: List[Dict[str, object]], label_map: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
        merged = []
        num_dropped_invalid = 0
        for row in feature_rows:
            label = label_map.get(row['pair_id'], {})
            merged_row = {
                **row,
                'label_reg': float(label.get('label_reg', np.nan)),
                'label_cls': int(label.get('label_cls', -1)),
                'class_name': str(label.get('class_name', '')),
                'label_source': str(label.get('label_source', '')),
            }
            valid_label = np.isfinite(merged_row['label_reg']) and merged_row['label_cls'] in CLASS_ID_TO_NAME
            if self.drop_invalid_labels and not valid_label:
                num_dropped_invalid += 1
                continue
            merged.append(merged_row)
        self.num_dropped_invalid = num_dropped_invalid
        return merged

    def _write_debug_csv(self, merged: List[Dict[str, object]]):
        fieldnames = ['pair_id', 'update_id', 'timestamp'] + self.feature_names + ['label_reg', 'label_cls', 'class_name', 'label_source']
        with open(self.debug_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in merged:
                writer.writerow(row)

        infer_fields = ['pair_id', 'update_id', 'timestamp'] + self.feature_names
        with open(self.feature_table_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=infer_fields)
            writer.writeheader()
            for row in merged:
                writer.writerow({k: row.get(k, '') for k in infer_fields})

    def _rows_to_matrix(self, rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = np.zeros((len(rows), len(self.feature_names)), dtype=np.float32)
        y_reg = np.full((len(rows),), np.nan, dtype=np.float32)
        y_cls = np.full((len(rows),), -1, dtype=np.int64)
        pair_ids = np.full((len(rows),), -1, dtype=np.int32)
        for i, row in enumerate(rows):
            pair_ids[i] = int(row['pair_id'])
            y_reg[i] = float(row['label_reg']) if np.isfinite(row['label_reg']) else np.nan
            y_cls[i] = int(row['label_cls'])
            for j, name in enumerate(self.feature_names):
                X[i, j] = float(row.get(name, np.nan))
        return X, y_reg, y_cls, pair_ids

    def _fit_fill_values(self, X_train_raw: np.ndarray) -> Dict[str, float]:
        fill_values = {}
        for j, name in enumerate(self.feature_names):
            fill_values[name] = self._median_ignore_invalid(X_train_raw[:, j], name, default_value=0.0)
        return fill_values

    def _apply_fill(self, X_raw: np.ndarray, fill_values: Dict[str, float]) -> np.ndarray:
        X = X_raw.copy().astype(np.float32)
        for j, name in enumerate(self.feature_names):
            col = X[:, j]
            invalid = ~np.isfinite(col)
            if name in NONNEGATIVE_FEATURES:
                invalid = invalid | (col < 0.0)
            if np.any(invalid):
                col[invalid] = float(fill_values[name])
                X[:, j] = col
        return X

    def _build_sequences_single_block(self, X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray, pair_ids: np.ndarray):
        if len(X) < self.seq_len:
            return None
        xs, ys_reg, ys_cls, pids = [], [], [], []
        for end_idx in range(self.seq_len - 1, len(X)):
            start_idx = end_idx - self.seq_len + 1
            xs.append(X[start_idx:end_idx + 1])
            ys_reg.append(y_reg[end_idx])
            ys_cls.append(y_cls[end_idx])
            pids.append(pair_ids[end_idx])
        return {
            'X': np.asarray(xs, dtype=np.float32),
            'y_reg': np.asarray(ys_reg, dtype=np.float32),
            'y_cls': np.asarray(ys_cls, dtype=np.int64),
            'pair_ids': np.asarray(pids, dtype=np.int32),
        }

    def _concat_seq_parts(self, parts):
        valid = [p for p in parts if p is not None and len(p['X']) > 0]
        if not valid:
            return None
        return {
            'X': np.concatenate([p['X'] for p in valid], axis=0),
            'y_reg': np.concatenate([p['y_reg'] for p in valid], axis=0),
            'y_cls': np.concatenate([p['y_cls'] for p in valid], axis=0),
            'pair_ids': np.concatenate([p['pair_ids'] for p in valid], axis=0),
        }

    def _save_npz(self, out_path: Path, data: Dict[str, np.ndarray]):
        np.savez_compressed(
            out_path,
            X=data['X'],
            y_reg=data['y_reg'],
            y_cls=data['y_cls'],
            pair_ids=data['pair_ids'],
            feature_names=np.asarray(self.feature_names, dtype=object),
            seq_len=np.asarray([self.seq_len], dtype=np.int32),
        )

    def _make_blocks(self, merged: List[Dict[str, object]]):
        blocks = []
        for block_id, start in enumerate(range(0, len(merged), self.block_size)):
            end = min(len(merged), start + self.block_size)
            block = merged[start:end]
            if len(block) < self.seq_len:
                continue
            y = np.array([int(r['label_cls']) for r in block], dtype=np.int64)
            counts = bincount3(y)
            blocks.append({
                'block_id': block_id,
                'rows': block,
                'num_rows': len(block),
                'class_counts': counts,
                'classes_present': tuple((counts > 0).astype(int).tolist()),
            })
        return blocks

    def _chronological_split(self, merged: List[Dict[str, object]]):
        n = len(merged)
        n_train = max(1, int(round(n * self.train_ratio)))
        n_val = max(1, int(round(n * self.val_ratio)))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
        train_rows = merged[:n_train]
        val_start = min(n, n_train + self.purge_gap)
        val_rows = merged[val_start: val_start + n_val]
        test_start = min(n, val_start + n_val + self.purge_gap)
        test_rows = merged[test_start:]
        if len(val_rows) < self.seq_len or len(test_rows) < self.seq_len:
            train_rows = merged[:n_train]
            val_rows = merged[n_train:n_train + n_val]
            test_rows = merged[n_train + n_val:]
        return {'train': [train_rows], 'val': [val_rows], 'test': [test_rows]}

    def _block_mixed_split(self, merged: List[Dict[str, object]]):
        blocks = self._make_blocks(merged)
        if len(blocks) < 3:
            raise RuntimeError('block_mixed 需要至少 3 個有效 block')
        rng = random.Random(self.seed)
        order = list(range(len(blocks)))
        rng.shuffle(order)
        n_blocks = len(blocks)
        n_train = max(1, int(round(n_blocks * self.train_ratio)))
        n_val = max(1, int(round(n_blocks * self.val_ratio)))
        n_test = n_blocks - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
        train_ids = set(order[:n_train])
        val_ids = set(order[n_train:n_train + n_val])
        out = {'train': [], 'val': [], 'test': []}
        for i, block in enumerate(blocks):
            if i in train_ids:
                out['train'].append(block['rows'])
            elif i in val_ids:
                out['val'].append(block['rows'])
            else:
                out['test'].append(block['rows'])
        return out

    def _score_assignment(self, assignment, blocks, target_rows):
        # smaller is better
        row_penalty = 0.0
        class_presence_penalty = 0.0
        rarity_penalty = 0.0
        for split in SPLITS:
            row_count = sum(blocks[i]['num_rows'] for i in assignment[split])
            class_counts = sum((blocks[i]['class_counts'] for i in assignment[split]), np.zeros((3,), dtype=np.int64))
            row_penalty += abs(row_count - target_rows[split]) / max(target_rows[split], 1)
            missing = (class_counts == 0).sum()
            class_presence_penalty += 1000.0 * missing
            rarity_penalty += 10.0 / max(class_counts[0], 1)  # push harmful to appear everywhere if possible
        return class_presence_penalty + 10.0 * row_penalty + rarity_penalty

    def _class_aware_block_split(self, merged: List[Dict[str, object]]):
        blocks = self._make_blocks(merged)
        if len(blocks) < 3:
            raise RuntimeError('block_class_aware 需要至少 3 個有效 block')

        total_rows = sum(b['num_rows'] for b in blocks)
        target_rows = {
            'train': total_rows * self.train_ratio,
            'val': total_rows * self.val_ratio,
            'test': total_rows * self.test_ratio,
        }

        best_assignment = None
        best_score = float('inf')
        for trial in range(self.search_trials):
            rng = random.Random(self.seed + trial)
            order = list(range(len(blocks)))
            # random + favor blocks that contain harmful or multiple classes
            rng.shuffle(order)
            order.sort(key=lambda i: (blocks[i]['class_counts'][0] > 0, sum(blocks[i]['class_counts'] > 0), blocks[i]['num_rows']), reverse=True)

            assignment = {k: [] for k in SPLITS}
            split_rows = {k: 0 for k in SPLITS}
            split_counts = {k: np.zeros((3,), dtype=np.int64) for k in SPLITS}

            # seed each split with one block if possible, prioritizing class coverage
            for split in SPLITS:
                best_i = None
                best_local_score = None
                for i in order:
                    if i in assignment['train'] or i in assignment['val'] or i in assignment['test']:
                        continue
                    new_counts = split_counts[split] + blocks[i]['class_counts']
                    local_score = (
                        (new_counts == 0).sum(),
                        -int(new_counts[0]),
                        abs((split_rows[split] + blocks[i]['num_rows']) - target_rows[split]),
                    )
                    if best_i is None or local_score < best_local_score:
                        best_i = i
                        best_local_score = local_score
                assignment[split].append(best_i)
                split_rows[split] += blocks[best_i]['num_rows']
                split_counts[split] += blocks[best_i]['class_counts']

            # assign remaining blocks to the split with worst row deficit + class deficit
            used = set(sum(assignment.values(), []))
            for i in order:
                if i in used:
                    continue
                candidates = []
                for split in SPLITS:
                    deficit = max(target_rows[split] - split_rows[split], 0.0)
                    class_gain = int(((split_counts[split] == 0) & (blocks[i]['class_counts'] > 0)).sum())
                    score = (-class_gain, -deficit, split_rows[split])
                    candidates.append((score, split))
                candidates.sort()
                chosen = candidates[0][1]
                assignment[chosen].append(i)
                split_rows[chosen] += blocks[i]['num_rows']
                split_counts[chosen] += blocks[i]['class_counts']

            score = self._score_assignment(assignment, blocks, target_rows)
            if score < best_score:
                best_score = score
                best_assignment = assignment
                if best_score < 1e-6:
                    break

        out = {k: [] for k in SPLITS}
        for split in SPLITS:
            for i in best_assignment[split]:
                out[split].append(blocks[i]['rows'])
        return out

    def _build_split_data(self, split_blocks):
        train_rows_all = [row for block in split_blocks['train'] for row in block]
        if len(train_rows_all) < self.min_rows:
            raise RuntimeError(f'有效 train rows 太少: {len(train_rows_all)}')
        X_train_raw, _, _, _ = self._rows_to_matrix(train_rows_all)
        fill_values = self._fit_fill_values(X_train_raw)
        X_train_filled = self._apply_fill(X_train_raw, fill_values)
        mean, std = self._compute_mean_std(X_train_filled)

        split_out = {}
        split_row_counts = {}
        split_block_counts = {}
        split_class_counts = {}
        for split_name, blocks in split_blocks.items():
            seq_parts = []
            row_count = 0
            y_all = []
            for block in blocks:
                X_raw, y_reg, y_cls, pair_ids = self._rows_to_matrix(block)
                X = self._apply_fill(X_raw, fill_values)
                X = ((X - mean) / std).astype(np.float32)
                seq_part = self._build_sequences_single_block(X, y_reg, y_cls, pair_ids)
                if seq_part is not None:
                    seq_parts.append(seq_part)
                row_count += len(block)
                y_all.append(y_cls)
            data = self._concat_seq_parts(seq_parts)
            if data is None:
                raise RuntimeError(f'{split_name} split 無法建立任何 sequence')
            split_out[split_name] = data
            split_row_counts[split_name] = row_count
            split_block_counts[split_name] = len(blocks)
            split_class_counts[split_name] = bincount3(np.concatenate(y_all) if len(y_all) > 0 else np.array([], dtype=np.int64)).tolist()
        return split_out, fill_values, mean, std, split_row_counts, split_block_counts, split_class_counts

    def run(self):
        feature_rows = self._read_feature_rows()
        label_map = self._read_label_map()
        merged = self._merge_rows(feature_rows, label_map)
        self._write_debug_csv(merged)

        if len(merged) < self.min_rows:
            raise RuntimeError(f'有效列數太少，至少需要 {self.min_rows} 列，目前只有 {len(merged)} 列')

        if self.split_mode == 'chronological':
            split_blocks = self._chronological_split(merged)
        elif self.split_mode == 'block_mixed':
            split_blocks = self._block_mixed_split(merged)
        elif self.split_mode == 'block_class_aware':
            split_blocks = self._class_aware_block_split(merged)
        else:
            raise ValueError(f'不支援的 split_mode: {self.split_mode}')

        split_out, fill_values, mean, std, split_row_counts, split_block_counts, split_class_counts = self._build_split_data(split_blocks)

        self._save_npz(self.out_dir / 'train.npz', split_out['train'])
        self._save_npz(self.out_dir / 'val.npz', split_out['val'])
        self._save_npz(self.out_dir / 'test.npz', split_out['test'])

        stats = {
            'feature_csv': str(self.feature_csv),
            'label_csv': str(self.label_csv),
            'feature_table_csv': str(self.feature_table_csv),
            'feature_names': self.feature_names,
            'fill_values': fill_values,
            'mean': mean.astype(float).tolist(),
            'std': std.astype(float).tolist(),
            'seq_len': self.seq_len,
            'num_feature_rows_raw': len(feature_rows),
            'num_rows_after_merge': len(merged),
            'num_dropped_invalid_labels': int(getattr(self, 'num_dropped_invalid', 0)),
            'num_train_rows': split_row_counts['train'],
            'num_val_rows': split_row_counts['val'],
            'num_test_rows': split_row_counts['test'],
            'num_train_seq': int(len(split_out['train']['X'])),
            'num_val_seq': int(len(split_out['val']['X'])),
            'num_test_seq': int(len(split_out['test']['X'])),
            'num_train_blocks': split_block_counts['train'],
            'num_val_blocks': split_block_counts['val'],
            'num_test_blocks': split_block_counts['test'],
            'train_class_counts': split_class_counts['train'],
            'val_class_counts': split_class_counts['val'],
            'test_class_counts': split_class_counts['test'],
            'class_map': CLASS_NAME_TO_ID,
            'feature_source': 'vins_native_features_v1',
            'drop_invalid_labels': self.drop_invalid_labels,
            'split_mode': self.split_mode,
            'block_size': self.block_size,
            'seed': self.seed,
            'search_trials': self.search_trials,
        }
        with open(self.stats_json, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f'[OK] dataset -> {self.out_dir}')
        print(f'[OK] train -> {self.out_dir / "train.npz"}')
        print(f'[OK] val   -> {self.out_dir / "val.npz"}')
        print(f'[OK] test  -> {self.out_dir / "test.npz"}')
        print(f'[OK] stats -> {self.stats_json}')
        print(f'[OK] feature_table_for_infer -> {self.feature_table_csv}')
        print(json.dumps(stats, indent=2, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--feature_csv', type=str, required=True)
    p.add_argument('--label_csv', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--train_ratio', type=float, default=0.70)
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--test_ratio', type=float, default=0.15)
    p.add_argument('--purge_gap', type=int, default=None)
    p.add_argument('--min_rows', type=int, default=32)
    p.add_argument('--keep_invalid_labels', action='store_true')
    p.add_argument('--split_mode', type=str, default='chronological', choices=['chronological', 'block_mixed', 'block_class_aware'])
    p.add_argument('--block_size', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--search_trials', type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    builder = VinsReliabilityDatasetBuilder(
        feature_csv=args.feature_csv,
        label_csv=args.label_csv,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        purge_gap=args.purge_gap,
        min_rows=args.min_rows,
        drop_invalid_labels=(not args.keep_invalid_labels),
        split_mode=args.split_mode,
        block_size=args.block_size,
        seed=args.seed,
        search_trials=args.search_trials,
    )
    builder.run()


if __name__ == '__main__':
    main()
