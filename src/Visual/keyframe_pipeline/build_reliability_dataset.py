import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_BASE_FEATURE_NAMES = [
    'odom_translation_m',
    'odom_rotation_deg',
    'num_keypoints0',
    'num_keypoints1',
    'num_matches',
    'num_inliers',
    'match_inlier_ratio',
    'mean_match_score',
    'coverage0',
    'coverage1',
    'parallax_mean_px',
    'parallax_median_px',
    'geo_error_mean',
    'geo_error_median',
    'vis_pose_ok',
    'vis_rot_deg',
    'e_rot_iv_deg',
    'e_trans_dir_iv_deg',
    'blur0',
    'blur1',
    'brightness0',
    'brightness1',
    'texture0',
    'texture1',
    'edge_density0',
    'edge_density1',
    'is_candidate_valid',
    'is_visual_usable',
    'has_missing_feature',
]

NONNEGATIVE_FEATURES = {
    'odom_translation_m',
    'odom_rotation_deg',
    'num_keypoints0',
    'num_keypoints1',
    'num_matches',
    'num_inliers',
    'match_inlier_ratio',
    'mean_match_score',
    'coverage0',
    'coverage1',
    'parallax_mean_px',
    'parallax_median_px',
    'geo_error_mean',
    'geo_error_median',
    'vis_pose_ok',
    'vis_rot_deg',
    'e_rot_iv_deg',
    'e_trans_dir_iv_deg',
    'blur0',
    'blur1',
    'brightness0',
    'brightness1',
    'texture0',
    'texture1',
    'edge_density0',
    'edge_density1',
    'is_candidate_valid',
    'is_visual_usable',
    'has_missing_feature',
}

CLASS_NAME_TO_ID = {
    'harmful': 0,
    'neutral': 1,
    'helpful': 2,
}

CLASS_ID_TO_NAME = {
    0: 'harmful',
    1: 'neutral',
    2: 'helpful',
}


class ReliabilityDatasetBuilder:
    """
    v2 重點：
    1. 不再從 feature 自己組 soft target
    2. 改成讀 feature_csv + label_csv 後 merge
    3. 先切 raw rows，再各 split 內 build sequence（purge split）
    4. fill / mean / std 只用 train split 計算
    5. validity mask 預設開啟
    """

    def __init__(
        self,
        sequence_dir,
        feature_csv='',
        label_csv='',
        out_dir='',
        seq_len=8,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        min_rows=32,
        purge_gap=None,
        use_validity_mask=True,
        base_feature_names=None,
        require_label_cls=False,
        debug_csv_name='feature_label_debug.csv',
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.feature_csv = (
            Path(feature_csv).expanduser().resolve()
            if feature_csv
            else self.sequence_dir / 'features' / 'all_candidate_features.csv'
        )
        self.label_csv = (
            Path(label_csv).expanduser().resolve()
            if label_csv
            else self.sequence_dir / 'reliability_labels' / 'reliability_labels.csv'
        )
        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.sequence_dir / 'reliability_dataset'

        self.seq_len = int(seq_len)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.min_rows = int(min_rows)
        self.purge_gap = int(purge_gap) if purge_gap is not None else max(self.seq_len - 1, 0)
        self.use_validity_mask = bool(use_validity_mask)
        self.require_label_cls = bool(require_label_cls)
        self.debug_csv_name = str(debug_csv_name)

        if base_feature_names is None:
            self.base_feature_names = list(DEFAULT_BASE_FEATURE_NAMES)
        else:
            self.base_feature_names = list(base_feature_names)

    @staticmethod
    def read_csv_rows(path: Path) -> List[Dict[str, str]]:
        with open(path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    @staticmethod
    def to_float(value, default=np.nan) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def to_int(value, default=-1) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def to_str(value, default='') -> str:
        if value is None:
            return default
        return str(value)

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

    def _check_inputs(self):
        if not self.feature_csv.exists():
            raise FileNotFoundError(f'找不到 feature_csv: {self.feature_csv}')
        if not self.label_csv.exists():
            raise FileNotFoundError(f'找不到 label_csv: {self.label_csv}')

        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f'train/val/test ratio 總和必須為 1.0，目前為 {total_ratio}')

        if self.seq_len < 1:
            raise ValueError('seq_len 必須 >= 1')

    def _parse_label_row(self, row: Dict[str, str]) -> Optional[Dict[str, object]]:
        pair_id = self.to_int(row.get('pair_id', -1), -1)
        if pair_id < 0:
            return None

        # regression label
        reg_candidates = [
            'label_reg',
            'y_reg',
            'normalized_improvement',
            'improvement_score',
            'score_reg',
            'soft_target',
        ]
        label_reg = np.nan
        for k in reg_candidates:
            if k in row:
                label_reg = self.to_float(row.get(k, np.nan), np.nan)
                if np.isfinite(label_reg):
                    break

        # classification label id
        cls_candidates = [
            'label_cls',
            'class_id',
            'class_index',
            'y_cls',
        ]
        label_cls = -1
        for k in cls_candidates:
            if k in row:
                label_cls = self.to_int(row.get(k, -1), -1)
                if label_cls in CLASS_ID_TO_NAME:
                    break

        # classification label name
        if label_cls not in CLASS_ID_TO_NAME:
            name_candidates = [
                'class_name',
                'label_name',
                'class_label',
                'label_cls_name',
            ]
            for k in name_candidates:
                if k in row:
                    name = self.to_str(row.get(k, '')).strip().lower()
                    if name in CLASS_NAME_TO_ID:
                        label_cls = CLASS_NAME_TO_ID[name]
                        break

        out = {
            'pair_id': int(pair_id),
            'label_reg': float(label_reg) if np.isfinite(label_reg) else np.nan,
            'label_cls': int(label_cls) if label_cls in CLASS_ID_TO_NAME else -1,
            'class_name': CLASS_ID_TO_NAME.get(label_cls, ''),
        }
        return out

    def _load_label_map(self) -> Dict[int, Dict[str, object]]:
        rows = self.read_csv_rows(self.label_csv)
        label_map = {}

        for row in rows:
            parsed = self._parse_label_row(row)
            if parsed is None:
                continue
            label_map[int(parsed['pair_id'])] = parsed

        return label_map

    def _merge_feature_and_label_rows(
        self,
        feature_rows: List[Dict[str, str]],
        label_map: Dict[int, Dict[str, object]],
    ) -> List[Dict[str, object]]:
        merged = []

        for row in feature_rows:
            pair_id = self.to_int(row.get('pair_id', -1), -1)
            if pair_id < 0:
                continue

            if pair_id not in label_map:
                continue

            label_info = label_map[pair_id]
            label_reg = float(label_info['label_reg'])
            label_cls = int(label_info['label_cls'])

            if not np.isfinite(label_reg):
                continue
            if self.require_label_cls and label_cls < 0:
                continue

            merged_row = {
                'pair_id': int(pair_id),
                'kf_prev_id': self.to_int(row.get('kf_prev_id', -1), -1),
                'kf_curr_id': self.to_int(row.get('kf_curr_id', -1), -1),
                'src_prev_id': self.to_int(row.get('src_prev_id', -1), -1),
                'src_curr_id': self.to_int(row.get('src_curr_id', -1), -1),
                'image_prev': self.to_str(row.get('image_prev', '')),
                'image_curr': self.to_str(row.get('image_curr', '')),
                'timestamp_prev': self.to_str(row.get('timestamp_prev', '')),
                'timestamp_curr': self.to_str(row.get('timestamp_curr', '')),
                'reason': self.to_str(row.get('reason', '')),
                'accepted': self.to_int(row.get('accepted', -1), -1),
                'label_reg': float(label_reg),
                'label_cls': int(label_cls),
                'class_name': self.to_str(label_info.get('class_name', '')),
            }

            for feat_name in self.base_feature_names:
                merged_row[feat_name] = self.to_float(row.get(feat_name, np.nan), np.nan)

            merged.append(merged_row)

        merged.sort(key=lambda x: x['pair_id'])
        return merged

    def _split_raw_rows(
        self,
        rows: List[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
        n = len(rows)
        if n < self.min_rows:
            raise ValueError(f'可用 labeled rows 太少，目前 {n} 筆，小於 min_rows={self.min_rows}')

        train_end = int(round(n * self.train_ratio))
        val_count = int(round(n * self.val_ratio))
        val_end = min(n, train_end + val_count)

        # 基本保護
        train_end = max(train_end, self.seq_len)
        val_end = max(val_end, train_end)

        # purge
        val_start = min(n, train_end + self.purge_gap)
        test_start = min(n, val_end + self.purge_gap)

        train_rows = rows[:train_end]
        val_rows = rows[val_start:val_end]
        test_rows = rows[test_start:]

        if len(train_rows) < self.seq_len:
            raise ValueError('train split 太短，無法建立 sequence')
        if len(val_rows) < self.seq_len:
            raise ValueError(
                f'val split 太短，無法建立 sequence。'
                f'目前 len(val_rows)={len(val_rows)}，seq_len={self.seq_len}。'
                f'可考慮減少 purge_gap、減少 seq_len、或增加資料量。'
            )
        if len(test_rows) < self.seq_len:
            raise ValueError(
                f'test split 太短，無法建立 sequence。'
                f'目前 len(test_rows)={len(test_rows)}，seq_len={self.seq_len}。'
                f'可考慮減少 purge_gap、減少 seq_len、或增加資料量。'
            )

        return train_rows, val_rows, test_rows

    def _build_train_fill_values(self, train_rows: List[Dict[str, object]]) -> Dict[str, float]:
        fill_values = {}

        for feat_name in self.base_feature_names:
            arr = np.array([self.to_float(r.get(feat_name, np.nan), np.nan) for r in train_rows], dtype=np.float64)
            default_value = 0.0
            fill_values[feat_name] = self._median_ignore_invalid(arr, feat_name, default_value=default_value)

        return fill_values

    def _feature_names_with_mask(self) -> List[str]:
        if not self.use_validity_mask:
            return list(self.base_feature_names)

        out = []
        for name in self.base_feature_names:
            out.append(name)
            out.append(f'valid_{name}')
        return out

    def _rows_to_feature_matrix(
        self,
        rows: List[Dict[str, object]],
        fill_values: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feature_names = self._feature_names_with_mask()

        X_cols = []
        pair_ids = []
        y_reg = []
        y_cls = []

        for row in rows:
            pair_ids.append(int(row['pair_id']))
            y_reg.append(float(row['label_reg']))
            y_cls.append(int(row['label_cls']))

        for feat_name in self.base_feature_names:
            raw_vals = np.array([self.to_float(r.get(feat_name, np.nan), np.nan) for r in rows], dtype=np.float64)

            valid_mask = np.array(
                [1.0 if self._is_valid_feature_value(feat_name, v) else 0.0 for v in raw_vals],
                dtype=np.float32
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

        return (
            X,
            np.array(pair_ids, dtype=np.int32),
            np.array(y_reg, dtype=np.float32),
            np.array(y_cls, dtype=np.int64),
        )

    def _standardize_with_train_stats(
        self,
        X_train_raw: np.ndarray,
        X_other_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, std = self._compute_mean_std(X_train_raw)
        X_train = ((X_train_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
        X_other = ((X_other_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
        return X_train, X_other, (mean, std)

    def _build_sequences_in_split(
        self,
        X: np.ndarray,
        pair_ids: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xs = []
        ys_reg = []
        ys_cls = []
        end_pair_ids = []

        for end_idx in range(self.seq_len - 1, len(X)):
            start_idx = end_idx - self.seq_len + 1
            xs.append(X[start_idx:end_idx + 1])
            ys_reg.append(y_reg[end_idx])
            ys_cls.append(y_cls[end_idx])
            end_pair_ids.append(pair_ids[end_idx])

        X_seq = np.stack(xs).astype(np.float32)
        y_reg_seq = np.array(ys_reg, dtype=np.float32)
        y_cls_seq = np.array(ys_cls, dtype=np.int64)
        end_pair_ids = np.array(end_pair_ids, dtype=np.int32)

        return X_seq, y_reg_seq, y_cls_seq, end_pair_ids

    def _write_debug_csv(
        self,
        train_rows: List[Dict[str, object]],
        val_rows: List[Dict[str, object]],
        test_rows: List[Dict[str, object]],
    ):
        debug_path = self.out_dir / self.debug_csv_name
        fieldnames = [
            'split',
            'pair_id',
            'kf_prev_id',
            'kf_curr_id',
            'src_prev_id',
            'src_curr_id',
            'accepted',
            'reason',
            'label_reg',
            'label_cls',
            'class_name',
        ] + list(self.base_feature_names)

        with open(debug_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for split_name, rows in [('train', train_rows), ('val', val_rows), ('test', test_rows)]:
                for row in rows:
                    out = {'split': split_name}
                    for k in fieldnames:
                        if k == 'split':
                            continue
                        out[k] = row.get(k, '')
                    writer.writerow(out)

    def build(self):
        self._check_inputs()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        feature_rows = self.read_csv_rows(self.feature_csv)
        label_map = self._load_label_map()
        merged_rows = self._merge_feature_and_label_rows(feature_rows, label_map)

        if len(merged_rows) < self.min_rows:
            raise ValueError(
                f'merged 後可用 rows 太少，只有 {len(merged_rows)} 筆；'
                f'請確認 label_csv 是否與 feature_csv 的 pair_id 對得上。'
            )

        train_rows, val_rows, test_rows = self._split_raw_rows(merged_rows)

        fill_values = self._build_train_fill_values(train_rows)

        X_train_raw, pair_train, y_train_reg, y_train_cls = self._rows_to_feature_matrix(train_rows, fill_values)
        X_val_raw, pair_val, y_val_reg, y_val_cls = self._rows_to_feature_matrix(val_rows, fill_values)
        X_test_raw, pair_test, y_test_reg, y_test_cls = self._rows_to_feature_matrix(test_rows, fill_values)

        feature_names = self._feature_names_with_mask()

        X_train_std, X_val_std, (mean, std) = self._standardize_with_train_stats(X_train_raw, X_val_raw)
        _, X_test_std, _ = self._standardize_with_train_stats(X_train_raw, X_test_raw)

        train_X_seq, train_y_reg_seq, train_y_cls_seq, train_pair_ids = self._build_sequences_in_split(
            X_train_std, pair_train, y_train_reg, y_train_cls
        )
        val_X_seq, val_y_reg_seq, val_y_cls_seq, val_pair_ids = self._build_sequences_in_split(
            X_val_std, pair_val, y_val_reg, y_val_cls
        )
        test_X_seq, test_y_reg_seq, test_y_cls_seq, test_pair_ids = self._build_sequences_in_split(
            X_test_std, pair_test, y_test_reg, y_test_cls
        )

        # 為了跟你目前舊 trainer 相容，先保留 y = y_reg
        np.savez_compressed(
            self.out_dir / 'train.npz',
            X=train_X_seq.astype(np.float32),
            y=train_y_reg_seq.astype(np.float32),
            y_reg=train_y_reg_seq.astype(np.float32),
            y_cls=train_y_cls_seq.astype(np.int64),
            pair_ids=train_pair_ids.astype(np.int32),
            feature_names=np.array(feature_names, dtype=object),
            seq_len=np.array([self.seq_len], dtype=np.int32),
        )
        np.savez_compressed(
            self.out_dir / 'val.npz',
            X=val_X_seq.astype(np.float32),
            y=val_y_reg_seq.astype(np.float32),
            y_reg=val_y_reg_seq.astype(np.float32),
            y_cls=val_y_cls_seq.astype(np.int64),
            pair_ids=val_pair_ids.astype(np.int32),
            feature_names=np.array(feature_names, dtype=object),
            seq_len=np.array([self.seq_len], dtype=np.int32),
        )
        np.savez_compressed(
            self.out_dir / 'test.npz',
            X=test_X_seq.astype(np.float32),
            y=test_y_reg_seq.astype(np.float32),
            y_reg=test_y_reg_seq.astype(np.float32),
            y_cls=test_y_cls_seq.astype(np.int64),
            pair_ids=test_pair_ids.astype(np.int32),
            feature_names=np.array(feature_names, dtype=object),
            seq_len=np.array([self.seq_len], dtype=np.int32),
        )

        stats = {
            'feature_csv': str(self.feature_csv),
            'label_csv': str(self.label_csv),
            'num_feature_rows': int(len(feature_rows)),
            'num_label_rows': int(len(label_map)),
            'num_merged_rows': int(len(merged_rows)),

            'train_raw_rows': int(len(train_rows)),
            'val_raw_rows': int(len(val_rows)),
            'test_raw_rows': int(len(test_rows)),

            'train_sequences': int(len(train_X_seq)),
            'val_sequences': int(len(val_X_seq)),
            'test_sequences': int(len(test_X_seq)),

            'seq_len': int(self.seq_len),
            'purge_gap': int(self.purge_gap),
            'use_validity_mask': bool(self.use_validity_mask),

            'base_feature_names': list(self.base_feature_names),
            'feature_names': list(feature_names),
            'fill_values': {k: float(v) for k, v in fill_values.items()},
            'mean': mean.astype(np.float32).tolist(),
            'std': std.astype(np.float32).tolist(),
            'class_name_to_id': dict(CLASS_NAME_TO_ID),
            'class_id_to_name': {str(k): v for k, v in CLASS_ID_TO_NAME.items()},
        }

        with open(self.out_dir / 'dataset_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self._write_debug_csv(train_rows, val_rows, test_rows)

        summary = {
            'out_dir': str(self.out_dir),
            'train_npz': str(self.out_dir / 'train.npz'),
            'val_npz': str(self.out_dir / 'val.npz'),
            'test_npz': str(self.out_dir / 'test.npz'),
            'dataset_stats_json': str(self.out_dir / 'dataset_stats.json'),
            'debug_csv': str(self.out_dir / self.debug_csv_name),
            'num_merged_rows': int(len(merged_rows)),
            'train_sequences': int(len(train_X_seq)),
            'val_sequences': int(len(val_X_seq)),
            'test_sequences': int(len(test_X_seq)),
            'seq_len': int(self.seq_len),
            'purge_gap': int(self.purge_gap),
            'use_validity_mask': bool(self.use_validity_mask),
        }

        print('========== Build Reliability Dataset v2 Finished ==========')
        for k, v in summary.items():
            print(f'{k}: {v}')

        return summary


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
        '--label_csv',
        type=str,
        default='',
        help='若留空，預設 sequence_dir/reliability_labels/reliability_labels.csv'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='若留空，預設輸出到 sequence_dir/reliability_dataset'
    )
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--min_rows', type=int, default=32)
    parser.add_argument(
        '--purge_gap',
        type=int,
        default=-1,
        help='若 < 0，則自動使用 seq_len - 1'
    )
    parser.add_argument(
        '--disable_validity_mask',
        action='store_true',
        help='若加上這個參數，則不附加 valid_xxx mask 欄位'
    )
    parser.add_argument(
        '--require_label_cls',
        action='store_true',
        help='若加上，則要求 label_csv 中必須有有效 label_cls'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    purge_gap = None if args.purge_gap < 0 else int(args.purge_gap)

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
        use_validity_mask=not args.disable_validity_mask,
        require_label_cls=args.require_label_cls,
    )
    builder.build()


if __name__ == '__main__':
    main()