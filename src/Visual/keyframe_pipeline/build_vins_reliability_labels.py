import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


CLASS_NAME_TO_ID = {
    'harmful': 0,
    'neutral': 1,
    'helpful': 2,
}

CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


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


def clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def robust_scale(values: List[float], default_value: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr >= 0.0)]
    if len(arr) == 0:
        return float(default_value)
    q75 = float(np.percentile(arr, 75))
    q90 = float(np.percentile(arr, 90))
    return max(q75, q90 * 0.5, float(default_value), 1e-6)


def quat_xyzw_to_rotmat(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = q / n
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)


def rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    qy = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    qz = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    qx, qy, qz, qw = q / n
    return float(qx), float(qy), float(qz), float(qw)


def rotation_angle_deg(R: np.ndarray) -> float:
    trace_val = np.trace(R)
    c = float(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(c))


def rotation_distance_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return rotation_angle_deg(Ra.T @ Rb)


def read_csv_rows(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_feature_rows(path: Path):
    rows = read_csv_rows(path)
    parsed = []
    for row in rows:
        update_id = to_int(row.get('update_id', -1), -1)
        if update_id < 0:
            continue
        parsed.append({
            'pair_id': update_id,
            'update_id': update_id,
            'timestamp': to_float(row.get('timestamp', np.nan), np.nan),
            'failure_detected_last': to_int(row.get('failure_detected_last', 0), 0),
            'outlier_ratio_last': to_float(row.get('outlier_ratio_last', 0.0), 0.0),
            'est_p_x': to_float(row.get('est_p_x', np.nan), np.nan),
            'est_p_y': to_float(row.get('est_p_y', np.nan), np.nan),
            'est_p_z': to_float(row.get('est_p_z', np.nan), np.nan),
            'est_q_x': to_float(row.get('est_q_x', np.nan), np.nan),
            'est_q_y': to_float(row.get('est_q_y', np.nan), np.nan),
            'est_q_z': to_float(row.get('est_q_z', np.nan), np.nan),
            'est_q_w': to_float(row.get('est_q_w', np.nan), np.nan),
        })
    parsed.sort(key=lambda x: x['pair_id'])
    return parsed


def read_tum_rows(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            t, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            rows.append({
                'timestamp': t,
                'p': np.array([tx, ty, tz], dtype=np.float64),
                'R': quat_xyzw_to_rotmat(qx, qy, qz, qw),
            })
    rows.sort(key=lambda x: x['timestamp'])
    return rows


def read_gt_rows(path: Path):
    if path.suffix.lower() == '.tum':
        return read_tum_rows(path)

    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = to_float(row.get('timestamp', np.nan), np.nan)
            if not np.isfinite(t):
                continue
            px = to_float(row.get('pos_x', np.nan), np.nan)
            py = to_float(row.get('pos_y', np.nan), np.nan)
            pz = to_float(row.get('pos_z', np.nan), np.nan)
            qx = to_float(row.get('quat_x', np.nan), np.nan)
            qy = to_float(row.get('quat_y', np.nan), np.nan)
            qz = to_float(row.get('quat_z', np.nan), np.nan)
            qw = to_float(row.get('quat_w', np.nan), np.nan)
            if not np.all(np.isfinite([px, py, pz, qx, qy, qz, qw])):
                continue
            rows.append({
                'timestamp': t,
                'p': np.array([px, py, pz], dtype=np.float64),
                'R': quat_xyzw_to_rotmat(qx, qy, qz, qw),
            })
    rows.sort(key=lambda x: x['timestamp'])
    return rows


def shift_gt_rows(gt_rows, gt_time_shift_sec: float):
    if abs(gt_time_shift_sec) < 1e-12:
        return gt_rows
    out = []
    for row in gt_rows:
        out.append({
            'timestamp': float(row['timestamp']) + float(gt_time_shift_sec),
            'p': row['p'],
            'R': row['R'],
        })
    out.sort(key=lambda x: x['timestamp'])
    return out


def nearest_gt_index(gt_rows: List[Dict[str, object]], t: float, hint_idx: int = 0) -> Tuple[int, float]:
    if len(gt_rows) == 0:
        return -1, np.inf
    hint_idx = max(0, min(len(gt_rows) - 1, hint_idx))
    best_idx = hint_idx
    best_dt = abs(gt_rows[best_idx]['timestamp'] - t)

    i = hint_idx - 1
    while i >= 0:
        dt = abs(gt_rows[i]['timestamp'] - t)
        if dt <= best_dt:
            best_dt = dt
            best_idx = i
            i -= 1
        else:
            break

    i = hint_idx + 1
    while i < len(gt_rows):
        dt = abs(gt_rows[i]['timestamp'] - t)
        if dt < best_dt:
            best_dt = dt
            best_idx = i
            i += 1
        else:
            break

    return best_idx, float(best_dt)


def row_est_pose(row):
    p = np.array([row['est_p_x'], row['est_p_y'], row['est_p_z']], dtype=np.float64)
    R = quat_xyzw_to_rotmat(row['est_q_x'], row['est_q_y'], row['est_q_z'], row['est_q_w'])
    return p, R


def align_gt_rows_to_vins(gt_rows: List[Dict[str, object]], gt_ref_idx: int, feature_ref_row: Dict[str, object]):
    p_est0, R_est0 = row_est_pose(feature_ref_row)
    p_gt0 = gt_rows[gt_ref_idx]['p']
    R_gt0 = gt_rows[gt_ref_idx]['R']

    R_align = R_est0 @ R_gt0.T
    p_align = p_est0 - R_align @ p_gt0

    aligned = []
    for row in gt_rows:
        p_new = R_align @ row['p'] + p_align
        R_new = R_align @ row['R']
        aligned.append({
            'timestamp': row['timestamp'],
            'p': p_new,
            'R': R_new,
        })
    return aligned, p_align, R_align


def compute_row_errors(feature_row: Dict[str, object], gt_row: Dict[str, object]) -> Tuple[float, float]:
    p_est, R_est = row_est_pose(feature_row)
    pos_err = float(np.linalg.norm(p_est - gt_row['p']))
    rot_err = float(rotation_distance_deg(R_est, gt_row['R']))
    return pos_err, rot_err


class VinsReliabilityLabelBuilder:
    def __init__(
        self,
        feature_csv: str,
        gt_path: str,
        out_dir: str,
        horizon_rows: int = 10,
        max_match_dt_sec: float = 0.02,
        helpful_thr: float = 0.65,
        harmful_thr: float = 0.40,
        pos_weight: float = 0.60,
        rot_weight: float = 0.30,
        outlier_weight: float = 0.10,
        gt_time_shift_sec: float = 0.0,
        auto_shift_to_first_feature: bool = False,
    ):
        self.feature_csv = Path(feature_csv).expanduser().resolve()
        self.gt_path = Path(gt_path).expanduser().resolve()
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_csv = self.out_dir / 'reliability_labels_vins.csv'
        self.summary_json = self.out_dir / 'reliability_labels_summary.json'

        self.horizon_rows = int(horizon_rows)
        self.max_match_dt_sec = float(max_match_dt_sec)
        self.helpful_thr = float(helpful_thr)
        self.harmful_thr = float(harmful_thr)
        self.pos_weight = float(pos_weight)
        self.rot_weight = float(rot_weight)
        self.outlier_weight = float(outlier_weight)
        self.gt_time_shift_sec = float(gt_time_shift_sec)
        self.auto_shift_to_first_feature = bool(auto_shift_to_first_feature)

    def run(self):
        feature_rows = read_feature_rows(self.feature_csv)
        gt_rows_raw = read_gt_rows(self.gt_path)
        if len(feature_rows) == 0:
            raise RuntimeError(f'feature_csv 沒有有效資料: {self.feature_csv}')
        if len(gt_rows_raw) == 0:
            raise RuntimeError(f'gt_path 沒有有效資料: {self.gt_path}')

        auto_shift_estimate = float(feature_rows[0]['timestamp'] - gt_rows_raw[0]['timestamp'])
        applied_gt_shift = self.gt_time_shift_sec
        if self.auto_shift_to_first_feature:
            applied_gt_shift += auto_shift_estimate
        gt_rows_shifted = shift_gt_rows(gt_rows_raw, applied_gt_shift)

        # first pass: find reference matched pair for rigid alignment
        gt_hint = 0
        ref_feature_idx = -1
        ref_gt_idx = -1
        ref_dt = np.inf
        for i, row in enumerate(feature_rows):
            idx, dt = nearest_gt_index(gt_rows_shifted, row['timestamp'], gt_hint)
            if idx >= 0:
                gt_hint = idx
            if idx >= 0 and dt <= self.max_match_dt_sec:
                ref_feature_idx = i
                ref_gt_idx = idx
                ref_dt = dt
                break

        if ref_feature_idx < 0 or ref_gt_idx < 0:
            raise RuntimeError(
                '找不到任何 valid GT match。\n'
                '請先檢查時間軸，或增加 --max_match_dt_sec，或指定 --gt_time_shift_sec / --auto_shift_to_first_feature。'
            )

        gt_rows_aligned, p_align, R_align = align_gt_rows_to_vins(
            gt_rows_shifted,
            gt_ref_idx=ref_gt_idx,
            feature_ref_row=feature_rows[ref_feature_idx],
        )

        # second pass: align GT and compute instant errors in the aligned local frame
        gt_hint = ref_gt_idx
        aligned_rows = []
        instant_pos_errors = []
        instant_rot_errors = []
        outlier_ratios = []
        matched_dts = []
        for row in feature_rows:
            idx, dt = nearest_gt_index(gt_rows_aligned, row['timestamp'], gt_hint)
            if idx >= 0:
                gt_hint = idx
            matched = (idx >= 0 and dt <= self.max_match_dt_sec)
            gt_pos_err = np.nan
            gt_rot_err = np.nan
            if matched:
                gt_pos_err, gt_rot_err = compute_row_errors(row, gt_rows_aligned[idx])
                instant_pos_errors.append(gt_pos_err)
                instant_rot_errors.append(gt_rot_err)
                matched_dts.append(dt)
            outlier_ratios.append(float(row['outlier_ratio_last']))
            aligned_rows.append({
                **row,
                'gt_match_idx': idx,
                'gt_match_dt_sec': dt,
                'gt_matched': bool(matched),
                'gt_pos_err_m': gt_pos_err,
                'gt_rot_err_deg': gt_rot_err,
            })

        pos_scale = robust_scale(instant_pos_errors, default_value=1.0)
        rot_scale = robust_scale(instant_rot_errors, default_value=10.0)
        outlier_scale = robust_scale(outlier_ratios, default_value=0.05)

        rows_out = []
        num_valid = 0
        class_counts = {'harmful': 0, 'neutral': 0, 'helpful': 0, 'invalid': 0}

        for i, row in enumerate(aligned_rows):
            end_idx = min(len(aligned_rows), i + self.horizon_rows)
            window = aligned_rows[i:end_idx]

            valid_pos = [r['gt_pos_err_m'] for r in window if r['gt_matched'] and np.isfinite(r['gt_pos_err_m'])]
            valid_rot = [r['gt_rot_err_deg'] for r in window if r['gt_matched'] and np.isfinite(r['gt_rot_err_deg'])]

            future_pos_mean = float(np.mean(valid_pos)) if len(valid_pos) > 0 else np.nan
            future_rot_mean = float(np.mean(valid_rot)) if len(valid_rot) > 0 else np.nan

            outlier_ratio = float(row['outlier_ratio_last']) if np.isfinite(row['outlier_ratio_last']) else 1.0
            failure_flag = int(row['failure_detected_last'])

            if len(valid_pos) == 0 or len(valid_rot) == 0:
                label_reg = np.nan
                label_cls = -1
                class_name = 'invalid'
                label_source = 'future_gt_proxy_v2_missing_gt'
                class_counts['invalid'] += 1
            else:
                score_pos = math.exp(-future_pos_mean / max(pos_scale, 1e-6))
                score_rot = math.exp(-future_rot_mean / max(rot_scale, 1e-6))
                score_outlier = math.exp(-outlier_ratio / max(outlier_scale, 1e-6))
                score_fail = 0.0 if failure_flag else 1.0

                label_reg = clip01(
                    self.pos_weight * score_pos +
                    self.rot_weight * score_rot +
                    self.outlier_weight * score_outlier
                )
                label_reg = clip01(label_reg * (0.25 + 0.75 * score_fail))

                if label_reg >= self.helpful_thr:
                    class_name = 'helpful'
                elif label_reg <= self.harmful_thr:
                    class_name = 'harmful'
                else:
                    class_name = 'neutral'
                label_cls = CLASS_NAME_TO_ID[class_name]
                label_source = 'future_gt_proxy_v2_aligned'
                class_counts[class_name] += 1
                num_valid += 1

            rows_out.append({
                'pair_id': int(row['pair_id']),
                'update_id': int(row['update_id']),
                'timestamp': float(row['timestamp']),
                'label_reg': float(label_reg) if np.isfinite(label_reg) else np.nan,
                'label_cls': int(label_cls),
                'class_name': class_name,
                'label_source': label_source,
                'gt_match_dt_sec': float(row['gt_match_dt_sec']),
                'gt_pos_err_m': float(row['gt_pos_err_m']) if np.isfinite(row['gt_pos_err_m']) else np.nan,
                'gt_rot_err_deg': float(row['gt_rot_err_deg']) if np.isfinite(row['gt_rot_err_deg']) else np.nan,
                'future_pos_mean_m': float(future_pos_mean) if np.isfinite(future_pos_mean) else np.nan,
                'future_rot_mean_deg': float(future_rot_mean) if np.isfinite(future_rot_mean) else np.nan,
                'outlier_ratio_last': float(outlier_ratio),
                'failure_detected_last': int(failure_flag),
            })

        with open(self.out_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'pair_id', 'update_id', 'timestamp', 'label_reg', 'label_cls', 'class_name', 'label_source',
                'gt_match_dt_sec', 'gt_pos_err_m', 'gt_rot_err_deg',
                'future_pos_mean_m', 'future_rot_mean_deg',
                'outlier_ratio_last', 'failure_detected_last',
            ])
            writer.writeheader()
            for row in rows_out:
                writer.writerow(row)

        align_qx, align_qy, align_qz, align_qw = rotmat_to_quat_xyzw(R_align)
        summary = {
            'feature_csv': str(self.feature_csv),
            'gt_path': str(self.gt_path),
            'out_csv': str(self.out_csv),
            'num_rows': len(rows_out),
            'num_valid_labels': int(num_valid),
            'class_counts': class_counts,
            'horizon_rows': self.horizon_rows,
            'max_match_dt_sec': self.max_match_dt_sec,
            'pos_scale': pos_scale,
            'rot_scale': rot_scale,
            'outlier_scale': outlier_scale,
            'helpful_thr': self.helpful_thr,
            'harmful_thr': self.harmful_thr,
            'label_source': 'future_gt_proxy_v2_aligned',
            'note': 'GT 先做時間 shift，再用第一個有效 matched pair rigid-align 到 VINS local frame。',
            'feature_first_timestamp': float(feature_rows[0]['timestamp']),
            'feature_last_timestamp': float(feature_rows[-1]['timestamp']),
            'gt_first_timestamp_raw': float(gt_rows_raw[0]['timestamp']),
            'gt_last_timestamp_raw': float(gt_rows_raw[-1]['timestamp']),
            'auto_shift_estimate_from_first_timestamps': auto_shift_estimate,
            'applied_gt_time_shift_sec': applied_gt_shift,
            'reference_feature_idx': int(ref_feature_idx),
            'reference_gt_idx': int(ref_gt_idx),
            'reference_match_dt_sec': float(ref_dt),
            'matched_dt_mean_sec': float(np.mean(matched_dts)) if len(matched_dts) > 0 else np.nan,
            'matched_dt_median_sec': float(np.median(matched_dts)) if len(matched_dts) > 0 else np.nan,
            'R_align_quat_xyzw': [align_qx, align_qy, align_qz, align_qw],
            'p_align_xyz': p_align.astype(float).tolist(),
        }
        with open(self.summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f'[OK] labels -> {self.out_csv}')
        print(f'[OK] summary -> {self.summary_json}')
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--feature_csv', type=str, required=True,
                   help='例如: /home/ivlab3/vins_project/results/vins_eval_results/KAIST_urban28_pankyo/reliability_features_vins.csv')
    p.add_argument('--gt_path', type=str, required=True,
                   help='GT 路徑，支援 .tum 或帶 timestamp,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w 的 csv')
    p.add_argument('--out_dir', type=str, required=True,
                   help='輸出資料夾，例如: /home/ivlab3/vins_project/results/vins_eval_results/KAIST_urban28_pankyo/reliability_labels_vins')
    p.add_argument('--horizon_rows', type=int, default=10)
    p.add_argument('--max_match_dt_sec', type=float, default=0.02)
    p.add_argument('--helpful_thr', type=float, default=0.65)
    p.add_argument('--harmful_thr', type=float, default=0.40)
    p.add_argument('--pos_weight', type=float, default=0.60)
    p.add_argument('--rot_weight', type=float, default=0.30)
    p.add_argument('--outlier_weight', type=float, default=0.10)
    p.add_argument('--gt_time_shift_sec', type=float, default=0.0,
                   help='先對 GT timestamp 加上的時間平移（秒）')
    p.add_argument('--auto_shift_to_first_feature', action='store_true',
                   help='用 feature 第一筆時間戳 - GT 第一筆時間戳，自動估計一個初始 GT 時間平移')
    return p.parse_args()


def main():
    args = parse_args()
    builder = VinsReliabilityLabelBuilder(
        feature_csv=args.feature_csv,
        gt_path=args.gt_path,
        out_dir=args.out_dir,
        horizon_rows=args.horizon_rows,
        max_match_dt_sec=args.max_match_dt_sec,
        helpful_thr=args.helpful_thr,
        harmful_thr=args.harmful_thr,
        pos_weight=args.pos_weight,
        rot_weight=args.rot_weight,
        outlier_weight=args.outlier_weight,
        gt_time_shift_sec=args.gt_time_shift_sec,
        auto_shift_to_first_feature=args.auto_shift_to_first_feature,
    )
    builder.run()


if __name__ == '__main__':
    main()
