import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from matcher import LightGlueMatcher
from keyframe_pipeline.extract_features import (
    load_camera_matrix,
    compute_odom_relative_pose,
    rotation_distance_deg,
    estimate_visual_geometry,
)


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
        help='若留空，預設 sequence_dir/features/local_visual_features.csv'
    )
    parser.add_argument(
        '--pred_csv',
        type=str,
        default='',
        help='若留空，預設 sequence_dir/reliability_inference/reliability_predictions.csv'
    )
    parser.add_argument(
        '--camera_info',
        type=str,
        default='',
        help='若留空，預設 sequence_dir/camera_info.yaml'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='若留空，預設輸出到 sequence_dir/local_backend'
    )
    parser.add_argument('--tau', type=float, default=0.5, help='hard gate 門檻')
    parser.add_argument('--warmup_w', type=float, default=0.0, help='前 seq_len-1 個 pair 沒有預測時的預設 w')
    parser.add_argument('--sim_rot_noise_deg', type=float, default=0.8, help='模擬 inertial 相對旋轉雜訊標準差(度)')
    parser.add_argument('--sim_trans_noise_m', type=float, default=0.03, help='模擬 inertial 相對平移雜訊標準差(公尺)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_matches_for_geometry', type=int, default=8)
    return parser.parse_args()


def read_csv_rows(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def read_keyframes_csv(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['keyframe_id'] = int(to_float(row.get('keyframe_id', -1), -1))
            row['source_frame_id'] = int(to_float(row.get('source_frame_id', -1), -1))
            row['image_stamp_sec'] = int(to_float(row.get('image_stamp_sec', 0), 0))
            row['image_stamp_nsec'] = int(to_float(row.get('image_stamp_nsec', 0), 0))

            row['pos_x'] = float(to_float(row.get('pos_x', 0.0), 0.0))
            row['pos_y'] = float(to_float(row.get('pos_y', 0.0), 0.0))
            row['pos_z'] = float(to_float(row.get('pos_z', 0.0), 0.0))

            row['quat_x'] = float(to_float(row.get('quat_x', 0.0), 0.0))
            row['quat_y'] = float(to_float(row.get('quat_y', 0.0), 0.0))
            row['quat_z'] = float(to_float(row.get('quat_z', 0.0), 0.0))
            row['quat_w'] = float(to_float(row.get('quat_w', 1.0), 1.0))

            row['yaw_deg'] = float(to_float(row.get('yaw_deg', 0.0), 0.0))
            row['image_file'] = row.get('image_file', '')
            row['timestamp_token'] = row.get('timestamp_token', '')
            rows.append(row)

    rows.sort(key=lambda x: x['keyframe_id'])
    return rows


def build_keyframe_index(keyframe_rows):
    return {row['keyframe_id']: row for row in keyframe_rows}


def parse_feature_rows(path: Path):
    rows = read_csv_rows(path)
    parsed = []
    for row in rows:
        parsed.append({
            'pair_id': int(to_float(row.get('pair_id', -1), -1)),
            'kf_prev_id': int(to_float(row.get('kf_prev_id', -1), -1)),
            'kf_curr_id': int(to_float(row.get('kf_curr_id', -1), -1)),
            'src_prev_id': int(to_float(row.get('src_prev_id', -1), -1)),
            'src_curr_id': int(to_float(row.get('src_curr_id', -1), -1)),
            'image_prev': row.get('image_prev', ''),
            'image_curr': row.get('image_curr', ''),
            'timestamp_prev': row.get('timestamp_prev', ''),
            'timestamp_curr': row.get('timestamp_curr', ''),
        })
    parsed.sort(key=lambda x: x['pair_id'])
    return parsed


def parse_prediction_csv(path: Path):
    pred_map = {}
    if not path.exists():
        return pred_map

    rows = read_csv_rows(path)
    for row in rows:
        end_pid = int(to_float(row.get('end_pair_id', -1), -1))
        if end_pid < 0:
            continue
        pred_map[end_pid] = {
            'w_pred': float(to_float(row.get('w_pred', 0.0), 0.0)),
            'gate_pass': int(to_float(row.get('gate_pass', 0), 0)),
            'soft_target_proxy': float(to_float(row.get('soft_target_proxy', np.nan), np.nan)),
        }
    return pred_map


def row_world_position(row):
    return np.array([row['pos_x'], row['pos_y'], row['pos_z']], dtype=np.float64)


def row_world_rotation(row):
    qx = row['quat_x']
    qy = row['quat_y']
    qz = row['quat_z']
    qw = row['quat_w']

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)

    qx, qy, qz, qw = q / n
    R = np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)
    return R


def normalize(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def vector_angle_deg(v0, v1):
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return -1.0
    c = float(np.dot(v0, v1) / (n0 * n1))
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def random_noisy_relative(R_gt_rel, t_gt_rel, rng, rot_noise_deg, trans_noise_m):
    noise_rvec = np.deg2rad(rng.normal(0.0, rot_noise_deg, size=3)).reshape(3, 1)
    R_noise, _ = cv2.Rodrigues(noise_rvec)
    t_noise = rng.normal(0.0, trans_noise_m, size=3)

    R_noisy = R_noise @ R_gt_rel
    t_noisy = t_gt_rel + t_noise

    rot_noise_mag_deg = float(np.linalg.norm(noise_rvec) * 180.0 / math.pi)
    trans_noise_mag_m = float(np.linalg.norm(t_noise))
    return R_noisy, t_noisy, rot_noise_mag_deg, trans_noise_mag_m


def interpolate_rotation(R0, R1, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return R0.copy()
    if alpha >= 1.0:
        return R1.copy()

    R_delta = R0.T @ R1
    rvec, _ = cv2.Rodrigues(R_delta)
    R_inc, _ = cv2.Rodrigues(rvec * alpha)
    return R0 @ R_inc


def select_alpha(mode, w_pred, pose_ok, tau):
    if not pose_ok:
        return 0.0

    if mode == 'inertial_only':
        return 0.0
    if mode == 'visual_always':
        return 1.0
    if mode == 'hard_gate':
        return 1.0 if w_pred >= tau else 0.0
    if mode == 'soft_weight':
        return float(np.clip(w_pred, 0.0, 1.0))
    if mode == 'gate_and_weight':
        return float(np.clip(w_pred, 0.0, 1.0)) if w_pred >= tau else 0.0

    raise ValueError(f'未知模式: {mode}')


def propagate_mode(pair_infos, keyframe_rows, mode, tau):
    gt0 = keyframe_rows[0]
    est_R = row_world_rotation(gt0)
    est_p = row_world_position(gt0)

    est_states = [(est_R.copy(), est_p.copy())]
    debug_rows = []

    for pair in pair_infos:
        alpha = select_alpha(mode, pair['w_pred'], pair['visual_pose_ok'], tau)

        R_rel = pair['R_inertial_rel']
        t_rel = pair['t_inertial_rel']

        if alpha > 0.0:
            R_rel = interpolate_rotation(pair['R_inertial_rel'], pair['R_visual_rel'], alpha)
            t_rel = (1.0 - alpha) * pair['t_inertial_rel'] + alpha * pair['t_visual_scaled']

        next_R = est_R @ R_rel
        next_p = est_p + est_R @ t_rel

        est_states.append((next_R.copy(), next_p.copy()))
        est_R = next_R
        est_p = next_p

        debug_rows.append({
            'pair_id': pair['pair_id'],
            'mode': mode,
            'alpha': alpha,
            'w_pred': pair['w_pred'],
            'pose_ok': int(pair['visual_pose_ok']),
        })

    return est_states, debug_rows


def evaluate_states(est_states, gt_rows):
    gt_positions = np.array([row_world_position(r) for r in gt_rows], dtype=np.float64)
    est_positions = np.array([p for _, p in est_states], dtype=np.float64)

    trans_errors = np.linalg.norm(est_positions - gt_positions, axis=1)

    rot_errors = []
    for (R_est, _), row_gt in zip(est_states, gt_rows):
        R_gt = row_world_rotation(row_gt)
        rot_errors.append(rotation_distance_deg(R_est, R_gt))
    rot_errors = np.array(rot_errors, dtype=np.float64)

    metrics = {
        'num_states': int(len(est_states)),
        'position_rmse_m': float(np.sqrt(np.mean(trans_errors ** 2))),
        'position_mean_m': float(np.mean(trans_errors)),
        'position_median_m': float(np.median(trans_errors)),
        'position_final_m': float(trans_errors[-1]),
        'rotation_mean_deg': float(np.mean(rot_errors)),
        'rotation_median_deg': float(np.median(rot_errors)),
        'rotation_final_deg': float(rot_errors[-1]),
    }
    return metrics, est_positions, gt_positions, trans_errors, rot_errors


def save_trajectory_csv(path, gt_rows, est_states, trans_errors, rot_errors):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'kf_id',
            'gt_x', 'gt_y', 'gt_z',
            'est_x', 'est_y', 'est_z',
            'trans_error_m',
            'rot_error_deg',
        ])

        for row_gt, (R_est, p_est), te, re in zip(gt_rows, est_states, trans_errors, rot_errors):
            p_gt = row_world_position(row_gt)
            writer.writerow([
                row_gt['keyframe_id'],
                p_gt[0], p_gt[1], p_gt[2],
                p_est[0], p_est[1], p_est[2],
                te,
                re,
            ])


def main():
    args = parse_args()

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    keyframes_csv = sequence_dir / 'keyframes' / 'keyframes.csv'
    keyframe_images_dir = sequence_dir / 'keyframes' / 'images'
    feature_csv = Path(args.feature_csv).expanduser().resolve() if args.feature_csv else sequence_dir / 'features' / 'local_visual_features.csv'
    pred_csv = Path(args.pred_csv).expanduser().resolve() if args.pred_csv else sequence_dir / 'reliability_inference' / 'reliability_predictions.csv'
    camera_info = Path(args.camera_info).expanduser().resolve() if args.camera_info else sequence_dir / 'camera_info.yaml'
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else sequence_dir / 'local_backend'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not keyframes_csv.exists():
        raise FileNotFoundError(f'找不到 {keyframes_csv}')
    if not feature_csv.exists():
        raise FileNotFoundError(f'找不到 {feature_csv}')
    if not keyframe_images_dir.exists():
        raise FileNotFoundError(f'找不到 {keyframe_images_dir}')

    keyframe_rows = read_keyframes_csv(keyframes_csv)
    kf_index = build_keyframe_index(keyframe_rows)
    feature_rows = parse_feature_rows(feature_csv)
    pred_map = parse_prediction_csv(pred_csv)

    if len(keyframe_rows) < 2:
        raise RuntimeError('keyframe 數量不足，至少要 2 張')

    # 先讀第一張圖拿 image shape；如果沒有 camera_info.yaml 就 fallback 成近似內參
    first_img_path = keyframe_images_dir / keyframe_rows[0]['image_file']
    first_img = cv2.imread(str(first_img_path))
    if first_img is None:
        raise RuntimeError(f'無法讀取第一張 keyframe 影像: {first_img_path}')
    K = load_camera_matrix(camera_info, image_shape=first_img.shape)

    matcher = LightGlueMatcher()
    rng = np.random.default_rng(args.seed)

    pair_infos = []

    for feat in feature_rows:
        kf_prev = kf_index.get(feat['kf_prev_id'], None)
        kf_curr = kf_index.get(feat['kf_curr_id'], None)

        if kf_prev is None or kf_curr is None:
            print(f'[WARN] 找不到 keyframe id: {feat["kf_prev_id"]}, {feat["kf_curr_id"]}，略過')
            continue

        img0_path = keyframe_images_dir / feat['image_prev']
        img1_path = keyframe_images_dir / feat['image_curr']

        if not img0_path.exists() or not img1_path.exists():
            print(f'[WARN] 找不到影像: {img0_path.name}, {img1_path.name}，略過')
            continue

        img0 = cv2.imread(str(img0_path))
        img1 = cv2.imread(str(img1_path))
        if img0 is None or img1 is None:
            print(f'[WARN] 影像讀取失敗: {img0_path.name}, {img1_path.name}，略過')
            continue

        R_gt_rel, t_gt_rel = compute_odom_relative_pose(kf_prev, kf_curr)

        R_inertial_rel, t_inertial_rel, rot_noise_mag_deg, trans_noise_mag_m = random_noisy_relative(
            R_gt_rel,
            t_gt_rel,
            rng=rng,
            rot_noise_deg=args.sim_rot_noise_deg,
            trans_noise_m=args.sim_trans_noise_m
        )

        match_result = matcher.match(img0, img1)
        geo = estimate_visual_geometry(
            match_result['mkpts0'],
            match_result['mkpts1'],
            K,
            min_matches_for_geometry=args.min_matches_for_geometry
        )

        pose_ok = bool(geo['pose_ok'])
        R_visual_rel = geo['R_vis'] if pose_ok else R_inertial_rel.copy()
        t_vis_unit = geo['t_vis_unit'] if pose_ok else np.zeros((3,), dtype=np.float64)

        # 目前單目相對平移只有方向，先借用 inertial 預測的平移大小
        t_visual_scaled = t_vis_unit * max(np.linalg.norm(t_inertial_rel), 1e-8) if pose_ok else t_inertial_rel.copy()

        pred = pred_map.get(feat['pair_id'], None)
        if pred is None:
            w_pred = float(args.warmup_w)
            gate_pass = int(w_pred >= args.tau)
            soft_target_proxy = np.nan
        else:
            w_pred = float(pred['w_pred'])
            gate_pass = int(pred['gate_pass'])
            soft_target_proxy = float(pred['soft_target_proxy'])

        visual_vs_gt_rot_deg = rotation_distance_deg(R_visual_rel, R_gt_rel) if pose_ok else -1.0
        visual_vs_gt_tdir_deg = vector_angle_deg(t_vis_unit, t_gt_rel) if pose_ok else -1.0

        pair_infos.append({
            'pair_id': feat['pair_id'],
            'kf_prev_id': feat['kf_prev_id'],
            'kf_curr_id': feat['kf_curr_id'],
            'src_prev_id': feat['src_prev_id'],
            'src_curr_id': feat['src_curr_id'],
            'w_pred': w_pred,
            'gate_pass': gate_pass,
            'soft_target_proxy': soft_target_proxy,
            'visual_pose_ok': pose_ok,

            'R_gt_rel': R_gt_rel,
            't_gt_rel': t_gt_rel,

            'R_inertial_rel': R_inertial_rel,
            't_inertial_rel': t_inertial_rel,

            'R_visual_rel': R_visual_rel,
            't_visual_scaled': t_visual_scaled,

            'match_num': int(match_result['num_matches']),
            'geo_inlier_ratio': float(geo['inlier_ratio']),
            'rot_noise_mag_deg': rot_noise_mag_deg,
            'trans_noise_mag_m': trans_noise_mag_m,
            'visual_vs_gt_rot_deg': visual_vs_gt_rot_deg,
            'visual_vs_gt_tdir_deg': visual_vs_gt_tdir_deg,
        })

    if len(pair_infos) == 0:
        raise RuntimeError('pair_infos 為空，無法進行 local backend')

    # 儲存 pair debug
    pair_debug_csv = out_dir / 'pair_debug.csv'
    with open(pair_debug_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pair_id',
            'kf_prev_id',
            'kf_curr_id',
            'src_prev_id',
            'src_curr_id',
            'w_pred',
            'gate_pass',
            'soft_target_proxy',
            'visual_pose_ok',
            'match_num',
            'geo_inlier_ratio',
            'rot_noise_mag_deg',
            'trans_noise_mag_m',
            'visual_vs_gt_rot_deg',
            'visual_vs_gt_tdir_deg',
        ])
        for p in pair_infos:
            writer.writerow([
                p['pair_id'],
                p['kf_prev_id'],
                p['kf_curr_id'],
                p['src_prev_id'],
                p['src_curr_id'],
                p['w_pred'],
                p['gate_pass'],
                p['soft_target_proxy'],
                int(p['visual_pose_ok']),
                p['match_num'],
                p['geo_inlier_ratio'],
                p['rot_noise_mag_deg'],
                p['trans_noise_mag_m'],
                p['visual_vs_gt_rot_deg'],
                p['visual_vs_gt_tdir_deg'],
            ])

    modes = [
        'inertial_only',
        'visual_always',
        'hard_gate',
        'soft_weight',
        'gate_and_weight',
    ]

    results = {}
    all_mode_debug = []

    for mode in modes:
        est_states, mode_debug = propagate_mode(pair_infos, keyframe_rows, mode=mode, tau=args.tau)
        metrics, est_positions, gt_positions, trans_errors, rot_errors = evaluate_states(est_states, keyframe_rows)

        traj_csv = out_dir / f'trajectory_{mode}.csv'
        save_trajectory_csv(traj_csv, keyframe_rows, est_states, trans_errors, rot_errors)

        results[mode] = {
            'metrics': metrics,
            'trajectory_csv': str(traj_csv),
        }

        all_mode_debug.extend(mode_debug)

    mode_debug_csv = out_dir / 'mode_debug.csv'
    with open(mode_debug_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pair_id', 'mode', 'alpha', 'w_pred', 'pose_ok'])
        for row in all_mode_debug:
            writer.writerow([
                row['pair_id'],
                row['mode'],
                row['alpha'],
                row['w_pred'],
                row['pose_ok'],
            ])

    summary = {
        'sequence_dir': str(sequence_dir),
        'feature_csv': str(feature_csv),
        'pred_csv': str(pred_csv),
        'camera_info': str(camera_info),
        'tau': float(args.tau),
        'warmup_w': float(args.warmup_w),
        'sim_rot_noise_deg': float(args.sim_rot_noise_deg),
        'sim_trans_noise_m': float(args.sim_trans_noise_m),
        'num_pairs_used': int(len(pair_infos)),
        'results': results,
        'notes': [
            '這版是離線 local backend / proxy backend，用來先驗證 GRU 的 w_k 對 local correction 是否有幫助。',
            '目前 inertial relative motion 是用 keyframe odom relative pose 加入模擬漂移噪聲而成，尚未接真正的 IMU preintegration factor graph。',
            '目前單目 visual relative translation 只有方向，因此 local fusion 時使用 inertial 預測的平移大小，視覺提供方向修正。',
        ]
    }

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'完成，local backend 輸出到: {out_dir}')
    print(f'pair debug: {pair_debug_csv}')
    print(f'mode debug: {mode_debug_csv}')
    print(f'summary: {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()