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
    estimate_visual_geometry,
)


# =========================================================
# 基本工具
# =========================================================

def wrap_angle_rad(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def rotz(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def yaw_from_row(row) -> float:
    return math.radians(float(row['yaw_deg']))


def row_world_position(row) -> np.ndarray:
    return np.array([row['pos_x'], row['pos_y'], row['pos_z']], dtype=np.float64)


def row_world_rotation_from_yaw(row) -> np.ndarray:
    return rotz(yaw_from_row(row))


def to_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def vector_angle_deg(v0: np.ndarray, v1: np.ndarray) -> float:
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return -1.0
    c = float(np.dot(v0, v1) / (n0 * n1))
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def rotation_angle_deg_from_R(R: np.ndarray) -> float:
    trace_val = np.trace(R)
    c = (trace_val - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def yaw_from_rotation_matrix(R: np.ndarray) -> float:
    # 只取平面 yaw，先做第一版 local backend
    return math.atan2(R[1, 0], R[0, 0])


def interpolate_yaw(yaw0: float, yaw1: float, alpha: float) -> float:
    dy = wrap_angle_rad(yaw1 - yaw0)
    return wrap_angle_rad(yaw0 + alpha * dy)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# =========================================================
# 讀檔
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_dir', type=str, required=True, help='例如: /mnt/sata4t/dataset/sequence_001')
    parser.add_argument('--pred_csv', type=str, default='', help='若留空，預設 sequence_dir/reliability_inference/reliability_predictions.csv')
    parser.add_argument('--camera_info', type=str, default='', help='若留空，預設 sequence_dir/camera_info.yaml')
    parser.add_argument('--out_dir', type=str, default='', help='若留空，預設 sequence_dir/local_factor_graph')
    parser.add_argument('--tau', type=float, default=0.5, help='hard gate 門檻')
    parser.add_argument('--warmup_w', type=float, default=0.0, help='前 seq_len-1 個 pair 沒有預測時使用的預設 w')

    # 模擬 inertial 漂移
    parser.add_argument('--sim_rot_noise_deg', type=float, default=0.8)
    parser.add_argument('--sim_trans_noise_m', type=float, default=0.03)

    # 因子權重 / 噪聲模型
    parser.add_argument('--sigma_prior_p', type=float, default=1e-3)
    parser.add_argument('--sigma_prior_yaw_deg', type=float, default=1e-3)
    parser.add_argument('--sigma_imu_p', type=float, default=0.05)
    parser.add_argument('--sigma_imu_yaw_deg', type=float, default=1.0)
    parser.add_argument('--sigma_vis_p', type=float, default=0.08)
    parser.add_argument('--sigma_vis_yaw_deg', type=float, default=3.0)

    parser.add_argument('--max_gn_iters', type=int, default=15)
    parser.add_argument('--fd_eps', type=float, default=1e-5)
    parser.add_argument('--damping', type=float, default=1e-6)
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


def read_keyframes_csv(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['keyframe_id'] = int(to_float(row.get('keyframe_id', -1), -1))
            row['source_frame_id'] = int(to_float(row.get('source_frame_id', -1), -1))
            row['pos_x'] = float(to_float(row.get('pos_x', 0.0), 0.0))
            row['pos_y'] = float(to_float(row.get('pos_y', 0.0), 0.0))
            row['pos_z'] = float(to_float(row.get('pos_z', 0.0), 0.0))
            row['yaw_deg'] = float(to_float(row.get('yaw_deg', 0.0), 0.0))
            row['image_file'] = row.get('image_file', '')
            row['timestamp_token'] = row.get('timestamp_token', '')
            row['quat_x'] = float(to_float(row.get('quat_x', 0.0), 0.0))
            row['quat_y'] = float(to_float(row.get('quat_y', 0.0), 0.0))
            row['quat_z'] = float(to_float(row.get('quat_z', 0.0), 0.0))
            row['quat_w'] = float(to_float(row.get('quat_w', 1.0), 1.0))
            rows.append(row)

    rows.sort(key=lambda x: x['keyframe_id'])
    return rows


def read_prediction_csv(path: Path):
    pred_map = {}
    if not path.exists():
        return pred_map

    rows = read_csv_rows(path)
    for row in rows:
        end_pair_id = int(to_float(row.get('end_pair_id', -1), -1))
        if end_pair_id < 0:
            continue

        pred_map[end_pair_id] = {
            'w_pred': float(to_float(row.get('w_pred', 0.0), 0.0)),
            'gate_pass': int(to_float(row.get('gate_pass', 0), 0)),
            'soft_target_proxy': float(to_float(row.get('soft_target_proxy', np.nan), np.nan)),
        }
    return pred_map


# =========================================================
# 狀態表示
# 每個 keyframe state = [px, py, pz, yaw]
# =========================================================

def states_to_vector(states):
    # states: list of dict {'p': (3,), 'yaw': float}
    xs = []
    for s in states:
        xs.extend([s['p'][0], s['p'][1], s['p'][2], s['yaw']])
    return np.array(xs, dtype=np.float64)


def vector_to_states(x):
    assert len(x) % 4 == 0
    n = len(x) // 4
    states = []
    for i in range(n):
        px, py, pz, yaw = x[4 * i: 4 * i + 4]
        states.append({
            'p': np.array([px, py, pz], dtype=np.float64),
            'yaw': float(wrap_angle_rad(yaw)),
        })
    return states


def relative_pose_pred(state_i, state_j):
    """
    以 state_i 為 local frame，預測 i->j 的相對位姿
    """
    Ri = rotz(state_i['yaw'])
    dp_world = state_j['p'] - state_i['p']
    dp_local = Ri.T @ dp_world
    dyaw = wrap_angle_rad(state_j['yaw'] - state_i['yaw'])
    return dp_local, dyaw


# =========================================================
# 建立測量
# =========================================================

def build_initial_states_from_gt(gt_rows):
    states = []
    for row in gt_rows:
        states.append({
            'p': row_world_position(row).copy(),
            'yaw': yaw_from_row(row),
        })
    return states


def build_noisy_inertial_measurements(gt_rows, rng, sim_rot_noise_deg, sim_trans_noise_m):
    """
    用 keyframe GT 相對位姿 + 模擬噪聲，當作第一版 inertial/pseudo-FOG proxy
    """
    meas = []
    for i in range(len(gt_rows) - 1):
        row0 = gt_rows[i]
        row1 = gt_rows[i + 1]

        R_rel_gt, t_rel_gt = compute_odom_relative_pose(row0, row1)
        yaw_rel_gt = yaw_from_rotation_matrix(R_rel_gt)

        yaw_rel_meas = wrap_angle_rad(yaw_rel_gt + math.radians(rng.normal(0.0, sim_rot_noise_deg)))
        t_rel_meas = t_rel_gt + rng.normal(0.0, sim_trans_noise_m, size=3)

        meas.append({
            'pair_id': i,
            'i': i,
            'j': i + 1,
            't_gt': t_rel_gt,
            'yaw_gt': yaw_rel_gt,
            't_meas': t_rel_meas,
            'yaw_meas': yaw_rel_meas,
        })
    return meas


def build_visual_measurements(sequence_dir: Path, gt_rows, pred_map, camera_info_path: Path, warmup_w: float, tau: float, min_matches_for_geometry: int):
    keyframe_images_dir = sequence_dir / 'keyframes' / 'images'
    matcher = LightGlueMatcher()

    first_img = cv2.imread(str(keyframe_images_dir / gt_rows[0]['image_file']))
    if first_img is None:
        raise RuntimeError('無法讀取第一張 keyframe 影像，無法取得相機內參 fallback')
    K = load_camera_matrix(camera_info_path, image_shape=first_img.shape)

    visual_pairs = []

    for i in range(len(gt_rows) - 1):
        row0 = gt_rows[i]
        row1 = gt_rows[i + 1]

        img0_path = keyframe_images_dir / row0['image_file']
        img1_path = keyframe_images_dir / row1['image_file']

        img0 = cv2.imread(str(img0_path))
        img1 = cv2.imread(str(img1_path))
        if img0 is None or img1 is None:
            raise RuntimeError(f'讀取影像失敗: {img0_path}, {img1_path}')

        match_result = matcher.match(img0, img1)
        geo = estimate_visual_geometry(
            match_result['mkpts0'],
            match_result['mkpts1'],
            K,
            min_matches_for_geometry=min_matches_for_geometry
        )

        pred = pred_map.get(i, None)
        if pred is None:
            w_pred = float(warmup_w)
            gate_pass = int(w_pred >= tau)
            soft_target_proxy = np.nan
        else:
            w_pred = float(pred['w_pred'])
            gate_pass = int(pred['gate_pass'])
            soft_target_proxy = float(pred['soft_target_proxy'])

        pose_ok = bool(geo['pose_ok'])
        if pose_ok:
            R_vis = geo['R_vis']
            yaw_vis = yaw_from_rotation_matrix(R_vis)
            t_vis_dir = normalize(geo['t_vis_unit'])
        else:
            yaw_vis = 0.0
            t_vis_dir = np.zeros((3,), dtype=np.float64)

        visual_pairs.append({
            'pair_id': i,
            'i': i,
            'j': i + 1,
            'pose_ok': pose_ok,
            'num_matches': int(match_result['num_matches']),
            'inlier_ratio': float(geo['inlier_ratio']),
            'vis_rot_deg': float(geo['vis_rot_deg']),
            'yaw_vis': float(yaw_vis),
            't_vis_dir': t_vis_dir,
            'w_pred': w_pred,
            'gate_pass': gate_pass,
            'soft_target_proxy': soft_target_proxy,
        })

    return visual_pairs


# =========================================================
# 殘差與優化
# =========================================================

def select_alpha(mode: str, w_pred: float, pose_ok: bool, tau: float) -> float:
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


def build_residual_vector(
    x_vec,
    gt_rows,
    inertial_meas,
    visual_meas,
    mode,
    tau,
    sigma_prior_p,
    sigma_prior_yaw_rad,
    sigma_imu_p,
    sigma_imu_yaw_rad,
    sigma_vis_p,
    sigma_vis_yaw_rad,
):
    states = vector_to_states(x_vec)
    residuals = []

    # 1) first pose prior
    gt0 = gt_rows[0]
    p0_gt = row_world_position(gt0)
    yaw0_gt = yaw_from_row(gt0)

    r_prior_p = (states[0]['p'] - p0_gt) / sigma_prior_p
    r_prior_yaw = np.array([wrap_angle_rad(states[0]['yaw'] - yaw0_gt) / sigma_prior_yaw_rad], dtype=np.float64)

    residuals.append(r_prior_p)
    residuals.append(r_prior_yaw)

    # 2) inertial factors
    for meas in inertial_meas:
        i = meas['i']
        j = meas['j']

        dp_pred, dyaw_pred = relative_pose_pred(states[i], states[j])

        r_p = (dp_pred - meas['t_meas']) / sigma_imu_p
        r_yaw = np.array([wrap_angle_rad(dyaw_pred - meas['yaw_meas']) / sigma_imu_yaw_rad], dtype=np.float64)

        residuals.append(r_p)
        residuals.append(r_yaw)

    # 3) visual factors
    for vmeas, imeas in zip(visual_meas, inertial_meas):
        i = vmeas['i']
        j = vmeas['j']

        alpha = select_alpha(mode, vmeas['w_pred'], vmeas['pose_ok'], tau)
        if alpha <= 0.0:
            continue

        dp_pred, dyaw_pred = relative_pose_pred(states[i], states[j])

        # 單目平移只有方向，先借用 inertial 量測的平移大小，視覺只修正方向
        t_mag = np.linalg.norm(imeas['t_meas'])
        t_vis = vmeas['t_vis_dir'] * max(t_mag, 1e-8)

        weight_scale = math.sqrt(max(alpha, 1e-8))
        r_p_vis = weight_scale * (dp_pred - t_vis) / sigma_vis_p
        r_yaw_vis = np.array([
            weight_scale * wrap_angle_rad(dyaw_pred - vmeas['yaw_vis']) / sigma_vis_yaw_rad
        ], dtype=np.float64)

        residuals.append(r_p_vis)
        residuals.append(r_yaw_vis)

    if len(residuals) == 0:
        return np.zeros((0,), dtype=np.float64)

    return np.concatenate(residuals, axis=0)


def numeric_jacobian(fun, x, eps):
    r0 = fun(x)
    J = np.zeros((len(r0), len(x)), dtype=np.float64)

    for k in range(len(x)):
        x1 = x.copy()
        x1[k] += eps
        r1 = fun(x1)
        J[:, k] = (r1 - r0) / eps

    return J, r0


def gauss_newton_optimize(
    x0,
    residual_fun,
    max_iters=15,
    fd_eps=1e-5,
    damping=1e-6,
):
    x = x0.copy()
    history = []

    for it in range(max_iters):
        J, r = numeric_jacobian(residual_fun, x, fd_eps)
        cost = 0.5 * float(np.dot(r, r))

        H = J.T @ J + damping * np.eye(J.shape[1], dtype=np.float64)
        g = J.T @ r

        try:
            dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        x_new = x + dx
        r_new = residual_fun(x_new)
        cost_new = 0.5 * float(np.dot(r_new, r_new))

        history.append({
            'iter': it,
            'cost': cost,
            'cost_new': cost_new,
            'step_norm': float(np.linalg.norm(dx)),
        })

        if cost_new < cost:
            x = x_new
            if np.linalg.norm(dx) < 1e-8:
                break
        else:
            # 如果變差就停
            break

    return x, history


# =========================================================
# 評估
# =========================================================

def evaluate_states(est_states, gt_rows):
    gt_positions = np.array([row_world_position(r) for r in gt_rows], dtype=np.float64)
    est_positions = np.array([s['p'] for s in est_states], dtype=np.float64)

    trans_errors = np.linalg.norm(est_positions - gt_positions, axis=1)

    rot_errors_deg = []
    for s, gt in zip(est_states, gt_rows):
        yaw_gt = yaw_from_row(gt)
        dyaw = wrap_angle_rad(s['yaw'] - yaw_gt)
        rot_errors_deg.append(abs(math.degrees(dyaw)))
    rot_errors_deg = np.array(rot_errors_deg, dtype=np.float64)

    metrics = {
        'num_states': int(len(est_states)),
        'position_rmse_m': float(np.sqrt(np.mean(trans_errors ** 2))),
        'position_mean_m': float(np.mean(trans_errors)),
        'position_median_m': float(np.median(trans_errors)),
        'position_final_m': float(trans_errors[-1]),
        'rotation_mean_deg': float(np.mean(rot_errors_deg)),
        'rotation_median_deg': float(np.median(rot_errors_deg)),
        'rotation_final_deg': float(rot_errors_deg[-1]),
    }
    return metrics, gt_positions, est_positions, trans_errors, rot_errors_deg


def save_trajectory_csv(path: Path, gt_rows, est_states, trans_errors, rot_errors_deg):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'kf_id',
            'gt_x', 'gt_y', 'gt_z', 'gt_yaw_deg',
            'est_x', 'est_y', 'est_z', 'est_yaw_deg',
            'trans_error_m',
            'yaw_error_deg'
        ])

        for gt, est, te, re in zip(gt_rows, est_states, trans_errors, rot_errors_deg):
            p_gt = row_world_position(gt)
            yaw_gt_deg = gt['yaw_deg']
            writer.writerow([
                gt['keyframe_id'],
                p_gt[0], p_gt[1], p_gt[2], yaw_gt_deg,
                est['p'][0], est['p'][1], est['p'][2], math.degrees(est['yaw']),
                te,
                re,
            ])


# =========================================================
# 主程式
# =========================================================

def main():
    args = parse_args()

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    keyframes_csv = sequence_dir / 'keyframes' / 'keyframes.csv'
    pred_csv = Path(args.pred_csv).expanduser().resolve() if args.pred_csv else sequence_dir / 'reliability_inference' / 'reliability_predictions.csv'
    camera_info_path = Path(args.camera_info).expanduser().resolve() if args.camera_info else sequence_dir / 'camera_info.yaml'
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else sequence_dir / 'local_factor_graph'

    ensure_dir(out_dir)

    if not keyframes_csv.exists():
        raise FileNotFoundError(f'找不到 {keyframes_csv}')

    gt_rows = read_keyframes_csv(keyframes_csv)
    if len(gt_rows) < 2:
        raise RuntimeError('keyframe 數量不足，至少要 2 張')

    rng = np.random.default_rng(args.seed)
    pred_map = read_prediction_csv(pred_csv)

    inertial_meas = build_noisy_inertial_measurements(
        gt_rows=gt_rows,
        rng=rng,
        sim_rot_noise_deg=args.sim_rot_noise_deg,
        sim_trans_noise_m=args.sim_trans_noise_m,
    )

    visual_meas = build_visual_measurements(
        sequence_dir=sequence_dir,
        gt_rows=gt_rows,
        pred_map=pred_map,
        camera_info_path=camera_info_path,
        warmup_w=args.warmup_w,
        tau=args.tau,
        min_matches_for_geometry=args.min_matches_for_geometry,
    )

    # 初始值：用 noisy inertial propagation 建立
    init_states = [build_initial_states_from_gt(gt_rows)[0]]
    for k in range(len(inertial_meas)):
        prev = init_states[-1]
        meas = inertial_meas[k]

        R_prev = rotz(prev['yaw'])
        p_next = prev['p'] + R_prev @ meas['t_meas']
        yaw_next = wrap_angle_rad(prev['yaw'] + meas['yaw_meas'])

        init_states.append({
            'p': p_next,
            'yaw': yaw_next,
        })

    x0 = states_to_vector(init_states)

    sigma_prior_yaw_rad = math.radians(args.sigma_prior_yaw_deg)
    sigma_imu_yaw_rad = math.radians(args.sigma_imu_yaw_deg)
    sigma_vis_yaw_rad = math.radians(args.sigma_vis_yaw_deg)

    modes = [
        'inertial_only',
        'visual_always',
        'hard_gate',
        'soft_weight',
        'gate_and_weight',
    ]

    all_results = {}
    all_iter_rows = []
    all_factor_rows = []

    for mode in modes:
        def residual_fun(x):
            return build_residual_vector(
                x_vec=x,
                gt_rows=gt_rows,
                inertial_meas=inertial_meas,
                visual_meas=visual_meas,
                mode=mode,
                tau=args.tau,
                sigma_prior_p=args.sigma_prior_p,
                sigma_prior_yaw_rad=sigma_prior_yaw_rad,
                sigma_imu_p=args.sigma_imu_p,
                sigma_imu_yaw_rad=sigma_imu_yaw_rad,
                sigma_vis_p=args.sigma_vis_p,
                sigma_vis_yaw_rad=sigma_vis_yaw_rad,
            )

        x_opt, history = gauss_newton_optimize(
            x0=x0,
            residual_fun=residual_fun,
            max_iters=args.max_gn_iters,
            fd_eps=args.fd_eps,
            damping=args.damping,
        )

        est_states = vector_to_states(x_opt)
        metrics, gt_positions, est_positions, trans_errors, rot_errors_deg = evaluate_states(est_states, gt_rows)

        traj_csv = out_dir / f'trajectory_{mode}.csv'
        save_trajectory_csv(traj_csv, gt_rows, est_states, trans_errors, rot_errors_deg)

        all_results[mode] = {
            'metrics': metrics,
            'trajectory_csv': str(traj_csv),
            'num_iters': int(len(history)),
        }

        for row in history:
            all_iter_rows.append([
                mode,
                row['iter'],
                row['cost'],
                row['cost_new'],
                row['step_norm'],
            ])

        for v in visual_meas:
            alpha = select_alpha(mode, v['w_pred'], v['pose_ok'], args.tau)
            all_factor_rows.append([
                mode,
                v['pair_id'],
                v['i'],
                v['j'],
                v['pose_ok'],
                v['num_matches'],
                v['inlier_ratio'],
                v['w_pred'],
                alpha,
                v['gate_pass'],
                v['soft_target_proxy'],
            ])

    # 儲存 iteration history
    iter_csv = out_dir / 'optimization_history.csv'
    with open(iter_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'iter', 'cost', 'cost_new', 'step_norm'])
        writer.writerows(all_iter_rows)

    # 儲存 factor debug
    factor_csv = out_dir / 'factor_debug.csv'
    with open(factor_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mode',
            'pair_id',
            'i',
            'j',
            'pose_ok',
            'num_matches',
            'inlier_ratio',
            'w_pred',
            'alpha',
            'gate_pass',
            'soft_target_proxy',
        ])
        writer.writerows(all_factor_rows)

    summary = {
        'sequence_dir': str(sequence_dir),
        'pred_csv': str(pred_csv),
        'camera_info': str(camera_info_path),
        'tau': float(args.tau),
        'warmup_w': float(args.warmup_w),
        'sim_rot_noise_deg': float(args.sim_rot_noise_deg),
        'sim_trans_noise_m': float(args.sim_trans_noise_m),
        'sigma_prior_p': float(args.sigma_prior_p),
        'sigma_prior_yaw_deg': float(args.sigma_prior_yaw_deg),
        'sigma_imu_p': float(args.sigma_imu_p),
        'sigma_imu_yaw_deg': float(args.sigma_imu_yaw_deg),
        'sigma_vis_p': float(args.sigma_vis_p),
        'sigma_vis_yaw_deg': float(args.sigma_vis_yaw_deg),
        'num_keyframes': int(len(gt_rows)),
        'num_pairs': int(len(gt_rows) - 1),
        'results': all_results,
        'notes': [
            '這版已經是 factor-graph-style local backend：有 prior factor、inertial factor、visual factor。',
            '目前 state 先用 3D position + yaw，不是完整 SE(3)；目的是先驗證 w_k 是否真的對 local correction 有幫助。',
            '目前 inertial factor 仍是 odom/keyframe 相對運動 + 模擬噪聲，尚未接真正 IMU preintegration。',
            '目前 visual translation 仍採單目方向 + inertial 平移尺度，是第一版 local correction backend。'
        ]
    }

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'完成，local factor graph 輸出到: {out_dir}')
    print(f'summary: {out_dir / "summary.json"}')
    print(f'factor debug: {factor_csv}')
    print(f'optimization history: {iter_csv}')


if __name__ == '__main__':
    main()