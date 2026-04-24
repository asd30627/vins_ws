import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from matcher import LightGlueMatcher


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


# =========================================================
# 基本工具
# =========================================================

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


def clip01(x):
    return float(np.clip(float(x), 0.0, 1.0))


def sigmoid(x):
    x = float(np.clip(x, -50.0, 50.0))
    return 1.0 / (1.0 + math.exp(-x))


def robust_positive_scale(values, default_value):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values) & (values >= 0.0)]
    if len(valid) == 0:
        return float(default_value)
    q75 = np.percentile(valid, 75)
    q90 = np.percentile(valid, 90)
    return max(float(q75), float(q90) * 0.5, float(default_value))


def exp_good_from_error(value, scale, invalid_fill=0.0):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        return float(invalid_fill)
    return clip01(math.exp(-value / max(float(scale), 1e-6)))


def normalize(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def vector_angle_deg(v0, v1):
    v0 = np.asarray(v0, dtype=np.float64).reshape(-1)
    v1 = np.asarray(v1, dtype=np.float64).reshape(-1)

    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return -1.0

    c = float(np.dot(v0, v1) / (n0 * n1))
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def quat_xyzw_to_rotmat(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError('invalid quaternion norm')

    qx, qy, qz, qw = q / n
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)


def rotation_angle_deg(R):
    trace_val = np.trace(R)
    c = (trace_val - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def rotation_distance_deg(Ra, Rb):
    return rotation_angle_deg(Ra.T @ Rb)


def compute_translation_from_rows(row0, row1):
    required_keys = ['pos_x', 'pos_y', 'pos_z']
    if not all(k in row0 for k in required_keys):
        return None
    if not all(k in row1 for k in required_keys):
        return None

    return np.array([
        row1['pos_x'] - row0['pos_x'],
        row1['pos_y'] - row0['pos_y'],
        row1['pos_z'] - row0['pos_z'],
    ], dtype=np.float64)


def compute_row_rotation_matrix(row):
    required_keys = ['quat_x', 'quat_y', 'quat_z', 'quat_w']
    if not all(k in row for k in required_keys):
        return None
    try:
        return quat_xyzw_to_rotmat(
            row['quat_x'],
            row['quat_y'],
            row['quat_z'],
            row['quat_w'],
        )
    except Exception:
        return None


def compute_odom_relative_pose(row0, row1):
    R0 = compute_row_rotation_matrix(row0)
    R1 = compute_row_rotation_matrix(row1)
    t_w = compute_translation_from_rows(row0, row1)

    if R0 is None or R1 is None or t_w is None:
        return None, None

    R_rel = R0.T @ R1
    t_rel_in_k = R0.T @ t_w
    return R_rel, t_rel_in_k


def random_noisy_relative(R_gt_rel, t_gt_rel, rng, rot_noise_deg, trans_noise_m):
    noise_rvec = np.deg2rad(rng.normal(0.0, rot_noise_deg, size=3)).reshape(3, 1)
    R_noise, _ = cv2.Rodrigues(noise_rvec)
    t_noise = rng.normal(0.0, trans_noise_m, size=3)

    R_noisy = R_noise @ R_gt_rel
    t_noisy = t_gt_rel + t_noise

    rot_noise_mag_deg = float(np.linalg.norm(noise_rvec) * 180.0 / math.pi)
    trans_noise_mag_m = float(np.linalg.norm(t_noise))
    return R_noisy, t_noisy, rot_noise_mag_deg, trans_noise_mag_m


def load_camera_matrix(camera_info_path: Path, image_shape=None):
    if camera_info_path.exists():
        with open(camera_info_path, 'r', encoding='utf-8') as f:
            data = yaml_safe_load(f)

        if isinstance(data, dict) and 'camera_matrix' in data and 'data' in data['camera_matrix']:
            k = data['camera_matrix']['data']
            K = np.array(k, dtype=np.float64).reshape(3, 3)
            return K

    if image_shape is None:
        raise FileNotFoundError(f'找不到 camera_info.yaml，也沒有 image_shape 可建立近似內參: {camera_info_path}')

    h, w = image_shape[:2]
    fx = float(max(h, w))
    fy = float(max(h, w))
    cx = float(w) / 2.0
    cy = float(h) / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K


def yaml_safe_load(fobj):
    import yaml
    return yaml.safe_load(fobj)


def extract_first_essential_matrix(E):
    if E is None:
        return None
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        return E
    if E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        return E[:3, :]
    return None


def compute_sampson_error(F, pts0, pts1):
    if F is None or len(pts0) == 0:
        return np.array([], dtype=np.float64)

    pts0_h = np.hstack([pts0, np.ones((len(pts0), 1), dtype=np.float64)])
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1), dtype=np.float64)])

    Fx1 = (F @ pts0_h.T).T
    Ftx2 = (F.T @ pts1_h.T).T
    x2tFx1 = np.sum(pts1_h * Fx1, axis=1)

    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    denom = np.clip(denom, 1e-12, None)
    err = (x2tFx1 ** 2) / denom
    return err.astype(np.float64)


def normalize_t_if_possible(t):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(t)
    if n < 1e-12:
        return t
    return t / n


def estimate_visual_geometry(pts0, pts1, K, min_matches_for_geometry=8):
    result = {
        'pose_ok': False,
        'R_vis': np.eye(3, dtype=np.float64),
        't_vis_unit': np.zeros((3,), dtype=np.float64),
        'num_inliers': 0,
        'inlier_ratio': 0.0,
        'pose_mask': np.zeros((len(pts0),), dtype=bool),
        'geo_error_mean': -1.0,
        'geo_error_median': -1.0,
        'vis_rot_deg': -1.0,
    }

    if len(pts0) < min_matches_for_geometry:
        return result

    F, _ = cv2.findFundamentalMat(
        pts0,
        pts1,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999,
    )

    E, mask_e = cv2.findEssentialMat(
        pts0,
        pts1,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    E = extract_first_essential_matrix(E)
    if E is None or mask_e is None:
        return result

    mask_e = mask_e.reshape(-1).astype(bool)
    if int(mask_e.sum()) < min_matches_for_geometry:
        return result

    _, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, cameraMatrix=K, mask=mask_e.astype(np.uint8))
    if mask_pose is None:
        return result

    pose_mask = mask_pose.reshape(-1).astype(bool)
    num_inliers = int(pose_mask.sum())
    inlier_ratio = float(num_inliers) / float(len(pts0)) if len(pts0) > 0 else 0.0

    inlier_pts0 = pts0[pose_mask]
    inlier_pts1 = pts1[pose_mask]

    sampson = compute_sampson_error(F, inlier_pts0, inlier_pts1)
    geo_error_mean = float(sampson.mean()) if len(sampson) > 0 else -1.0
    geo_error_median = float(np.median(sampson)) if len(sampson) > 0 else -1.0

    vis_rot_deg = rotation_angle_deg(R)

    result.update({
        'pose_ok': True,
        'R_vis': R,
        't_vis_unit': normalize_t_if_possible(t.reshape(3)),
        'num_inliers': num_inliers,
        'inlier_ratio': inlier_ratio,
        'pose_mask': pose_mask,
        'geo_error_mean': geo_error_mean,
        'geo_error_median': geo_error_median,
        'vis_rot_deg': vis_rot_deg,
    })
    return result


# =========================================================
# 讀檔
# =========================================================

def read_csv_rows(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_keyframes_csv(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['keyframe_id'] = to_int(row.get('keyframe_id', -1), -1)
            row['source_frame_id'] = to_int(row.get('source_frame_id', -1), -1)

            row['pos_x'] = float(to_float(row.get('pos_x', 0.0), 0.0))
            row['pos_y'] = float(to_float(row.get('pos_y', 0.0), 0.0))
            row['pos_z'] = float(to_float(row.get('pos_z', 0.0), 0.0))

            row['quat_x'] = float(to_float(row.get('quat_x', 0.0), 0.0))
            row['quat_y'] = float(to_float(row.get('quat_y', 0.0), 0.0))
            row['quat_z'] = float(to_float(row.get('quat_z', 0.0), 0.0))
            row['quat_w'] = float(to_float(row.get('quat_w', 1.0), 1.0))

            row['yaw_deg'] = float(to_float(row.get('yaw_deg', 0.0), 0.0))
            row['image_file'] = to_str(row.get('image_file', ''))
            row['timestamp_token'] = to_str(row.get('timestamp_token', ''))
            rows.append(row)

    rows.sort(key=lambda x: x['keyframe_id'])
    return rows


# =========================================================
# Label Builder
# =========================================================

class ReliabilityLabelBuilder:
    def __init__(
        self,
        sequence_dir,
        feature_csv='',
        keyframes_csv='',
        camera_info='',
        out_dir='',
        label_mode='hybrid',
        alpha_hybrid=0.50,
        helpful_thr=0.60,
        harmful_thr=0.40,
        rot_noise_deg=0.80,
        trans_noise_m=0.03,
        min_matches_for_geometry=8,
        seed=42,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.feature_csv = (
            Path(feature_csv).expanduser().resolve()
            if feature_csv
            else self.sequence_dir / 'features' / 'all_candidate_features.csv'
        )
        self.keyframes_csv = (
            Path(keyframes_csv).expanduser().resolve()
            if keyframes_csv
            else self.sequence_dir / 'keyframes' / 'keyframes.csv'
        )
        self.camera_info = (
            Path(camera_info).expanduser().resolve()
            if camera_info
            else self.sequence_dir / 'camera_info.yaml'
        )
        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.sequence_dir / 'reliability_labels'

        self.label_mode = str(label_mode).strip().lower()
        self.alpha_hybrid = float(alpha_hybrid)
        self.helpful_thr = float(helpful_thr)
        self.harmful_thr = float(harmful_thr)

        self.rot_noise_deg = float(rot_noise_deg)
        self.trans_noise_m = float(trans_noise_m)
        self.min_matches_for_geometry = int(min_matches_for_geometry)
        self.seed = int(seed)

        self.matcher = LightGlueMatcher()
        self.rng = np.random.default_rng(self.seed)

        self.keyframe_rows = []
        self.keyframe_index = {}

        self.parallax_scale = 10.0
        self.geo_scale = 1.0
        self.rot_scale = 10.0
        self.tdir_scale = 20.0

    def _check_inputs(self):
        if not self.feature_csv.exists():
            raise FileNotFoundError(f'找不到 feature_csv: {self.feature_csv}')

        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.label_mode not in ['weak', 'hybrid']:
            raise ValueError(f'label_mode 只支援 weak / hybrid，目前收到: {self.label_mode}')

        if not (0.0 <= self.alpha_hybrid <= 1.0):
            raise ValueError('alpha_hybrid 必須介於 0 到 1')

        if not (0.0 <= self.harmful_thr <= self.helpful_thr <= 1.0):
            raise ValueError('threshold 必須滿足 0 <= harmful_thr <= helpful_thr <= 1')

    def _load_optional_keyframes(self):
        if self.keyframes_csv.exists():
            self.keyframe_rows = read_keyframes_csv(self.keyframes_csv)
            self.keyframe_index = {row['keyframe_id']: row for row in self.keyframe_rows}
        else:
            self.keyframe_rows = []
            self.keyframe_index = {}

    def _fit_scales_from_features(self, feature_rows):
        parallax_vals = [to_float(r.get('parallax_mean_px', np.nan), np.nan) for r in feature_rows]
        geo_vals = [to_float(r.get('geo_error_mean', np.nan), np.nan) for r in feature_rows]
        rot_vals = [to_float(r.get('e_rot_iv_deg', np.nan), np.nan) for r in feature_rows]
        tdir_vals = [to_float(r.get('e_trans_dir_iv_deg', np.nan), np.nan) for r in feature_rows]

        self.parallax_scale = robust_positive_scale(parallax_vals, default_value=10.0)
        self.geo_scale = robust_positive_scale(geo_vals, default_value=1.0)
        self.rot_scale = robust_positive_scale(rot_vals, default_value=10.0)
        self.tdir_scale = robust_positive_scale(tdir_vals, default_value=20.0)

    def _resolve_image_path(self, image_file: str) -> Path:
        image_file = to_str(image_file, '')
        p = Path(image_file)
        if p.is_absolute():
            return p
        return self.sequence_dir / 'keyframes' / 'images' / image_file

    def _compute_weak_score(self, row: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
        vis_pose_ok = clip01(to_float(row.get('vis_pose_ok', 0.0), 0.0))
        inlier = clip01(to_float(row.get('match_inlier_ratio', 0.0), 0.0))

        mean_match_score = to_float(row.get('mean_match_score', 0.0), 0.0)
        if not np.isfinite(mean_match_score) or mean_match_score < 0.0:
            mean_match_score = 0.0
        mean_match_score = clip01(mean_match_score)

        coverage0 = to_float(row.get('coverage0', 0.0), 0.0)
        coverage1 = to_float(row.get('coverage1', 0.0), 0.0)
        coverage = clip01(0.5 * (max(coverage0, 0.0) + max(coverage1, 0.0)))

        parallax = to_float(row.get('parallax_mean_px', -1.0), -1.0)
        if np.isfinite(parallax) and parallax >= 0.0:
            parallax_good = clip01(1.0 - math.exp(-parallax / max(self.parallax_scale, 1e-6)))
        else:
            parallax_good = 0.0

        geo_good = exp_good_from_error(
            to_float(row.get('geo_error_mean', -1.0), -1.0),
            scale=self.geo_scale,
            invalid_fill=0.0,
        )
        rot_good = exp_good_from_error(
            to_float(row.get('e_rot_iv_deg', -1.0), -1.0),
            scale=self.rot_scale,
            invalid_fill=0.0,
        )
        tdir_good = exp_good_from_error(
            to_float(row.get('e_trans_dir_iv_deg', -1.0), -1.0),
            scale=self.tdir_scale,
            invalid_fill=0.0,
        )

        is_candidate_valid = clip01(to_float(row.get('is_candidate_valid', 0.0), 0.0))
        is_visual_usable = clip01(to_float(row.get('is_visual_usable', 0.0), 0.0))
        has_missing_feature = clip01(to_float(row.get('has_missing_feature', 0.0), 0.0))

        weights = {
            'pose': 0.12,
            'inlier': 0.18,
            'score': 0.08,
            'coverage': 0.08,
            'parallax': 0.08,
            'geo': 0.12,
            'rot': 0.14,
            'tdir': 0.10,
            'candidate_valid': 0.05,
            'visual_usable': 0.08,
            'missing_penalty': 0.07,
        }
        total_w = sum(weights.values())

        weak_score = (
            weights['pose'] * vis_pose_ok +
            weights['inlier'] * inlier +
            weights['score'] * mean_match_score +
            weights['coverage'] * coverage +
            weights['parallax'] * parallax_good +
            weights['geo'] * geo_good +
            weights['rot'] * rot_good +
            weights['tdir'] * tdir_good +
            weights['candidate_valid'] * is_candidate_valid +
            weights['visual_usable'] * is_visual_usable +
            weights['missing_penalty'] * (1.0 - has_missing_feature)
        ) / total_w
        weak_score = clip01(weak_score)

        comps = {
            'weak_pose_ok': vis_pose_ok,
            'weak_inlier': inlier,
            'weak_match_score': mean_match_score,
            'weak_coverage': coverage,
            'weak_parallax_good': parallax_good,
            'weak_geo_good': geo_good,
            'weak_rot_good': rot_good,
            'weak_tdir_good': tdir_good,
            'weak_candidate_valid': is_candidate_valid,
            'weak_visual_usable': is_visual_usable,
            'weak_missing_bonus': 1.0 - has_missing_feature,
            'weak_score': weak_score,
        }
        return weak_score, comps

    def _row_has_valid_pair_for_hybrid(self, row: Dict[str, str]) -> bool:
        kf_prev_id = to_int(row.get('kf_prev_id', -1), -1)
        kf_curr_id = to_int(row.get('kf_curr_id', -1), -1)

        if kf_prev_id < 0 or kf_curr_id < 0:
            return False
        if kf_prev_id not in self.keyframe_index:
            return False
        if kf_curr_id not in self.keyframe_index:
            return False
        return True

    def _compute_pair_error_metric(self, R_rel, t_rel, R_gt_rel, t_gt_rel) -> Tuple[float, float, float]:
        rot_err_deg = rotation_distance_deg(R_rel, R_gt_rel)

        t_rel_n = normalize(t_rel)
        t_gt_n = normalize(t_gt_rel)
        tdir_err_deg = vector_angle_deg(t_rel_n, t_gt_n)
        if tdir_err_deg < 0.0:
            tdir_err_deg = 180.0

        e = 0.6 * (rot_err_deg / max(self.rot_scale, 1e-6)) + 0.4 * (tdir_err_deg / max(self.tdir_scale, 1e-6))
        return float(e), float(rot_err_deg), float(tdir_err_deg)

    def _compute_hybrid_score(self, row: Dict[str, str], K: Optional[np.ndarray]) -> Tuple[Optional[float], Dict[str, float], str]:
        pair_id = to_int(row.get('pair_id', -1), -1)
        kf_prev_id = to_int(row.get('kf_prev_id', -1), -1)
        kf_curr_id = to_int(row.get('kf_curr_id', -1), -1)

        if not self._row_has_valid_pair_for_hybrid(row):
            return None, {}, 'no_valid_keyframe_pair'

        row0 = self.keyframe_index[kf_prev_id]
        row1 = self.keyframe_index[kf_curr_id]

        img0_path = self._resolve_image_path(row0.get('image_file', ''))
        img1_path = self._resolve_image_path(row1.get('image_file', ''))

        if not img0_path.exists() or not img1_path.exists():
            return None, {}, 'keyframe_image_missing'

        img0 = cv2.imread(str(img0_path), cv2.IMREAD_COLOR)
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
        if img0 is None or img1 is None:
            return None, {}, 'keyframe_image_read_fail'

        if K is None:
            try:
                K = load_camera_matrix(self.camera_info, image_shape=img0.shape)
            except Exception:
                return None, {}, 'camera_info_fail'

        R_gt_rel, t_gt_rel = compute_odom_relative_pose(row0, row1)
        if R_gt_rel is None or t_gt_rel is None:
            return None, {}, 'gt_relative_pose_fail'

        try:
            match_result = self.matcher.match(img0, img1)
        except Exception as e:
            return None, {}, f'matcher_fail:{e}'

        geo = estimate_visual_geometry(
            match_result['mkpts0'],
            match_result['mkpts1'],
            K,
            min_matches_for_geometry=self.min_matches_for_geometry,
        )
        if not geo['pose_ok']:
            return None, {
                'hybrid_pose_ok': 0.0,
                'hybrid_num_matches': float(match_result['num_matches']),
                'hybrid_inlier_ratio': float(geo['inlier_ratio']),
            }, 'visual_pose_fail'

        R_vis = geo['R_vis']
        t_vis_unit = normalize(geo['t_vis_unit'])

        R_inertial_rel, t_inertial_rel, rot_noise_mag_deg, trans_noise_mag_m = random_noisy_relative(
            R_gt_rel,
            t_gt_rel,
            rng=self.rng,
            rot_noise_deg=self.rot_noise_deg,
            trans_noise_m=self.trans_noise_m,
        )

        # without visual
        e_without, rot_err_without_deg, tdir_err_without_deg = self._compute_pair_error_metric(
            R_inertial_rel, t_inertial_rel, R_gt_rel, t_gt_rel
        )

        # with visual (單目平移方向用視覺、尺度仍以 GT/inertial 的 magnitude 當 proxy)
        t_mag = float(np.linalg.norm(t_gt_rel))
        if t_mag < 1e-12:
            t_mag = float(np.linalg.norm(t_inertial_rel))
        if t_mag < 1e-12:
            t_mag = 1.0

        t_visual_scaled = t_vis_unit * t_mag
        e_with, rot_err_with_deg, tdir_err_with_deg = self._compute_pair_error_metric(
            R_vis, t_visual_scaled, R_gt_rel, t_gt_rel
        )

        delta_improvement = float(e_without - e_with)
        hybrid_score = clip01(sigmoid(delta_improvement))

        comps = {
            'hybrid_pose_ok': 1.0,
            'hybrid_num_matches': float(match_result['num_matches']),
            'hybrid_inlier_ratio': float(geo['inlier_ratio']),
            'hybrid_rot_noise_mag_deg': float(rot_noise_mag_deg),
            'hybrid_trans_noise_mag_m': float(trans_noise_mag_m),

            'hybrid_e_without': float(e_without),
            'hybrid_e_with': float(e_with),
            'hybrid_delta_improvement': float(delta_improvement),

            'hybrid_rot_err_without_deg': float(rot_err_without_deg),
            'hybrid_tdir_err_without_deg': float(tdir_err_without_deg),
            'hybrid_rot_err_with_deg': float(rot_err_with_deg),
            'hybrid_tdir_err_with_deg': float(tdir_err_with_deg),

            'hybrid_visual_vs_gt_rot_deg': float(rot_err_with_deg),
            'hybrid_visual_vs_gt_tdir_deg': float(tdir_err_with_deg),
            'hybrid_score': float(hybrid_score),
            'hybrid_pair_id_check': float(pair_id),
        }
        return hybrid_score, comps, 'hybrid_ok'

    def _score_to_class(self, score: float) -> Tuple[int, str]:
        if score < self.harmful_thr:
            return CLASS_NAME_TO_ID['harmful'], 'harmful'
        if score > self.helpful_thr:
            return CLASS_NAME_TO_ID['helpful'], 'helpful'
        return CLASS_NAME_TO_ID['neutral'], 'neutral'

    def build(self):
        self._check_inputs()
        self._load_optional_keyframes()

        feature_rows = read_csv_rows(self.feature_csv)
        self._fit_scales_from_features(feature_rows)

        first_img_path = None
        K = None
        for row in feature_rows:
            image_prev = to_str(row.get('image_prev', ''))
            if image_prev != '':
                p = self._resolve_image_path(image_prev)
                if p.exists():
                    first_img_path = p
                    break

        if first_img_path is not None:
            first_img = cv2.imread(str(first_img_path), cv2.IMREAD_COLOR)
            if first_img is not None:
                try:
                    K = load_camera_matrix(self.camera_info, image_shape=first_img.shape)
                except Exception:
                    K = None

        label_rows = []
        debug_rows = []

        num_total = 0
        num_hybrid_used = 0
        num_weak_only = 0

        for row in feature_rows:
            num_total += 1
            pair_id = to_int(row.get('pair_id', -1), -1)
            if pair_id < 0:
                continue

            weak_score, weak_comps = self._compute_weak_score(row)

            hybrid_score = None
            hybrid_comps = {}
            hybrid_reason = 'disabled'

            if self.label_mode == 'hybrid' and len(self.keyframe_index) > 0:
                hybrid_score, hybrid_comps, hybrid_reason = self._compute_hybrid_score(row, K)

            if self.label_mode == 'weak':
                label_reg = float(weak_score)
                label_source = 'weak_only'
                num_weak_only += 1
            else:
                if hybrid_score is None:
                    label_reg = float(weak_score)
                    label_source = f'weak_only:{hybrid_reason}'
                    num_weak_only += 1
                else:
                    label_reg = clip01(
                        self.alpha_hybrid * float(hybrid_score) +
                        (1.0 - self.alpha_hybrid) * float(weak_score)
                    )
                    label_source = 'hybrid'
                    num_hybrid_used += 1

            label_cls, class_name = self._score_to_class(label_reg)

            label_row = {
                'pair_id': int(pair_id),
                'label_reg': float(label_reg),
                'label_cls': int(label_cls),
                'class_name': str(class_name),
                'label_source': str(label_source),

                'weak_score': float(weak_score),
                'hybrid_score': float(hybrid_score) if hybrid_score is not None else np.nan,
                'alpha_hybrid': float(self.alpha_hybrid),

                'kf_prev_id': to_int(row.get('kf_prev_id', -1), -1),
                'kf_curr_id': to_int(row.get('kf_curr_id', -1), -1),
                'accepted': to_int(row.get('accepted', -1), -1),
                'reason': to_str(row.get('reason', '')),
            }
            label_rows.append(label_row)

            debug_row = {
                'pair_id': int(pair_id),
                'label_reg': float(label_reg),
                'label_cls': int(label_cls),
                'class_name': str(class_name),
                'label_source': str(label_source),
                'accepted': to_int(row.get('accepted', -1), -1),
                'reason': to_str(row.get('reason', '')),
                'kf_prev_id': to_int(row.get('kf_prev_id', -1), -1),
                'kf_curr_id': to_int(row.get('kf_curr_id', -1), -1),
            }
            debug_row.update(weak_comps)
            debug_row.update(hybrid_comps)
            debug_rows.append(debug_row)

        label_csv_path = self.out_dir / 'reliability_labels.csv'
        debug_csv_path = self.out_dir / 'label_debug.csv'
        summary_json_path = self.out_dir / 'summary.json'

        with open(label_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'pair_id',
                    'label_reg',
                    'label_cls',
                    'class_name',
                    'label_source',
                    'weak_score',
                    'hybrid_score',
                    'alpha_hybrid',
                    'kf_prev_id',
                    'kf_curr_id',
                    'accepted',
                    'reason',
                ],
            )
            writer.writeheader()
            for row in label_rows:
                writer.writerow(row)

        debug_fieldnames = sorted(set().union(*[set(r.keys()) for r in debug_rows])) if len(debug_rows) > 0 else [
            'pair_id', 'label_reg', 'label_cls', 'class_name', 'label_source'
        ]
        with open(debug_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=debug_fieldnames)
            writer.writeheader()
            for row in debug_rows:
                writer.writerow(row)

        class_counts = {'harmful': 0, 'neutral': 0, 'helpful': 0}
        for row in label_rows:
            cname = row['class_name']
            if cname in class_counts:
                class_counts[cname] += 1

        summary = {
            'sequence_dir': str(self.sequence_dir),
            'feature_csv': str(self.feature_csv),
            'keyframes_csv': str(self.keyframes_csv),
            'camera_info': str(self.camera_info),
            'out_dir': str(self.out_dir),

            'label_mode': str(self.label_mode),
            'alpha_hybrid': float(self.alpha_hybrid),
            'helpful_thr': float(self.helpful_thr),
            'harmful_thr': float(self.harmful_thr),

            'rot_noise_deg': float(self.rot_noise_deg),
            'trans_noise_m': float(self.trans_noise_m),
            'min_matches_for_geometry': int(self.min_matches_for_geometry),
            'seed': int(self.seed),

            'scales': {
                'parallax_scale': float(self.parallax_scale),
                'geo_scale': float(self.geo_scale),
                'rot_scale': float(self.rot_scale),
                'tdir_scale': float(self.tdir_scale),
            },

            'num_total_feature_rows': int(num_total),
            'num_labeled_rows': int(len(label_rows)),
            'num_hybrid_used': int(num_hybrid_used),
            'num_weak_only': int(num_weak_only),
            'class_counts': class_counts,
        }

        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print('========== Build Reliability Labels Finished ==========')
        print(f'label csv: {label_csv_path}')
        print(f'debug csv: {debug_csv_path}')
        print(f'summary json: {summary_json_path}')
        for k, v in summary.items():
            if k == 'scales' or k == 'class_counts':
                print(f'{k}: {v}')
            elif not isinstance(v, dict):
                print(f'{k}: {v}')

        return {
            'label_csv': str(label_csv_path),
            'debug_csv': str(debug_csv_path),
            'summary_json': str(summary_json_path),
            'num_labeled_rows': int(len(label_rows)),
            'num_hybrid_used': int(num_hybrid_used),
            'num_weak_only': int(num_weak_only),
            'class_counts': class_counts,
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
        '--keyframes_csv',
        type=str,
        default='',
        help='若留空，預設 sequence_dir/keyframes/keyframes.csv；若不存在，則自動退化為 weak-only'
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
        help='若留空，預設 sequence_dir/reliability_labels'
    )
    parser.add_argument(
        '--label_mode',
        type=str,
        default='hybrid',
        choices=['weak', 'hybrid'],
        help='weak: 只用 feature-based weak label；hybrid: 有效 pair 會再混入 GT/pair-wise improvement'
    )
    parser.add_argument(
        '--alpha_hybrid',
        type=float,
        default=0.50,
        help='final_label = alpha_hybrid * hybrid_score + (1-alpha_hybrid) * weak_score'
    )
    parser.add_argument('--helpful_thr', type=float, default=0.60)
    parser.add_argument('--harmful_thr', type=float, default=0.40)
    parser.add_argument('--rot_noise_deg', type=float, default=0.80)
    parser.add_argument('--trans_noise_m', type=float, default=0.03)
    parser.add_argument('--min_matches_for_geometry', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    builder = ReliabilityLabelBuilder(
        sequence_dir=args.sequence_dir,
        feature_csv=args.feature_csv,
        keyframes_csv=args.keyframes_csv,
        camera_info=args.camera_info,
        out_dir=args.out_dir,
        label_mode=args.label_mode,
        alpha_hybrid=args.alpha_hybrid,
        helpful_thr=args.helpful_thr,
        harmful_thr=args.harmful_thr,
        rot_noise_deg=args.rot_noise_deg,
        trans_noise_m=args.trans_noise_m,
        min_matches_for_geometry=args.min_matches_for_geometry,
        seed=args.seed,
    )
    builder.build()


if __name__ == '__main__':
    main()