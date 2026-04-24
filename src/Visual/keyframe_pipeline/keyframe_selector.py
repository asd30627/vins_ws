import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from matcher import LightGlueMatcher


ALL_CANDIDATE_PAIR_HEADER = [
    'pair_id',
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
    'has_motion_info',
    'prefilter_pass',
    'matcher_success',
    'visual_pass',
    'motion_pass',

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
]


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


def compute_translation(row0, row1):
    required_keys = ['pos_x', 'pos_y', 'pos_z']
    if not all(k in row0 for k in required_keys):
        return None
    if not all(k in row1 for k in required_keys):
        return None

    dx = row1['pos_x'] - row0['pos_x']
    dy = row1['pos_y'] - row0['pos_y']
    dz = row1['pos_z'] - row0['pos_z']
    return math.sqrt(dx * dx + dy * dy + dz * dz)


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


def compute_rotation_deg(row0, row1):
    required_keys = ['quat_x', 'quat_y', 'quat_z', 'quat_w']
    if not all(k in row0 for k in required_keys):
        return None
    if not all(k in row1 for k in required_keys):
        return None

    try:
        R0 = quat_xyzw_to_rotmat(row0['quat_x'], row0['quat_y'], row0['quat_z'], row0['quat_w'])
        R1 = quat_xyzw_to_rotmat(row1['quat_x'], row1['quat_y'], row1['quat_z'], row1['quat_w'])
    except Exception:
        return None

    R_rel = R0.T @ R1
    trace_val = np.trace(R_rel)
    c = float(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(c))


def load_camera_matrix(camera_info_path: Path, image_shape=None):
    if camera_info_path.exists():
        with open(camera_info_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

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


def compute_odom_relative_pose(row0, row1):
    R0 = compute_row_rotation_matrix(row0)
    R1 = compute_row_rotation_matrix(row1)
    t_w = compute_translation_from_rows(row0, row1)

    if R0 is None or R1 is None or t_w is None:
        return None, None

    R_rel = R0.T @ R1
    t_rel_in_k = R0.T @ t_w
    return R_rel, t_rel_in_k


def rotation_angle_deg(R):
    trace_val = np.trace(R)
    c = (trace_val - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def rotation_distance_deg(Ra, Rb):
    return rotation_angle_deg(Ra.T @ Rb)


def vector_angle_deg(v0, v1):
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return -1.0

    c = float(np.dot(v0, v1) / (n0 * n1))
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def compute_coverage(points, width, height, rows=4, cols=4):
    if len(points) == 0:
        return 0.0

    occupied = np.zeros((rows, cols), dtype=np.uint8)

    for x, y in points:
        c = min(cols - 1, max(0, int(x / max(width, 1) * cols)))
        r = min(rows - 1, max(0, int(y / max(height, 1) * rows)))
        occupied[r, c] = 1

    return float(occupied.sum()) / float(rows * cols)


def compute_parallax_stats(pts0, pts1):
    if len(pts0) == 0:
        return 0.0, 0.0
    d = np.linalg.norm(pts1 - pts0, axis=1)
    return float(d.mean()), float(np.median(d))


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


def extract_first_essential_matrix(E):
    if E is None:
        return None
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        return E
    if E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        return E[:3, :]
    return None


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


def read_frame_rows(path: Path) -> List[Dict[str, object]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)

            if 'keyframe_id' in row:
                parsed['keyframe_id'] = to_int(row.get('keyframe_id', -1), -1)
            else:
                parsed['keyframe_id'] = -1

            if 'source_frame_id' in row:
                parsed['source_frame_id'] = to_int(row.get('source_frame_id', -1), -1)
            else:
                parsed['source_frame_id'] = -1

            parsed['pos_x'] = float(to_float(row.get('pos_x', 0.0), 0.0))
            parsed['pos_y'] = float(to_float(row.get('pos_y', 0.0), 0.0))
            parsed['pos_z'] = float(to_float(row.get('pos_z', 0.0), 0.0))

            parsed['quat_x'] = float(to_float(row.get('quat_x', 0.0), 0.0))
            parsed['quat_y'] = float(to_float(row.get('quat_y', 0.0), 0.0))
            parsed['quat_z'] = float(to_float(row.get('quat_z', 0.0), 0.0))
            parsed['quat_w'] = float(to_float(row.get('quat_w', 1.0), 1.0))

            parsed['yaw_deg'] = float(to_float(row.get('yaw_deg', 0.0), 0.0))
            parsed['image_file'] = to_str(row.get('image_file', ''))
            parsed['timestamp_token'] = to_str(row.get('timestamp_token', ''))

            rows.append(parsed)

    def sort_key(x):
        if x['source_frame_id'] >= 0:
            return x['source_frame_id']
        if x['keyframe_id'] >= 0:
            return x['keyframe_id']
        return 0

    rows.sort(key=sort_key)
    return rows


def resolve_image_path(images_dir: Path, image_file: str) -> Path:
    p = Path(image_file)
    if p.is_absolute():
        return p
    return images_dir / image_file


# =========================================================
# KeyframeSelector
# =========================================================

class KeyframeSelector:
    """
    v2 重點：
    1. update(frame_packet) 保留 online 用法
    2. 每個 candidate pair 都寫入 all_candidate_pairs.csv
    3. accepted / reject_visual / reject_motion / prefilter_skip / matcher_fail 全部保留
    4. 不再使用舊版 matcher.extract / match_features，統一改成 matcher.match(img0, img1)
    """

    def __init__(
        self,
        prefilter_translation_m: float = 0.20,
        prefilter_rotation_deg: float = 2.0,
        min_matches: int = 120,
        min_inlier_ratio: float = 0.30,
        min_translation_m: float = 0.50,
        min_rotation_deg: float = 5.0,
        matcher=None,
        candidate_csv_path: str = '',
        camera_info_path: str = '',
        grid_rows: int = 4,
        grid_cols: int = 4,
        min_matches_for_geometry: int = 8,
        auto_init_csv: bool = True,
    ):
        self.prefilter_translation_m = float(prefilter_translation_m)
        self.prefilter_rotation_deg = float(prefilter_rotation_deg)
        self.min_matches = int(min_matches)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_translation_m = float(min_translation_m)
        self.min_rotation_deg = float(min_rotation_deg)

        self.matcher = matcher or LightGlueMatcher()

        self.candidate_csv_path = Path(candidate_csv_path).expanduser().resolve() if candidate_csv_path else None
        self.camera_info_path = Path(camera_info_path).expanduser().resolve() if camera_info_path else None
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.min_matches_for_geometry = int(min_matches_for_geometry)

        self.last_kf_row = None
        self.last_kf_img = None
        self.keyframe_id = 0
        self.candidate_pair_id = 0
        self.K = None

        if auto_init_csv and self.candidate_csv_path is not None:
            self._init_candidate_file()

    def _init_candidate_file(self):
        self.candidate_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.candidate_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ALL_CANDIDATE_PAIR_HEADER)
            writer.writeheader()

    def _append_candidate_row(self, row: Dict[str, object]):
        if self.candidate_csv_path is None:
            return
        with open(self.candidate_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ALL_CANDIDATE_PAIR_HEADER)
            writer.writerow(row)

    def _ensure_camera_matrix(self, img_shape):
        if self.K is not None:
            return self.K

        if self.camera_info_path is None:
            self.K = load_camera_matrix(Path(''), image_shape=img_shape)
        else:
            self.K = load_camera_matrix(self.camera_info_path, image_shape=img_shape)
        return self.K

    def _base_candidate_row(self, prev_row, curr_row, translation_m, rotation_deg):
        return {
            'pair_id': int(self.candidate_pair_id),
            'kf_prev_id': int(max(self.keyframe_id - 1, 0)),
            'kf_curr_id': -1,
            'src_prev_id': int(to_int(prev_row.get('source_frame_id', -1), -1)),
            'src_curr_id': int(to_int(curr_row.get('source_frame_id', -1), -1)),
            'image_prev': to_str(prev_row.get('image_file', '')),
            'image_curr': to_str(curr_row.get('image_file', '')),
            'timestamp_prev': to_str(prev_row.get('timestamp_token', '')),
            'timestamp_curr': to_str(curr_row.get('timestamp_token', '')),

            'accepted': 0,
            'reason': '',
            'has_motion_info': 0,
            'prefilter_pass': 0,
            'matcher_success': 0,
            'visual_pass': 0,
            'motion_pass': 0,

            'odom_translation_m': float(translation_m) if translation_m is not None else -1.0,
            'odom_rotation_deg': float(rotation_deg) if rotation_deg is not None else -1.0,

            'num_keypoints0': -1,
            'num_keypoints1': -1,
            'num_matches': -1,
            'num_inliers': -1,
            'match_inlier_ratio': -1.0,
            'mean_match_score': -1.0,

            'coverage0': -1.0,
            'coverage1': -1.0,
            'parallax_mean_px': -1.0,
            'parallax_median_px': -1.0,
            'geo_error_mean': -1.0,
            'geo_error_median': -1.0,

            'vis_pose_ok': 0,
            'vis_rot_deg': -1.0,
            'e_rot_iv_deg': -1.0,
            'e_trans_dir_iv_deg': -1.0,
        }

    def _fill_visual_metrics(self, out_row, prev_row, curr_row, prev_img, curr_img, match_result):
        K = self._ensure_camera_matrix(prev_img.shape)

        pts0 = match_result['mkpts0']
        pts1 = match_result['mkpts1']
        match_scores = match_result['match_scores']

        geo = estimate_visual_geometry(
            pts0,
            pts1,
            K,
            min_matches_for_geometry=self.min_matches_for_geometry,
        )

        pose_mask = geo['pose_mask']
        if len(pose_mask) == len(pts0):
            inlier_pts0 = pts0[pose_mask]
            inlier_pts1 = pts1[pose_mask]
        else:
            inlier_pts0 = pts0
            inlier_pts1 = pts1

        if len(match_scores) > 0 and len(pose_mask) == len(match_scores):
            valid_scores = match_scores[pose_mask]
            valid_scores = valid_scores[valid_scores >= 0.0]
            mean_match_score = float(valid_scores.mean()) if len(valid_scores) > 0 else float(match_result['mean_match_score'])
        else:
            mean_match_score = float(match_result['mean_match_score'])

        h0, w0 = prev_img.shape[:2]
        h1, w1 = curr_img.shape[:2]

        coverage0 = compute_coverage(inlier_pts0, w0, h0, rows=self.grid_rows, cols=self.grid_cols)
        coverage1 = compute_coverage(inlier_pts1, w1, h1, rows=self.grid_rows, cols=self.grid_cols)
        parallax_mean_px, parallax_median_px = compute_parallax_stats(inlier_pts0, inlier_pts1)

        R_odom_rel, t_odom_rel = compute_odom_relative_pose(prev_row, curr_row)
        if geo['pose_ok'] and R_odom_rel is not None and t_odom_rel is not None:
            e_rot_iv_deg = rotation_distance_deg(geo['R_vis'], R_odom_rel)
            e_trans_dir_iv_deg = vector_angle_deg(geo['t_vis_unit'], t_odom_rel)
        else:
            e_rot_iv_deg = -1.0
            e_trans_dir_iv_deg = -1.0

        out_row.update({
            'num_keypoints0': int(match_result['num_keypoints0']),
            'num_keypoints1': int(match_result['num_keypoints1']),
            'num_matches': int(match_result['num_matches']),
            'num_inliers': int(geo['num_inliers']),
            'match_inlier_ratio': float(geo['inlier_ratio']),
            'mean_match_score': float(mean_match_score),

            'coverage0': float(coverage0),
            'coverage1': float(coverage1),
            'parallax_mean_px': float(parallax_mean_px),
            'parallax_median_px': float(parallax_median_px),

            'geo_error_mean': float(geo['geo_error_mean']),
            'geo_error_median': float(geo['geo_error_median']),

            'vis_pose_ok': int(geo['pose_ok']),
            'vis_rot_deg': float(geo['vis_rot_deg']),
            'e_rot_iv_deg': float(e_rot_iv_deg),
            'e_trans_dir_iv_deg': float(e_trans_dir_iv_deg),
        })
        return out_row

    def reset(self):
        self.last_kf_row = None
        self.last_kf_img = None
        self.keyframe_id = 0
        self.candidate_pair_id = 0
        self.K = None

        if self.candidate_csv_path is not None:
            self._init_candidate_file()

    def update(self, frame_packet):
        if frame_packet is None:
            raise ValueError('frame_packet 是 None')
        if 'image_bgr' not in frame_packet:
            raise ValueError("frame_packet 缺少 'image_bgr'")
        img = frame_packet['image_bgr']
        if img is None:
            raise ValueError("frame_packet['image_bgr'] 是 None")

        source_frame_id = frame_packet.get('source_frame_id', -1)

        # 第一張直接作為起始 keyframe，不產生 pair row
        if self.last_kf_row is None:
            result = {
                'accepted': True,
                'reason': 'first_frame',
                'keyframe_id': self.keyframe_id,
                'source_frame_id': source_frame_id,
                'translation_m': 0.0,
                'rotation_deg': 0.0,
                'num_matches': -1,
                'inlier_ratio': -1.0,
                'match_result': None,
                'candidate_row': None,
                'csv_written': False,
            }
            self.last_kf_row = frame_packet
            self.last_kf_img = img
            self.keyframe_id += 1
            return result

        prev_row = self.last_kf_row
        prev_img = self.last_kf_img

        translation_m = compute_translation(prev_row, frame_packet)
        rotation_deg = compute_rotation_deg(prev_row, frame_packet)

        has_motion_info = (translation_m is not None) or (rotation_deg is not None)
        out_row = self._base_candidate_row(prev_row, frame_packet, translation_m, rotation_deg)
        out_row['has_motion_info'] = int(has_motion_info)

        too_small_translation = (translation_m is not None and translation_m < self.prefilter_translation_m)
        too_small_rotation = (rotation_deg is not None and rotation_deg < self.prefilter_rotation_deg)

        prefilter_pass = True
        if has_motion_info:
            checks = []
            if translation_m is not None:
                checks.append(too_small_translation)
            if rotation_deg is not None:
                checks.append(too_small_rotation)
            if len(checks) > 0 and all(checks):
                prefilter_pass = False

        out_row['prefilter_pass'] = int(prefilter_pass)

        if not prefilter_pass:
            out_row['reason'] = 'prefilter_skip'
            self._append_candidate_row(out_row)
            self.candidate_pair_id += 1
            return {
                'accepted': False,
                'reason': 'prefilter_skip',
                'keyframe_id': None,
                'source_frame_id': source_frame_id,
                'translation_m': translation_m,
                'rotation_deg': rotation_deg,
                'num_matches': 0,
                'inlier_ratio': 0.0,
                'match_result': None,
                'candidate_row': out_row,
                'csv_written': self.candidate_csv_path is not None,
            }

        try:
            match_result = self.matcher.match(prev_img, img)
            out_row['matcher_success'] = 1
        except Exception as e:
            out_row['reason'] = f'matcher_fail: {e}'
            self._append_candidate_row(out_row)
            self.candidate_pair_id += 1
            return {
                'accepted': False,
                'reason': out_row['reason'],
                'keyframe_id': None,
                'source_frame_id': source_frame_id,
                'translation_m': translation_m,
                'rotation_deg': rotation_deg,
                'num_matches': 0,
                'inlier_ratio': 0.0,
                'match_result': None,
                'candidate_row': out_row,
                'csv_written': self.candidate_csv_path is not None,
            }

        out_row = self._fill_visual_metrics(
            out_row,
            prev_row=prev_row,
            curr_row=frame_packet,
            prev_img=prev_img,
            curr_img=img,
            match_result=match_result,
        )

        visual_pass = (
            out_row['num_matches'] >= self.min_matches and
            out_row['match_inlier_ratio'] >= self.min_inlier_ratio
        )

        if not has_motion_info:
            motion_pass = True
        else:
            motion_checks = []
            if translation_m is not None:
                motion_checks.append(translation_m >= self.min_translation_m)
            if rotation_deg is not None:
                motion_checks.append(rotation_deg >= self.min_rotation_deg)
            motion_pass = any(motion_checks) if len(motion_checks) > 0 else True

        out_row['visual_pass'] = int(visual_pass)
        out_row['motion_pass'] = int(motion_pass)

        accepted = bool(visual_pass and motion_pass)
        if accepted:
            out_row['accepted'] = 1
            out_row['reason'] = 'accept'
            out_row['kf_curr_id'] = int(self.keyframe_id)

            self.last_kf_row = frame_packet
            self.last_kf_img = img
            accepted_keyframe_id = self.keyframe_id
            self.keyframe_id += 1
        else:
            if not visual_pass and not motion_pass:
                out_row['reason'] = 'reject_visual_and_motion'
            elif not visual_pass:
                out_row['reason'] = 'reject_visual'
            else:
                out_row['reason'] = 'reject_motion'
            accepted_keyframe_id = None

        self._append_candidate_row(out_row)
        self.candidate_pair_id += 1

        return {
            'accepted': accepted,
            'reason': out_row['reason'],
            'keyframe_id': accepted_keyframe_id,
            'source_frame_id': source_frame_id,
            'translation_m': translation_m,
            'rotation_deg': rotation_deg,
            'num_matches': int(out_row['num_matches']),
            'inlier_ratio': float(out_row['match_inlier_ratio']),
            'match_result': match_result,
            'candidate_row': out_row,
            'csv_written': self.candidate_csv_path is not None,
        }


# =========================================================
# Batch Runner
# =========================================================

class KeyframeSelectionBatchRunner:
    """
    用一份 frames_csv + images_dir，批次產生 all_candidate_pairs.csv

    注意：
    - 如果 frames_csv 是「全部原始 frame」，那輸出的 all_candidate_pairs 才是完整候選。
    - 如果 frames_csv 其實只是已經篩過的 keyframes.csv，
      那輸出的 all_candidate_pairs 只會覆蓋這份 csv 內的 row。
    """

    def __init__(
        self,
        sequence_dir,
        frames_csv='',
        images_dir='',
        out_csv='',
        camera_info='',
        prefilter_translation_m=0.20,
        prefilter_rotation_deg=2.0,
        min_matches=120,
        min_inlier_ratio=0.30,
        min_translation_m=0.50,
        min_rotation_deg=5.0,
        grid_rows=4,
        grid_cols=4,
        min_matches_for_geometry=8,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.frames_csv = Path(frames_csv).expanduser().resolve() if frames_csv else self.sequence_dir / 'keyframes' / 'keyframes.csv'
        self.images_dir = Path(images_dir).expanduser().resolve() if images_dir else self.sequence_dir / 'keyframes' / 'images'
        self.out_csv = Path(out_csv).expanduser().resolve() if out_csv else self.sequence_dir / 'keyframes' / 'all_candidate_pairs.csv'
        self.camera_info = Path(camera_info).expanduser().resolve() if camera_info else self.sequence_dir / 'camera_info.yaml'

        self.prefilter_translation_m = float(prefilter_translation_m)
        self.prefilter_rotation_deg = float(prefilter_rotation_deg)
        self.min_matches = int(min_matches)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_translation_m = float(min_translation_m)
        self.min_rotation_deg = float(min_rotation_deg)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.min_matches_for_geometry = int(min_matches_for_geometry)

    def _check_inputs(self):
        if not self.frames_csv.exists():
            raise FileNotFoundError(f'找不到 frames_csv: {self.frames_csv}')
        if not self.images_dir.exists():
            raise FileNotFoundError(f'找不到 images_dir: {self.images_dir}')

    def run(self):
        self._check_inputs()

        rows = read_frame_rows(self.frames_csv)
        if len(rows) < 2:
            raise RuntimeError(f'frames_csv row 數量不足，至少需要 2 筆，目前只有 {len(rows)}')

        selector = KeyframeSelector(
            prefilter_translation_m=self.prefilter_translation_m,
            prefilter_rotation_deg=self.prefilter_rotation_deg,
            min_matches=self.min_matches,
            min_inlier_ratio=self.min_inlier_ratio,
            min_translation_m=self.min_translation_m,
            min_rotation_deg=self.min_rotation_deg,
            matcher=LightGlueMatcher(),
            candidate_csv_path=str(self.out_csv),
            camera_info_path=str(self.camera_info),
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            min_matches_for_geometry=self.min_matches_for_geometry,
            auto_init_csv=True,
        )

        num_accept = 0
        num_prefilter_skip = 0
        num_matcher_fail = 0
        num_reject_visual = 0
        num_reject_motion = 0
        num_reject_both = 0
        num_first_frame = 0

        for idx, row in enumerate(rows):
            img_path = resolve_image_path(self.images_dir, row['image_file'])
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f'影像讀取失敗: {img_path}')

            frame_packet = dict(row)
            frame_packet['image_bgr'] = img

            decision = selector.update(frame_packet)
            reason = decision['reason']

            if reason == 'first_frame':
                num_first_frame += 1
            elif reason == 'accept':
                num_accept += 1
            elif reason == 'prefilter_skip':
                num_prefilter_skip += 1
            elif str(reason).startswith('matcher_fail'):
                num_matcher_fail += 1
            elif reason == 'reject_visual':
                num_reject_visual += 1
            elif reason == 'reject_motion':
                num_reject_motion += 1
            elif reason == 'reject_visual_and_motion':
                num_reject_both += 1

            if (idx + 1) % 50 == 0:
                print(f'processed frames: {idx + 1}/{len(rows)}')

        summary = {
            'sequence_dir': str(self.sequence_dir),
            'frames_csv': str(self.frames_csv),
            'images_dir': str(self.images_dir),
            'out_csv': str(self.out_csv),
            'num_input_rows': int(len(rows)),
            'num_candidate_rows': int(max(len(rows) - 1, 0)),
            'num_first_frame': int(num_first_frame),
            'accept': int(num_accept),
            'prefilter_skip': int(num_prefilter_skip),
            'matcher_fail': int(num_matcher_fail),
            'reject_visual': int(num_reject_visual),
            'reject_motion': int(num_reject_motion),
            'reject_visual_and_motion': int(num_reject_both),
        }

        print('========== Keyframe Selection Batch Finished ==========')
        for k, v in summary.items():
            print(f'{k}: {v}')

        return summary


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_dir', type=str, required=True, help='例如: /mnt/sata4t/dataset/sequence_001')
    parser.add_argument('--frames_csv', type=str, default='', help='若留空，預設 sequence_dir/keyframes/keyframes.csv')
    parser.add_argument('--images_dir', type=str, default='', help='若留空，預設 sequence_dir/keyframes/images')
    parser.add_argument('--out_csv', type=str, default='', help='若留空，預設 sequence_dir/keyframes/all_candidate_pairs.csv')
    parser.add_argument('--camera_info', type=str, default='', help='若留空，預設 sequence_dir/camera_info.yaml')

    parser.add_argument('--prefilter_translation_m', type=float, default=0.20)
    parser.add_argument('--prefilter_rotation_deg', type=float, default=2.0)
    parser.add_argument('--min_matches', type=int, default=120)
    parser.add_argument('--min_inlier_ratio', type=float, default=0.30)
    parser.add_argument('--min_translation_m', type=float, default=0.50)
    parser.add_argument('--min_rotation_deg', type=float, default=5.0)

    parser.add_argument('--grid_rows', type=int, default=4)
    parser.add_argument('--grid_cols', type=int, default=4)
    parser.add_argument('--min_matches_for_geometry', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    runner = KeyframeSelectionBatchRunner(
        sequence_dir=args.sequence_dir,
        frames_csv=args.frames_csv,
        images_dir=args.images_dir,
        out_csv=args.out_csv,
        camera_info=args.camera_info,
        prefilter_translation_m=args.prefilter_translation_m,
        prefilter_rotation_deg=args.prefilter_rotation_deg,
        min_matches=args.min_matches,
        min_inlier_ratio=args.min_inlier_ratio,
        min_translation_m=args.min_translation_m,
        min_rotation_deg=args.min_rotation_deg,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        min_matches_for_geometry=args.min_matches_for_geometry,
    )
    runner.run()


if __name__ == '__main__':
    main()