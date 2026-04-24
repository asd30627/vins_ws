import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


FEATURE_HEADER = [
    # pair / frame meta
    'pair_id',
    'kf_prev_id',
    'kf_curr_id',
    'src_prev_id',
    'src_curr_id',
    'image_prev',
    'image_curr',
    'timestamp_prev',
    'timestamp_curr',

    # decision / status from selector
    'accepted',
    'reason',
    'has_motion_info',
    'prefilter_pass',
    'matcher_success',
    'visual_pass',
    'motion_pass',

    # helper flags for v2 dataset
    'is_candidate_valid',
    'is_visual_usable',
    'has_missing_feature',

    # odom motion
    'odom_translation_m',
    'odom_rotation_deg',

    # visual matching
    'num_keypoints0',
    'num_keypoints1',
    'num_matches',
    'num_inliers',
    'match_inlier_ratio',
    'mean_match_score',

    # spatial / geometry
    'coverage0',
    'coverage1',
    'parallax_mean_px',
    'parallax_median_px',
    'geo_error_mean',
    'geo_error_median',

    # visual geometry consistency
    'vis_pose_ok',
    'vis_rot_deg',
    'e_rot_iv_deg',
    'e_trans_dir_iv_deg',

    # image quality features
    'blur0',
    'blur1',
    'brightness0',
    'brightness1',
    'texture0',
    'texture1',
    'edge_density0',
    'edge_density1',
]


COPIED_FIELDS = [
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


class ReliabilityFeatureExtractor:
    """
    v2 重點：
    1. 輸入改成 all_candidate_pairs.csv，而不是 accepted_pairs_cache.csv
    2. 每一個 candidate pair 都保留，不只 accepted
    3. 補上影像品質特徵 blur / brightness / texture / edge_density
    4. 額外輸出三個輔助欄位：
       - is_candidate_valid
       - is_visual_usable
       - has_missing_feature
    """

    def __init__(
        self,
        sequence_dir,
        candidate_csv_path='',
        out_dir='',
        out_csv_name='all_candidate_features.csv',
        clean_output_dir=False,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.keyframes_dir = self.sequence_dir / 'keyframes'
        self.keyframe_images_dir = self.keyframes_dir / 'images'

        self.candidate_csv_path = (
            Path(candidate_csv_path).expanduser().resolve()
            if candidate_csv_path
            else self.keyframes_dir / 'all_candidate_pairs.csv'
        )

        self.feature_dir = Path(out_dir).expanduser().resolve() if out_dir else self.sequence_dir / 'features'
        self.feature_csv = self.feature_dir / out_csv_name

        self.clean_output_dir = bool(clean_output_dir)

    @staticmethod
    def ensure_dir(path: Path, clean=False):
        if clean and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_csv_rows(path: Path) -> List[Dict[str, str]]:
        with open(path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    @staticmethod
    def to_float(value, default=-1.0) -> float:
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
    def compute_blur_score(img_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_brightness(img_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    @staticmethod
    def compute_texture_score(img_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        return float(mag.mean())

    @staticmethod
    def compute_edge_density(img_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return float((edges > 0).mean())

    def _check_inputs(self):
        if not self.candidate_csv_path.exists():
            raise FileNotFoundError(f'找不到 all_candidate_pairs.csv: {self.candidate_csv_path}')
        if not self.keyframe_images_dir.exists():
            raise FileNotFoundError(f'找不到 keyframe images 目錄: {self.keyframe_images_dir}')

    def _resolve_image_path(self, image_field: str) -> Path:
        image_field = self.to_str(image_field, '')
        if image_field == '':
            return self.keyframe_images_dir / '__missing__.png'

        p = Path(image_field)
        if p.is_absolute():
            return p

        return self.keyframe_images_dir / image_field

    def _read_image_safe(self, image_field: str) -> Tuple[np.ndarray, bool]:
        path = self._resolve_image_path(image_field)
        if not path.exists():
            return None, False

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None, False
        return img, True

    def _copy_base_fields(self, candidate_row: Dict[str, str]) -> Dict[str, object]:
        out = {}

        for name in COPIED_FIELDS:
            if name in [
                'pair_id',
                'kf_prev_id',
                'kf_curr_id',
                'src_prev_id',
                'src_curr_id',
                'accepted',
                'has_motion_info',
                'prefilter_pass',
                'matcher_success',
                'visual_pass',
                'motion_pass',
                'num_keypoints0',
                'num_keypoints1',
                'num_matches',
                'num_inliers',
                'vis_pose_ok',
            ]:
                out[name] = self.to_int(candidate_row.get(name, -1), -1)

            elif name in [
                'odom_translation_m',
                'odom_rotation_deg',
                'match_inlier_ratio',
                'mean_match_score',
                'coverage0',
                'coverage1',
                'parallax_mean_px',
                'parallax_median_px',
                'geo_error_mean',
                'geo_error_median',
                'vis_rot_deg',
                'e_rot_iv_deg',
                'e_trans_dir_iv_deg',
            ]:
                out[name] = self.to_float(candidate_row.get(name, -1.0), -1.0)

            else:
                out[name] = self.to_str(candidate_row.get(name, ''))

        return out

    def _compute_image_quality_features(self, img0, img1) -> Tuple[Dict[str, float], int]:
        has_missing_feature = 0

        if img0 is None:
            blur0 = -1.0
            brightness0 = -1.0
            texture0 = -1.0
            edge_density0 = -1.0
            has_missing_feature = 1
        else:
            blur0 = self.compute_blur_score(img0)
            brightness0 = self.compute_brightness(img0)
            texture0 = self.compute_texture_score(img0)
            edge_density0 = self.compute_edge_density(img0)

        if img1 is None:
            blur1 = -1.0
            brightness1 = -1.0
            texture1 = -1.0
            edge_density1 = -1.0
            has_missing_feature = 1
        else:
            blur1 = self.compute_blur_score(img1)
            brightness1 = self.compute_brightness(img1)
            texture1 = self.compute_texture_score(img1)
            edge_density1 = self.compute_edge_density(img1)

        features = {
            'blur0': float(blur0),
            'blur1': float(blur1),
            'brightness0': float(brightness0),
            'brightness1': float(brightness1),
            'texture0': float(texture0),
            'texture1': float(texture1),
            'edge_density0': float(edge_density0),
            'edge_density1': float(edge_density1),
        }
        return features, has_missing_feature

    def _derive_helper_flags(self, row: Dict[str, object], images_ok: bool, has_missing_feature: int) -> Tuple[int, int]:
        prefilter_pass = int(row['prefilter_pass']) == 1
        matcher_success = int(row['matcher_success']) == 1

        # candidate valid:
        # 代表這筆 pair 至少通過 prefilter，且 matcher 成功，且影像能正常讀取
        is_candidate_valid = int(prefilter_pass and matcher_success and images_ok)

        # visual usable:
        # 代表 visual 幾何是可用的，不一定會被收成 keyframe，
        # 但至少 recover pose / inlier 幾何合理
        vis_pose_ok = int(row['vis_pose_ok']) == 1
        num_inliers = int(row['num_inliers'])
        match_inlier_ratio = float(row['match_inlier_ratio'])

        is_visual_usable = int(
            matcher_success and
            vis_pose_ok and
            num_inliers > 0 and
            match_inlier_ratio >= 0.0
        )

        # 如果影像特徵缺失，candidate_valid 應降為 0
        if has_missing_feature == 1:
            is_candidate_valid = 0

        return is_candidate_valid, is_visual_usable

    def _build_output_row(self, candidate_row: Dict[str, str]) -> Dict[str, object]:
        row = self._copy_base_fields(candidate_row)

        img0, ok0 = self._read_image_safe(candidate_row.get('image_prev', ''))
        img1, ok1 = self._read_image_safe(candidate_row.get('image_curr', ''))
        images_ok = bool(ok0 and ok1)

        image_quality_features, has_missing_feature = self._compute_image_quality_features(img0, img1)

        is_candidate_valid, is_visual_usable = self._derive_helper_flags(
            row=row,
            images_ok=images_ok,
            has_missing_feature=has_missing_feature,
        )

        row['is_candidate_valid'] = int(is_candidate_valid)
        row['is_visual_usable'] = int(is_visual_usable)
        row['has_missing_feature'] = int(has_missing_feature)

        row.update(image_quality_features)

        return row

    def run(self):
        self._check_inputs()
        self.ensure_dir(self.feature_dir, clean=self.clean_output_dir)

        candidate_rows = self.read_csv_rows(self.candidate_csv_path)

        output_rows = []
        num_total = 0
        num_accepted = 0
        num_rejected = 0
        num_candidate_valid = 0
        num_visual_usable = 0
        num_missing_feature = 0

        for candidate_row in candidate_rows:
            out_row = self._build_output_row(candidate_row)
            output_rows.append(out_row)

            num_total += 1
            if int(out_row['accepted']) == 1:
                num_accepted += 1
            else:
                num_rejected += 1

            if int(out_row['is_candidate_valid']) == 1:
                num_candidate_valid += 1
            if int(out_row['is_visual_usable']) == 1:
                num_visual_usable += 1
            if int(out_row['has_missing_feature']) == 1:
                num_missing_feature += 1

        with open(self.feature_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_HEADER)
            writer.writeheader()
            for row in output_rows:
                writer.writerow(row)

        summary = {
            'candidate_csv_path': str(self.candidate_csv_path),
            'feature_csv': str(self.feature_csv),
            'num_total_pairs': int(num_total),
            'num_accepted_pairs': int(num_accepted),
            'num_rejected_pairs': int(num_rejected),
            'num_candidate_valid': int(num_candidate_valid),
            'num_visual_usable': int(num_visual_usable),
            'num_missing_feature': int(num_missing_feature),
        }

        print('========== Extract Features v2 Finished ==========')
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
        '--candidate_csv_path',
        type=str,
        default='',
        help='若留空，預設 sequence_dir/keyframes/all_candidate_pairs.csv'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='若留空，預設輸出到 sequence_dir/features'
    )
    parser.add_argument(
        '--out_csv_name',
        type=str,
        default='all_candidate_features.csv',
        help='輸出 CSV 名稱，預設 all_candidate_features.csv'
    )
    parser.add_argument(
        '--clean_output_dir',
        action='store_true',
        help='是否先清空輸出資料夾'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    extractor = ReliabilityFeatureExtractor(
        sequence_dir=args.sequence_dir,
        candidate_csv_path=args.candidate_csv_path,
        out_dir=args.out_dir,
        out_csv_name=args.out_csv_name,
        clean_output_dir=args.clean_output_dir,
    )
    extractor.run()


if __name__ == '__main__':
    main()