import os
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent


def _resolve_lightglue_repo() -> Path:
    env_path = os.environ.get('LIGHTGLUE_REPO', '').strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([
        CURRENT_DIR / 'LightGlue',
        CURRENT_DIR.parent / 'LightGlue',
        Path.cwd() / 'LightGlue',
    ])

    for path in candidates:
        if (path / 'lightglue').exists():
            return path.resolve()

    tried = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(
        '找不到 LightGlue repo。\n'
        '請確認 LightGlue 已下載，或設定環境變數 LIGHTGLUE_REPO。\n'
        f'已嘗試路徑:\n{tried}'
    )


LIGHTGLUE_REPO = _resolve_lightglue_repo()
if str(LIGHTGLUE_REPO) not in sys.path:
    sys.path.insert(0, str(LIGHTGLUE_REPO))

from lightglue import LightGlue, SuperPoint  # noqa: E402
from lightglue.utils import numpy_image_to_torch, rbd  # noqa: E402


class LightGlueMatcher:
    def __init__(self, max_num_keypoints: int = 2048, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_num_keypoints = int(max_num_keypoints)

        self.extractor = SuperPoint(
            max_num_keypoints=self.max_num_keypoints,
        ).eval().to(self.device)

        self.matcher = LightGlue(
            features='superpoint',
        ).eval().to(self.device)

    @torch.inference_mode()
    def match(self, img0_bgr: np.ndarray, img1_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        if img0_bgr is None or img1_bgr is None:
            raise ValueError('img0_bgr 或 img1_bgr 是 None')

        img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)

        t0 = numpy_image_to_torch(img0_rgb).to(self.device)
        t1 = numpy_image_to_torch(img1_rgb).to(self.device)

        feats0 = self.extractor.extract(t0)
        feats1 = self.extractor.extract(t1)
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})

        feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

        kpts0 = feats0['keypoints']
        kpts1 = feats1['keypoints']
        matches = matches01.get('matches', None)
        scores = matches01.get('scores', None)

        keypoints0 = kpts0.detach().cpu().numpy().astype(np.float32)
        keypoints1 = kpts1.detach().cpu().numpy().astype(np.float32)

        if matches is None or len(matches) == 0:
            empty_pts = np.empty((0, 2), dtype=np.float32)
            empty_scores = np.empty((0,), dtype=np.float32)
            empty_mask = np.empty((0,), dtype=bool)
            return {
                'num_keypoints0': int(len(keypoints0)),
                'num_keypoints1': int(len(keypoints1)),
                'keypoints0': keypoints0,
                'keypoints1': keypoints1,
                'num_matches': 0,
                'mkpts0': empty_pts,
                'mkpts1': empty_pts,
                'pts0': empty_pts,
                'pts1': empty_pts,
                'match_scores': empty_scores,
                'mean_match_score': -1.0,
                'inlier_mask': empty_mask,
                'inlier_ratio': 0.0,
            }

        mkpts0 = kpts0[matches[:, 0]].detach().cpu().numpy().astype(np.float32)
        mkpts1 = kpts1[matches[:, 1]].detach().cpu().numpy().astype(np.float32)

        if scores is None:
            match_scores = np.full((len(matches),), -1.0, dtype=np.float32)
        else:
            match_scores = scores.detach().cpu().numpy().astype(np.float32).reshape(-1)

        inlier_mask, inlier_ratio = self._compute_fundamental_inlier_mask_and_ratio(mkpts0, mkpts1)
        valid_scores = match_scores[match_scores >= 0.0]
        mean_match_score = float(valid_scores.mean()) if len(valid_scores) > 0 else -1.0

        return {
            'num_keypoints0': int(len(keypoints0)),
            'num_keypoints1': int(len(keypoints1)),
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'num_matches': int(len(matches)),
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'pts0': mkpts0,
            'pts1': mkpts1,
            'match_scores': match_scores,
            'mean_match_score': mean_match_score,
            'inlier_mask': inlier_mask,
            'inlier_ratio': float(inlier_ratio),
        }

    def _compute_fundamental_inlier_mask_and_ratio(self, pts0: np.ndarray, pts1: np.ndarray):
        if len(pts0) < 8:
            mask = np.zeros((len(pts0),), dtype=bool)
            return mask, 0.0

        _, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.99,
        )

        if mask is None:
            mask = np.zeros((len(pts0),), dtype=bool)
            return mask, 0.0

        mask = mask.reshape(-1).astype(bool)
        total = int(len(mask))
        if total == 0:
            return mask, 0.0

        inliers = int(mask.sum())
        inlier_ratio = float(inliers) / float(total)
        return mask, inlier_ratio
