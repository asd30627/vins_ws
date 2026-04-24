from pathlib import Path

import cv2
import numpy as np


class MatchVisualizer:
    def __init__(self, max_draw_matches=120):
        if max_draw_matches <= 0:
            raise ValueError('max_draw_matches 必須 > 0')
        self.max_draw_matches = max_draw_matches

    def to_bgr(self, img):
        if img is None:
            raise ValueError('img 是 None')

        if not isinstance(img, np.ndarray):
            raise TypeError('img 必須是 numpy.ndarray')

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3 and img.shape[2] == 3:
            return img.copy()

        raise ValueError(f'不支援的影像 shape: {img.shape}')

    def get_result_points(self, result):
        if result is None:
            raise ValueError('result 是 None')

        required_keys = ['mkpts0', 'mkpts1', 'inlier_mask']
        for k in required_keys:
            if k not in result:
                raise KeyError(f"result 缺少必要欄位: '{k}'")

        pts0 = result['mkpts0']
        pts1 = result['mkpts1']
        inlier_mask = result['inlier_mask']
        return pts0, pts1, inlier_mask

    def draw(
        self,
        img0,
        img1,
        pts0,
        pts1,
        inlier_mask=None,
        max_draw_matches=None,
        title_text=''
    ):
        img0 = self.to_bgr(img0)
        img1 = self.to_bgr(img1)

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        H = max(h0, h1)
        W = w0 + w1

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:h0, :w0] = img0
        canvas[:h1, w0:w0 + w1] = img1

        pts0 = np.asarray(pts0, dtype=np.float32)
        pts1 = np.asarray(pts1, dtype=np.float32)

        if len(pts0) != len(pts1):
            raise ValueError('pts0 和 pts1 長度不一致')

        if len(pts0) == 0:
            if title_text:
                cv2.putText(
                    canvas,
                    title_text,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            return canvas

        if inlier_mask is None:
            raise ValueError('inlier_mask 是 None')

        inlier_mask = np.asarray(inlier_mask).reshape(-1).astype(bool)

        if len(inlier_mask) != len(pts0):
            raise ValueError('inlier_mask 長度和 pts0 不一致')

        max_draw = self.max_draw_matches if max_draw_matches is None else max_draw_matches
        if max_draw <= 0:
            raise ValueError('max_draw_matches 必須 > 0')

        total = len(pts0)
        if total > max_draw:
            draw_indices = np.linspace(0, total - 1, max_draw, dtype=int)
        else:
            draw_indices = np.arange(total)

        for idx in draw_indices:
            x0, y0 = pts0[idx]
            x1, y1 = pts1[idx]

            p0 = (int(round(x0)), int(round(y0)))
            p1 = (int(round(x1 + w0)), int(round(y1)))

            color = (0, 255, 0) if inlier_mask[idx] else (0, 0, 255)

            cv2.circle(canvas, p0, 3, color, -1)
            cv2.circle(canvas, p1, 3, color, -1)
            cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

        if title_text:
            cv2.putText(
                canvas,
                title_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        return canvas

    def draw_from_result(
        self,
        img0,
        img1,
        result,
        max_draw_matches=None,
        title_text=''
    ):
        pts0, pts1, inlier_mask = self.get_result_points(result)

        return self.draw(
            img0=img0,
            img1=img1,
            pts0=pts0,
            pts1=pts1,
            inlier_mask=inlier_mask,
            max_draw_matches=max_draw_matches,
            title_text=title_text
        )

    def save(
        self,
        out_path,
        img0,
        img1,
        pts0,
        pts1,
        inlier_mask=None,
        max_draw_matches=None,
        title_text=''
    ):
        vis = self.draw(
            img0=img0,
            img1=img1,
            pts0=pts0,
            pts1=pts1,
            inlier_mask=inlier_mask,
            max_draw_matches=max_draw_matches,
            title_text=title_text
        )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(str(out_path), vis)
        if not ok:
            raise IOError(f'影像存檔失敗: {out_path}')

    def save_from_result(
        self,
        out_path,
        img0,
        img1,
        result,
        max_draw_matches=None,
        title_text=''
    ):
        vis = self.draw_from_result(
            img0=img0,
            img1=img1,
            result=result,
            max_draw_matches=max_draw_matches,
            title_text=title_text
        )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(str(out_path), vis)
        if not ok:
            raise IOError(f'影像存檔失敗: {out_path}')