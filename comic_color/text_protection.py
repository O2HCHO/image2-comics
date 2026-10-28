"""Comic segmentation and conservative image registration."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

VERSION = "ctd-balloon-strokes-3"


class NeedsReview(ValueError):
    pass


class AlignmentError(NeedsReview):
    """Keep measured failure reasons and recovery steps available to the UI/log."""

    def __init__(self, original_size, generated_size, scores):
        w, h = original_size
        gw, gh = generated_size
        lines = [f"无法安全恢复文字：原稿 {w}×{h}，生成图 {gw}×{gh}。"]
        if not scores:
            lines.append("没有找到可靠的对应线条；生成图可能改变了裁切、比例或构图。")
        else:
            best = max(scores, key=lambda item: item['edge_score'] + .1 * item['coverage'])
            lines.append(f"最佳方案：画面覆盖 {best['coverage']:.0%}（需 ≥55%），"
                         f"双向线条匹配 {best['edge_score']:.0%}（需 ≥43%）。")
            failed = [str(i + 1) for i, score in enumerate(best['local_scores']) if 0 <= score < .32]
            if failed:
                lines.append("保护区域 " + "、".join(failed) + " 附近线条匹配不足32%，可能发生局部重绘或位移。")
        lines.extend([
            "右侧可查看原始上色图（尚未恢复文字）。原稿和已有结果均保留。",
            "修复步骤：",
            "1. 对照两图，检查是否同一页，以及分镜、边缘裁切和气泡位置。",
            "2. 仅某处文字有问题时，取消该处保护区域，再点击“预览融合”；该处文字将保留生成图版本。",
            "3. 若整页构图或气泡明显改变，返回监修页修改提示词，要求严格保留构图，再仅重新生成此页（会调用 API）。",
            "外扩/羽化只调整文字笔画，不能修复画面错位。确认预览正确后再保存并修复。",
        ])
        self.scores = scores
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class TextOptions:
    enabled: bool = True
    expand: int = 1
    feather: int = 1

    def values(self):
        return {**asdict(self), "algorithm": VERSION}


def rgb(path_or_bytes) -> np.ndarray:
    with Image.open(BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def png(array: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(array).save(stream, "PNG")
    return stream.getvalue()


def cache_path(output: Path, source: Path) -> Path:
    key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:24]
    return output / ".comic-color/text" / (key + ".json")


def edge_quality(original, aligned, coverage, boxes) -> tuple[float, list[float]]:
    a = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), 70, 170)
    b = cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY), 70, 170)
    distance = cv2.distanceTransform(255 - b, cv2.DIST_L2, 3)
    points = (a > 0) & (coverage > 0)
    target_points = (b > 0) & (coverage > 0)
    # Exclude text from global registration scoring; those strokes are expected to change.
    for x1, y1, x2, y2 in boxes:
        points[max(0, y1):y2, max(0, x1):x2] = False
        target_points[max(0, y1):y2, max(0, x1):x2] = False
    score = float(np.mean(distance[points] < 4)) if np.count_nonzero(points) > 100 else 0.
    reverse_distance = cv2.distanceTransform(255 - a, cv2.DIST_L2, 3)
    reverse_score = float(np.mean(reverse_distance[target_points] < 4)) if np.count_nonzero(target_points) > 100 else 0.
    score = min(score, reverse_score)
    local = []
    for x1, y1, x2, y2 in boxes:
        pad = max(20, min(65, max(x2 - x1, y2 - y1) // 2))
        ring = np.zeros_like(points)
        ring[max(0, y1-pad):min(points.shape[0], y2+pad), max(0, x1-pad):min(points.shape[1], x2+pad)] = True
        ring[max(0, y1):y2, max(0, x1):x2] = False
        region = points & ring
        local.append(float(np.mean(distance[region] < 5)) if np.count_nonzero(region) >= 12 else -1.)
    return score, local


def align(original: np.ndarray, colored: np.ndarray, boxes: list) -> tuple[np.ndarray, dict]:
    h, w = original.shape[:2]
    gh, gw = colored.shape[:2]
    candidates = []
    # Image services can resize the two axes independently. Validate this
    # candidate using the same global/local checks as feature registration.
    if abs(w / h - gw / gh) / (w / h) < .15:
        candidates.append(np.array([[w / gw, 0, 0], [0, h / gh, 0]], dtype=float))
    sift = cv2.SIFT_create(nfeatures=6500)
    for limit in (1200, 2000):
        s1, s2 = min(1., limit / max(w, h)), min(1., limit / max(gw, gh))
        a = cv2.resize(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), None, fx=s1, fy=s1)
        b = cv2.resize(cv2.cvtColor(colored, cv2.COLOR_RGB2GRAY), None, fx=s2, fy=s2)
        mask = np.full(a.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            mask[int(y1*s1):int(y2*s1)+1, int(x1*s1):int(x2*s1)+1] = 0
        ka, da = sift.detectAndCompute(a, mask)
        kb, db = sift.detectAndCompute(b, None)
        if da is None or db is None or len(da) < 12 or len(db) < 12:
            continue
        matches = cv2.BFMatcher().knnMatch(db, da, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < .72*n.distance]
        if len(good) < 12:
            continue
        p = np.float32([kb[m.queryIdx].pt for m in good]) / s2
        q = np.float32([ka[m.trainIdx].pt for m in good]) / s1
        matrix, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=max(3, max(w,h)*.003), maxIters=5000)
        if matrix is None or inliers.sum() < 10 or inliers.mean() < .3:
            continue
        angle = abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        if angle <= 10 and np.linalg.det(matrix[:, :2]) > 0:
            locations = q[inliers.ravel() > 0]
            if np.ptp(locations[:, 0]) > w*.25 and np.ptp(locations[:, 1]) > h*.25:
                candidates.append(matrix)
    best = None
    scores = []
    for matrix in candidates:
        aligned = cv2.warpAffine(colored, matrix, (w, h), flags=cv2.INTER_LANCZOS4)
        coverage = cv2.warpAffine(np.full((gh, gw), 255, np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
        global_score, local = edge_quality(original, aligned, coverage, boxes)
        covered = float(np.mean(coverage > 0))
        scores.append({"edge_score": global_score, "local_scores": local, "coverage": covered})
        # Reject a layout change even if a small patch happens to match.
        if covered < .55 or global_score < .43 or any(0 <= score < .32 for score in local):
            continue
        quality = global_score + .1*covered
        if best is None or quality > best[0]:
            aligned[coverage == 0] = original[coverage == 0]
            best = (quality, aligned, {"matrix": matrix.tolist(), "edge_score": global_score,
                                      "local_scores": local, "coverage": covered})
    if best is None:
        raise AlignmentError((w, h), (gw, gh), scores)
    return best[1], best[2]



from .text_pipeline import detection, detect_regions, detect_page, regions, make_mask, restoration_mask, fuse
