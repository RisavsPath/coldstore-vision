"""Module 2c: aisle orthomosaic by chained affine registration + multi-band blending.

Returns the panorama AND each keyframe's 3x3 transform into mosaic coordinates. The analytics use
those transforms to deduplicate sacks seen in several keyframes and to index bays/tiers along the aisle.
Day 2 keyframes are composited onto the *Day 1* canvas through their Day1 pair, so the two
mosaics are pixel-aligned for the curtain slider.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.registration.homography_warper import _prep


@dataclass
class Mosaic:
    image: np.ndarray                  # BGR panorama
    transforms: list[np.ndarray]       # per-frame 3x3: frame px -> mosaic px (None if not placed)
    valid: np.ndarray                  # HxW bool coverage


def _affine_link(a: np.ndarray, b: np.ndarray, cfg: dict) -> np.ndarray | None:
    """3x3 affine mapping b's pixels into a's frame."""
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(_prep(a), None)
    k2, d2 = sift.detectAndCompute(_prep(b), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d2, d1, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance <= cfg["registration"]["lowe_ratio"] * n.distance]
    if len(good) < 10:
        return None
    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None or inl.sum() < 10:
        return None
    return np.vstack([A, [0, 0, 1]])


def chain_transforms(frames: list[np.ndarray], cfg: dict) -> list[np.ndarray]:
    """Frame i -> frame 0 transforms. A failed link reuses the previous step's motion."""
    T = [np.eye(3)]
    last_step = np.eye(3)
    for i in range(1, len(frames)):
        step = _affine_link(frames[i - 1], frames[i], cfg)
        if step is None:
            step = last_step
        last_step = step
        T.append(T[-1] @ step)
    return T


def _blend(frames, transforms, size, bands: int = 5):
    w, h = size
    try:
        blender = cv2.detail_MultiBandBlender(0, bands)
        blender.prepare((0, 0, w, h))
        cover = np.zeros((h, w), np.uint8)
        for f, T in zip(frames, transforms):
            if T is None:
                continue
            warped = cv2.warpPerspective(f, T, (w, h))
            m = cv2.warpPerspective(np.full(f.shape[:2], 255, np.uint8), T, (w, h), flags=cv2.INTER_NEAREST)
            m = cv2.erode(m, np.ones((5, 5), np.uint8))
            blender.feed(warped.astype(np.int16), m, (0, 0))
            cover |= m
        out, _ = blender.blend(None, None)
        return np.clip(out, 0, 255).astype(np.uint8), cover > 0
    except (cv2.error, AttributeError):
        acc = np.zeros((h, w, 3), np.float32)
        wsum = np.zeros((h, w), np.float32)
        for f, T in zip(frames, transforms):
            if T is None:
                continue
            m = np.full(f.shape[:2], 1, np.float32)
            m = cv2.distanceTransform((m * 255).astype(np.uint8), cv2.DIST_L2, 3) + 1e-3  # feather
            acc += cv2.warpPerspective(f, T, (w, h)).astype(np.float32) * cv2.warpPerspective(m, T, (w, h))[..., None]
            wsum += cv2.warpPerspective(m, T, (w, h))
        out = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)
        return out, wsum > 1e-3


def build_mosaic(frames: list[np.ndarray], cfg: dict, max_width: int = 12000) -> Mosaic:
    T = chain_transforms(frames, cfg)
    corners = []
    for f, t in zip(frames, T):
        h, w = f.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, t).reshape(-1, 2))
    allc = np.concatenate(corners)
    x0, y0 = np.floor(allc.min(0)).astype(int)
    x1, y1 = np.ceil(allc.max(0)).astype(int)
    scale = min(1.0, max_width / max(x1 - x0, 1))
    offset = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]])
    Tm = [offset @ t for t in T]
    size = (int((x1 - x0) * scale), int((y1 - y0) * scale))
    img, valid = _blend(frames, Tm, size)
    return Mosaic(img, Tm, valid)


def composite_onto(base: Mosaic, frames: list[np.ndarray], transforms: list[np.ndarray | None]) -> Mosaic:
    """Blend frames (e.g. Day 2 keyframes, already mapped through their Day 1 pair) onto base's canvas."""
    h, w = base.image.shape[:2]
    img, valid = _blend(frames, transforms, (w, h))
    img[~valid] = 0
    return Mosaic(img, list(transforms), valid)


def project_box(box: np.ndarray, T: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(pts, T).reshape(-1, 2)
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], np.float32)
