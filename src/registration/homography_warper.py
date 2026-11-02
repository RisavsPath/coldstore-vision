"""Module 2a: SIFT + Lowe ratio + RANSAC homography registration of Day 2 onto Day 1."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Registration:
    H: np.ndarray | None       # maps moving (Day 2) pixels -> reference (Day 1) pixels
    inliers: int
    matches: int
    ok: bool

    def warp(self, img: np.ndarray, shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpPerspective(img, self.H, (shape[1], shape[0]), flags=flags, borderMode=cv2.BORDER_CONSTANT)

    def valid_mask(self, src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> np.ndarray:
        ones = np.full(src_shape[:2], 255, np.uint8)
        return self.warp(ones, dst_shape, nearest=True) > 0


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _prep(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return _clahe.apply(g)  # dim cold-store frames: boost local contrast for features


def make_detector(kind: str = "sift"):
    if kind == "orb":
        return cv2.ORB_create(4000), cv2.NORM_HAMMING
    return cv2.SIFT_create(nfeatures=6000), cv2.NORM_L2


def register(ref: np.ndarray, mov: np.ndarray, cfg: dict, ref_mask: np.ndarray | None = None,
             mov_mask: np.ndarray | None = None) -> Registration:
    """Estimate H with mov -> ref. Optional masks restrict features (e.g. to rigid rack uprights)."""
    rcfg = cfg["registration"]
    det, norm = make_detector(rcfg["feature"])
    m1 = None if ref_mask is None else ref_mask.astype(np.uint8) * 255
    m2 = None if mov_mask is None else mov_mask.astype(np.uint8) * 255
    k1, d1 = det.detectAndCompute(_prep(ref), m1)
    k2, d2 = det.detectAndCompute(_prep(mov), m2)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        if ref_mask is not None or mov_mask is not None:
            return register(ref, mov, cfg)  # too few features on masked region: use full frame
        return Registration(None, 0, 0, False)
    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(d2, d1, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance <= rcfg["lowe_ratio"] * n.distance]
    if len(good) < 8:
        return Registration(None, 0, len(good), False)
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, rcfg["ransac_reproj_px"])
    n_in = int(inl.sum()) if inl is not None else 0
    ok = H is not None and n_in >= rcfg["min_inliers"] and _sane(H, ref.shape)
    if not ok and (ref_mask is not None or mov_mask is not None):
        return register(ref, mov, cfg)
    return Registration(H, n_in, len(good), ok)


def _sane(H: np.ndarray, shape) -> bool:
    """Reject degenerate homographies (flips, extreme scale/perspective)."""
    det = np.linalg.det(H[:2, :2])
    if det <= 0.2 or det > 5:
        return False
    return abs(H[2, 0]) < 2e-3 and abs(H[2, 1]) < 2e-3


def _match_cached(f_ref, f_mov, cfg: dict) -> Registration:
    """Homography mov -> ref from precomputed (keypoints, descriptors, norm)."""
    k1, d1, norm = f_ref
    k2, d2, _ = f_mov
    rcfg = cfg["registration"]
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return Registration(None, 0, 0, False)
    knn = cv2.BFMatcher(norm).knnMatch(d2, d1, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance <= rcfg["lowe_ratio"] * n.distance]
    if len(good) < 8:
        return Registration(None, 0, len(good), False)
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, rcfg["ransac_reproj_px"] / 2)  # half-res features
    n_in = int(inl.sum()) if inl is not None else 0
    ok = H is not None and n_in >= rcfg["min_inliers"] and _sane(H, (1, 1))
    return Registration(H, n_in, len(good), ok)


def pair_keyframes(kf1: list, kf2: list, cfg: dict) -> list[tuple[int, int, Registration]]:
    """For each Day 1 keyframe pick the Day 2 keyframe that best overlaps it.

    With poses on both days: nearest camera position, then verified by registration.
    Without poses: search a window around the proportional index and keep the max-inlier match.
    """
    pairs = []
    have_pose = all(k.pose is not None for k in kf1 + kf2)
    small = lambda im: cv2.resize(im, None, fx=0.5, fy=0.5)  # noqa: E731
    det, norm = make_detector(cfg["registration"]["feature"])
    if cfg["registration"]["feature"] == "sift":
        det = cv2.SIFT_create(nfeatures=1500)  # pairing search only needs the strongest features
    cache: dict = {}

    def feats(idx: int, day: int, img: np.ndarray):
        if (day, idx) not in cache:
            k, d = det.detectAndCompute(_prep(small(img)), None)
            cache[(day, idx)] = (k, d, norm)
        return cache[(day, idx)]
    for i, k1 in enumerate(kf1):
        if have_pose:
            d = [np.linalg.norm(k2.pose - k1.pose) for k2 in kf2]
            cand = list(np.argsort(d)[:3])
        else:
            # no poses: the two walks can differ in speed, length and start point, so search every
            # Day 2 keyframe (features cached once per image keeps this cheap)
            cand = list(range(len(kf2)))
        best = None
        f1 = feats(i, 1, k1.image)
        for j in cand:
            r = _match_cached(f1, feats(j, 2, kf2[j].image), cfg)
            if r.ok and (best is None or r.inliers > best[1].inliers):
                best = (j, r)
        if best is None:
            continue
        j = best[0]
        full = register(k1.image, kf2[j].image, cfg)
        if full.ok:
            pairs.append((i, int(j), full))
    return pairs
