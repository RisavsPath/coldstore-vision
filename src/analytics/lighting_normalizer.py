"""Module 3a: illumination invariance between scans.

- CDF histogram specification of registered Day 2 onto Day 1, per channel, over the whole valid
  frame. Unmatched sacks (missing / new) are excluded so real occupancy changes don't skew the
  mapping. A background-only mapping was rejected: dark walls leave the sack mid-tones unconstrained.
  Localised spoilage is a small fraction of the frame, so it survives normalisation.
- Relative hue: each sack's hue change minus the facility-wide background hue shift.
"""
from __future__ import annotations

import cv2
import numpy as np


def _cdf_lut(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s_hist = np.bincount(src, minlength=256).astype(np.float64)
    r_hist = np.bincount(ref, minlength=256).astype(np.float64)
    s_cdf = np.cumsum(s_hist) / max(s_hist.sum(), 1)
    r_cdf = np.cumsum(r_hist) / max(r_hist.sum(), 1)
    return np.clip(np.searchsorted(r_cdf, s_cdf), 0, 255).astype(np.uint8)


def match_histograms(src: np.ndarray, ref: np.ndarray, src_mask: np.ndarray | None = None,
                     ref_mask: np.ndarray | None = None) -> np.ndarray:
    """Histogram-specify src (BGR) to ref; masks select the pixels the mapping is learnt from."""
    out = np.empty_like(src)
    for c in range(3):
        s = src[..., c][src_mask] if src_mask is not None else src[..., c].ravel()
        r = ref[..., c][ref_mask] if ref_mask is not None else ref[..., c].ravel()
        if s.size < 500 or r.size < 500:
            s, r = src[..., c].ravel(), ref[..., c].ravel()
        out[..., c] = cv2.LUT(src[..., c], _cdf_lut(s, r))
    return out


def circular_mean_hue_deg(img_bgr: np.ndarray, mask: np.ndarray, min_sat: int = 40) -> float | None:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sel = mask & (hsv[..., 1] >= min_sat) & (hsv[..., 2] >= 30)
    if sel.sum() < 30:
        return None
    ang = hsv[..., 0][sel].astype(np.float64) * 2 * np.pi / 180.0  # OpenCV hue 0-180 -> radians
    return float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)


def hue_delta_deg(h1: float | None, h2: float | None) -> float:
    if h1 is None or h2 is None:
        return 0.0
    return float((h2 - h1 + 180) % 360 - 180)


def global_hue_shift(img1: np.ndarray, img2: np.ndarray, bg_mask: np.ndarray, min_sat: int = 40) -> float:
    """Facility-wide hue drift: median of per-pixel wrapped hue differences on registered background.

    Per-pixel (not difference of means) because backgrounds are multi-modal (e.g. blue uprights and
    orange beams), where a circular mean is unstable.
    """
    h1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    h2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    sel = bg_mask & (h1[..., 1] >= min_sat) & (h2[..., 1] >= min_sat) & (h1[..., 2] >= 30) & (h2[..., 2] >= 30)
    if sel.sum() < 100:
        return 0.0
    d = (h2[..., 0][sel].astype(np.float32) - h1[..., 0][sel].astype(np.float32)) * 2.0
    d = (d + 180) % 360 - 180
    return float(np.median(d))


def _union(masks: list[np.ndarray], shape) -> np.ndarray:
    u = np.zeros(shape, bool)
    for m in masks:
        u |= m
    return u


def normalize_pair(img1: np.ndarray, img2_reg: np.ndarray, valid: np.ndarray, sack_masks1: list[np.ndarray],
                   sack_masks2: list[np.ndarray], unmatched_masks: list[np.ndarray] | None = None,
                   enabled: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (normalised Day 2, background mask, residual global hue shift in degrees).

    The hue shift is measured on the *normalised* frame, so it is in the same space as the per-sack
    hue deltas it is subtracted from.
    """
    h, w = img1.shape[:2]
    k = np.ones((15, 15), np.uint8)
    fg = _union(sack_masks1 + sack_masks2, (h, w))
    bg = valid & ~cv2.dilate(fg.astype(np.uint8), k).astype(bool)
    if bg.sum() < 0.02 * h * w:  # walls of sacks with almost no background: fall back to all valid pixels
        bg = valid
    norm = img2_reg
    if enabled:
        learn = valid & ~cv2.dilate(_union(unmatched_masks or [], (h, w)).astype(np.uint8), k).astype(bool)
        norm = match_histograms(img2_reg, img1, learn, learn)
        norm[~valid] = 0
    return norm, bg, global_hue_shift(img1, norm, bg)
