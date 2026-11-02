"""Module 2b: dense Farneback optical flow between Day 1 and registered Day 2 (stack settling)."""
from __future__ import annotations

import cv2
import numpy as np


def farneback(ref: np.ndarray, mov: np.ndarray, cfg: dict) -> np.ndarray:
    """Flow field (HxWx2) such that ref(y, x) ~ mov(y + vy, x + vx). Positive vy = moved down."""
    f = cfg["optical_flow"]
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if ref.ndim == 3 else ref
    g2 = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY) if mov.ndim == 3 else mov
    return cv2.calcOpticalFlowFarneback(g1, g2, None, f["pyr_scale"], f["levels"], f["winsize"],
                                        f["iterations"], f["poly_n"], f["poly_sigma"], 0)


def divergence(flow: np.ndarray) -> np.ndarray:
    dvx = np.gradient(flow[..., 0], axis=1)
    dvy = np.gradient(flow[..., 1], axis=0)
    return dvx + dvy


def slump_metrics(flow: np.ndarray, mask: np.ndarray) -> dict:
    """Mean vertical velocity and divergence inside a (eroded) sack mask.

    Slump signature (spec): v_y > threshold (downward) and divergence < 0 (compression).
    """
    if mask.sum() == 0:
        return {"flow_vy": 0.0, "flow_div": 0.0, "flow_mag": 0.0}
    vy = flow[..., 1][mask]
    mag = np.linalg.norm(flow[mask], axis=1)
    div = divergence(flow)[mask]
    return {"flow_vy": float(np.median(vy)), "flow_div": float(np.mean(div)), "flow_mag": float(np.median(mag))}
