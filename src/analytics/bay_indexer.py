"""Bay / tier assignment in mosaic coordinates, without markers or a bay config.

Bays: vertical rack_beam detections (tall, thin) split the aisle along X. If fewer than two
uprights are seen, sacks are grouped into columns of ~4 sack-widths.
Tiers: horizontal rack_beam detections split along Y; otherwise sack rows are clustered by
vertical gaps. Tier 1 is the bottom row, Bay 1 is the start of the walk.
"""
from __future__ import annotations

import numpy as np


def _merge_positions(vals: list[float], tol: float) -> list[float]:
    vals = sorted(vals)
    out: list[list[float]] = []
    for v in vals:
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(g)) for g in out]


def _cluster_1d(centers: np.ndarray, gap: float) -> np.ndarray:
    """Labels for sorted 1-D gap clustering (returns labels in the input order)."""
    order = np.argsort(centers)
    labels = np.zeros(len(centers), int)
    k = 0
    for a, b in zip(order[:-1], order[1:]):
        if centers[b] - centers[a] > gap:
            k += 1
        labels[b] = k
    return labels


def index_bays(sack_boxes: list[np.ndarray], beam_boxes: list[np.ndarray], mosaic_w: int, mosaic_h: int,
               min_aspect: float = 3.0) -> list[tuple[int, int]]:
    """Returns (bay, tier), 1-based, for every sack box (mosaic coordinates x1,y1,x2,y2)."""
    if not sack_boxes:
        return []
    S = np.array(sack_boxes, np.float32)
    cx, cy = (S[:, 0] + S[:, 2]) / 2, (S[:, 1] + S[:, 3]) / 2
    sw, sh = np.median(S[:, 2] - S[:, 0]), np.median(S[:, 3] - S[:, 1])

    verts, horiz = [], []
    for b in beam_boxes:
        w, h = b[2] - b[0], b[3] - b[1]
        if h >= min_aspect * w:
            verts.append((b[0] + b[2]) / 2)
        elif w >= min_aspect * h:
            horiz.append((b[1] + b[3]) / 2)

    uprights = _merge_positions(verts, tol=0.5 * sw)
    if len(uprights) >= 2:
        bay = np.searchsorted(np.array(uprights), cx)   # 0 = before first upright
        _, bay = np.unique(bay, return_inverse=True)     # compact to occupied bays
    else:
        bay_w = max(4 * sw, 1.0)
        bay = (cx // bay_w).astype(int)
        _, bay = np.unique(bay, return_inverse=True)

    shelves = _merge_positions(horiz, tol=0.5 * sh)
    tier = np.zeros(len(S), int)
    if len(shelves) >= 1:
        t = len(shelves) - np.searchsorted(np.array(shelves), cy)  # count shelves below
        _, tier = np.unique(t, return_inverse=True)
    else:
        # cluster rows within each bay; bottom row = tier 1
        for b in np.unique(bay):
            idx = np.nonzero(bay == b)[0]
            lab = _cluster_1d(cy[idx], gap=0.6 * sh)
            # higher y (lower in image) = lower tier
            order = {lv: r for r, lv in enumerate(sorted(set(lab), key=lambda lv: -cy[idx][lab == lv].mean()))}
            tier[idx] = [order[lv] for lv in lab]
    return [(int(b) + 1, int(t) + 1) for b, t in zip(bay, tier)]
