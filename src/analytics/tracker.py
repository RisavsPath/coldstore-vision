"""Module 3b: Hungarian IoU association of sack instances between scans / keyframes."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def iou_matrix(inst_a: list, inst_b: list, use_masks: bool = True) -> np.ndarray:
    M = np.zeros((len(inst_a), len(inst_b)))
    for i, a in enumerate(inst_a):
        for j, b in enumerate(inst_b):
            bi = box_iou(a.box, b.box)
            M[i, j] = mask_iou(a.mask, b.mask) if (use_masks and bi > 0) else bi
    return M


def hungarian(iou: np.ndarray, thr: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Optimal 1:1 assignment maximising IoU; pairs below thr are rejected."""
    if iou.size == 0:
        return [], list(range(iou.shape[0])), list(range(iou.shape[1]))
    r, c = linear_sum_assignment(1.0 - iou)
    pairs = [(int(i), int(j), float(iou[i, j])) for i, j in zip(r, c) if iou[i, j] >= thr]
    ma = {p[0] for p in pairs}
    mb = {p[1] for p in pairs}
    return pairs, [i for i in range(iou.shape[0]) if i not in ma], [j for j in range(iou.shape[1]) if j not in mb]


def match_instances(inst_a: list, inst_b: list, thr: float = 0.3):
    return hungarian(iou_matrix(inst_a, inst_b), thr)


def match_by_centroid(inst_a: list, inst_b: list, max_cost: float = 0.6):
    """Viewpoint-tolerant association: Hungarian on centroid distance normalised by sack size.

    After a global homography, a 3-D stack seen from a different spot still has parallax, so masks
    of the same sack may barely overlap (IoU fails) while their centres stay close. Cost =
    centre distance / sqrt(Day 1 sack area) + size-mismatch penalty; pairs above max_cost are rejected.
    Returns (pairs [(i, j, cost)], unmatched_a, unmatched_b).
    """
    if not inst_a or not inst_b:
        return [], list(range(len(inst_a))), list(range(len(inst_b)))
    ca = np.array([a.centroid for a in inst_a])
    cb = np.array([b.centroid for b in inst_b])
    sa = np.sqrt(np.array([max(a.area, 1) for a in inst_a], float))
    sb = np.sqrt(np.array([max(b.area, 1) for b in inst_b], float))
    dist = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2) / sa[:, None]
    size_pen = np.abs(np.log(sb[None, :] / sa[:, None])) * 0.5
    cost = dist + size_pen
    r, c = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(cost[i, j])) for i, j in zip(r, c) if cost[i, j] <= max_cost]
    ma, mb = {p[0] for p in pairs}, {p[1] for p in pairs}
    return pairs, [i for i in range(len(inst_a)) if i not in ma], [j for j in range(len(inst_b)) if j not in mb]
