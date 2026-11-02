"""Colour-key detector for the synthetic aisle ONLY (test double for SackDetector).

The synthetic renderer draws sacks separated by wall-coloured gaps and beams in fixed saturated
colours, so instances can be keyed out exactly. This exists so the downstream pipeline can be
tested deterministically; it is never used on real footage.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.segmentation.sack_detector import instance_from_mask


class SyntheticColorDetector:
    source = "synthetic colour-key (test only)"

    def __init__(self, min_area: int = 2500):
        self.min_area = min_area

    def detect(self, img: np.ndarray):
        b, g, r = [img[..., k].astype(np.int16) for k in range(3)]
        upright = (b > r + 60) & (b > 90)
        beam = (r > 170) & (b < 60) & (g > 70) & (g < 190) & (r - g > 50)
        wall = (np.abs(b - 35) < 30) & (np.abs(g - 30) < 28) & (np.abs(r - 28) < 28) & (np.maximum(b, np.maximum(g, r)) < 75)
        black = (b + g + r) < 25
        fg = ~(upright | beam | wall | black)
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        out = []
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < self.min_area:
                continue
            m = lab == k
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(bool)
            inst = instance_from_mask("sack", m)
            if inst is not None:
                out.append(inst)
        for mask in (upright, beam):
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] < 800:
                    continue
                inst = instance_from_mask("rack_beam", lab == k)
                if inst is not None:
                    out.append(inst)
        return out
