"""Step 5a: draw YOLO-seg labels onto images for the label-review pass.

Green = sack, orange = rack_beam, blue = empty_slot; each instance numbered.
Usage: python -m tools.dataset.render_overlays [--labels DIR] [--images DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tools.dataset.common import ACCEPTED_DIR, LABELS_DIR, OVERLAYS_DIR

COLORS = {0: (60, 200, 60), 1: (0, 140, 255), 2: (255, 120, 0)}  # BGR


def read_label(path: Path) -> list[tuple[int, np.ndarray]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        out.append((int(parts[0]), np.array(parts[1:], dtype=np.float32).reshape(-1, 2)))
    return out


def draw(img: np.ndarray, labels: list[tuple[int, np.ndarray]]) -> np.ndarray:
    h, w = img.shape[:2]
    fill = img.copy()
    for k, (c, xy) in enumerate(labels):
        pts = (xy * [w, h]).astype(np.int32)
        col = COLORS.get(c, (200, 200, 200))
        cv2.fillPoly(fill, [pts], col)
    out = cv2.addWeighted(fill, 0.4, img, 0.6, 0)
    scale = max(w, h) / 1000
    for k, (c, xy) in enumerate(labels):
        pts = (xy * [w, h]).astype(np.int32)
        col = COLORS.get(c, (200, 200, 200))
        cv2.polylines(out, [pts], True, col, max(2, int(2 * scale)))
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(out, str(k + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255),
                    max(2, int(3 * scale)), cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=LABELS_DIR)
    ap.add_argument("--images", type=Path, default=ACCEPTED_DIR)
    ap.add_argument("--out", type=Path, default=OVERLAYS_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for lbl in tqdm(sorted(args.labels.glob("*.txt")), desc="overlays"):
        img_path = args.images / f"{lbl.stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imwrite(str(args.out / f"{lbl.stem}.jpg"), draw(img, read_label(lbl)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        n += 1
    print(f"{n} overlays -> {args.out}")


if __name__ == "__main__":
    main()
