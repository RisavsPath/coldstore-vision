"""Stage 1a: fix sheets for agent-guided label correction.

For each image the label reviewer marked `fix`, renders the current overlay (numbered masks) with a
labelled coordinate grid around it, so the reviewer agent can say *how* to fix it: delete mask ids,
split merged masks, and add one grid point per missed sack (SAM 2.1 then draws those masks).

Grid: GRID_COLS columns A.. and GRID_ROWS rows 1.. over the image area; labels sit in a margin so
they never cover content. Point syntax understood by apply_fixes.py:
    "C4"           centre of cell C4
    "C4@0.2,0.7"   x 20% / y 70% inside cell C4

Usage: python -m tools.dataset.render_fix_sheets [--per-batch 6] [--ids id1 id2 ...]
"""
from __future__ import annotations

import argparse
import json
import string

import cv2
import numpy as np

from tools.dataset.common import ACCEPTED_DIR, LABEL_REVIEW_CSV, LABELS_DIR, REVIEW_DIR, ROOT, SHEETS_DIR, read_csv
from tools.dataset.render_overlays import draw, read_label

GRID_COLS, GRID_ROWS = 20, 12
MARGIN = 44
MAX_SIDE = 1500


def cell_to_xy(point: str, w: int, h: int) -> tuple[float, float]:
    """Grid point string -> pixel coordinates in the original image."""
    point = point.strip().upper()
    cell, _, frac = point.partition("@")
    col = string.ascii_uppercase.index(cell[0])
    row = int(cell[1:]) - 1
    if not (0 <= col < GRID_COLS and 0 <= row < GRID_ROWS):
        raise ValueError(point)
    fx, fy = 0.5, 0.5
    if frac:
        fx, fy = (min(max(float(v), 0.0), 1.0) for v in frac.split(","))
    cw, ch = w / GRID_COLS, h / GRID_ROWS
    return min(w - 1, (col + fx) * cw), min(h - 1, (row + fy) * ch)


def render(img: np.ndarray, labels) -> np.ndarray:
    s = min(1.0, MAX_SIDE / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ov = draw(img, labels)
    h, w = ov.shape[:2]
    canvas = np.full((h + 2 * MARGIN, w + 2 * MARGIN, 3), 255, np.uint8)
    canvas[MARGIN:MARGIN + h, MARGIN:MARGIN + w] = ov
    cw, ch = w / GRID_COLS, h / GRID_ROWS
    grid = canvas.copy()
    for c in range(GRID_COLS + 1):
        x = int(MARGIN + c * cw)
        cv2.line(grid, (x, MARGIN), (x, MARGIN + h), (0, 255, 255), 1)
    for r in range(GRID_ROWS + 1):
        y = int(MARGIN + r * ch)
        cv2.line(grid, (MARGIN, y), (MARGIN + w, y), (0, 255, 255), 1)
    canvas = cv2.addWeighted(grid, 0.45, canvas, 0.55, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for c in range(GRID_COLS):
        x = int(MARGIN + (c + 0.5) * cw) - 8
        for y in (MARGIN - 12, MARGIN + h + 32):
            cv2.putText(canvas, string.ascii_uppercase[c], (x, y), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    for r in range(GRID_ROWS):
        y = int(MARGIN + (r + 0.5) * ch) + 8
        for x in (4, MARGIN + w + 4):
            cv2.putText(canvas, str(r + 1), (x, y), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-batch", type=int, default=6)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--prefix", default="fix")
    args = ap.parse_args()

    ids = args.ids or [r["id"] for r in read_csv(LABEL_REVIEW_CSV) if r["verdict"] == "fix"]
    out_dir = SHEETS_DIR / args.prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / args.prefix).mkdir(parents=True, exist_ok=True)
    n = 0
    for b in range(0, len(ids), args.per_batch):
        name = f"{args.prefix}_{b // args.per_batch:03d}"
        images, masks = {}, {}
        for i in ids[b: b + args.per_batch]:
            img = cv2.imread(str(ACCEPTED_DIR / f"{i}.jpg"))
            labels = read_label(LABELS_DIR / f"{i}.txt")
            p = out_dir / f"{i}.jpg"
            cv2.imwrite(str(p), render(img, labels), [cv2.IMWRITE_JPEG_QUALITY, 88])
            images[i] = str(p.relative_to(ROOT)).replace("\\", "/")
            masks[i] = len(labels)
        spec = {"batch": name, "mode": "fix", "grid": {"cols": GRID_COLS, "rows": GRID_ROWS},
                "images": images, "mask_counts": masks,
                "output": f"data/scraped/reviews/{args.prefix}/{name}.json"}
        (out_dir / f"{name}.json").write_text(json.dumps(spec, indent=1))
        n += 1
    print(f"{len(ids)} images -> {n} fix batches in {out_dir}")


if __name__ == "__main__":
    main()
