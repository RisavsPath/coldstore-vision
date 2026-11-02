"""Smoke-test pair from a REAL sack photo: two panning .mp4 walkthroughs with known changes.

Day 2 differs by: one sack removed (inpainted), one sack with a dark rot patch, warmer/dimmer
lighting and a slightly different camera path. Exercises process.py end-to-end with the trained
detector on real sack texture (the colour-key synthetic aisle can't do that).

    python -m tools.synth.make_real_pair --image data/scraped/accepted/<id>.jpg --out data/test_pair
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FRAME = (1280, 720)


def pan_video(canvas: np.ndarray, path: Path, day: int, seconds: float = 4.0, fps: int = 30) -> None:
    h, w = canvas.shape[:2]
    view_h = h
    view_w = int(view_h * FRAME[0] / FRAME[1])
    if view_w >= w:  # image not wide enough: zoom in
        view_w = int(w * 0.6)
        view_h = int(view_w * FRAME[1] / FRAME[0])
    n = int(seconds * fps)
    x0, x1 = (0, w - view_w) if day == 1 else (int(0.02 * w), w - view_w - int(0.01 * w))
    dy = 0 if day == 1 else int(0.02 * h)
    roll = 0.0 if day == 1 else 0.8
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, FRAME)
    for i in range(n):
        x = x0 + (x1 - x0) * i / max(n - 1, 1)
        cx, cy = x + view_w / 2, min(h - view_h / 2, view_h / 2 + dy)
        s = FRAME[0] / view_w
        M = cv2.getRotationMatrix2D((cx, cy), roll, s)
        M[0, 2] += FRAME[0] / 2 - cx
        M[1, 2] += FRAME[1] / 2 - cy
        f = cv2.warpAffine(canvas, M, FRAME, borderMode=cv2.BORDER_REFLECT)
        noise = np.random.default_rng(day * 1000 + i).normal(0, 2, f.shape)
        wr.write(np.clip(f + noise, 0, 255).astype(np.uint8))
    wr.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/test_pair"))
    ap.add_argument("--weights", default="models/weights/best.pt")
    args = ap.parse_args()
    from ultralytics import YOLO

    img = cv2.imread(str(args.image))
    res = YOLO(args.weights).predict(img, conf=0.5, retina_masks=True, verbose=False)[0]
    if res.masks is None or len(res.masks) < 3:
        raise SystemExit("need an image where the model finds at least 3 sacks")
    masks = res.masks.data.cpu().numpy() > 0.5
    areas = masks.reshape(len(masks), -1).sum(1)
    order = np.argsort(-areas)
    h, w = img.shape[:2]
    cx = [np.nonzero(m)[1].mean() / w for m in masks]
    central = [k for k in order if 0.3 < cx[k] < 0.7]
    rm, rot = (central + list(order))[:2]

    day2 = img.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    hole = cv2.dilate(masks[rm].astype(np.uint8) * 255, k)
    # MISSING: a removed sack exposes the dark gap / back wall behind it (inpainting jute over jute
    # would just paint a fake sack and test nothing)
    gap = np.clip(np.array((38, 40, 46)) + np.random.default_rng(5).normal(0, 9, (h, w, 1)), 0, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(hole, (0, 0), 4).astype(np.float32)[..., None] / 255.0
    day2 = (day2 * (1 - soft) + gap * soft).astype(np.uint8)
    rng = np.random.default_rng(3)
    ys, xs = np.nonzero(masks[rot])
    patch = np.zeros((h, w), np.uint8)
    for _ in range(5):
        i = rng.integers(len(xs))
        cv2.ellipse(patch, (int(xs[i]), int(ys[i])), (int(rng.integers(15, 40)), int(rng.integers(10, 30))),
                    float(rng.uniform(0, 180)), 0, 360, 255, -1)
    patch = (cv2.GaussianBlur(patch, (0, 0), 3) > 128) & masks[rot]
    rotten = np.clip(np.array((35, 60, 95)) + rng.normal(0, 25, (h, w, 1)), 0, 255).astype(np.uint8)
    day2[patch] = rotten[patch]  # CRITICAL spoilage
    day2 = np.clip(day2.astype(np.float32) * 0.85 * np.array([0.95, 1.0, 1.06]), 0, 255).astype(np.uint8)

    for d in (1, 2):
        (args.out / f"day{d}").mkdir(parents=True, exist_ok=True)
    pan_video(img, args.out / "day1" / "walk.mp4", 1)
    pan_video(day2, args.out / "day2" / "walk.mp4", 2)
    gt = {"image": str(args.image), "removed_sack_centroid_x": round(float(cx[rm]), 3),
          "rotten_sack_centroid_x": round(float(cx[rot]), 3), "sacks_detected_day1_image": int(len(masks))}
    (args.out / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    print(gt)


if __name__ == "__main__":
    main()
