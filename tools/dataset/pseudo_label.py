"""Stage 2: model-in-the-loop pseudo-labels from a trained model (v0 -> v1, and Day 1 frames).

Runs models/weights/best.pt on every accepted image (including previously dropped ones) and merges
its confident masks into the current labels:
  - a model mask overlapping an existing mask (IoU > --dup-iou) is skipped (existing label wins)
  - otherwise it is added (recovers dense rows the auto-labeller missed)
Changed images get meta pseudo_added=N and are cleared of agent_dropped so they get re-reviewed.

Usage: python -m tools.dataset.pseudo_label [--weights models/weights/best.pt] [--conf 0.5]
       python -m tools.dataset.pseudo_label --images data/raw/day_01/frames_2fps --out data/day1_labels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tools.dataset.autolabel import to_polygon
from tools.dataset.common import ACCEPTED_DIR, LABELS_DIR, ROOT
from tools.dataset.apply_fixes import poly_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "weights" / "best.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--dup-iou", type=float, default=0.4)
    ap.add_argument("--images", type=Path, default=ACCEPTED_DIR)
    ap.add_argument("--out", type=Path, default=LABELS_DIR)
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names = model.names
    args.out.mkdir(parents=True, exist_ok=True)
    total_added, changed = 0, 0
    for img_path in tqdm(sorted(args.images.glob("*.jpg")), desc="pseudo-label"):
        lbl = args.out / f"{img_path.stem}.txt"
        meta_p = args.out / f"{img_path.stem}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        if meta.get("human_verified"):
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
        existing = [poly_mask(ln, w, h) for ln in lines]
        res = model.predict(img, conf=args.conf, retina_masks=True, verbose=False)[0]
        added = []
        if res.masks is not None:
            for k, m in enumerate(res.masks.data.cpu().numpy() > 0.5):
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                if any((m & e).sum() / max((m | e).sum(), 1) > args.dup_iou for e in existing):
                    continue
                poly = to_polygon(m)
                if poly is None:
                    continue
                cls = {v: i for i, v in enumerate(["sack", "rack_beam", "empty_slot"])}.get(
                    names[int(res.boxes.cls[k])], 0)
                existing.append(m)
                added.append(f"{cls} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly))
        if not added:
            continue
        lbl.write_text("\n".join(lines + added) + "\n")
        meta.update(pseudo_added=len(added))
        meta.pop("agent_dropped", None)
        meta_p.write_text(json.dumps(meta))
        total_added += len(added)
        changed += 1
    print(f"pseudo-labels: +{total_added} masks across {changed} images")


if __name__ == "__main__":
    main()
