"""Post-process auto-labels: remove sack polygons that mostly lie on people.

Label QA showed the dominant auto-label false positive is clothing/legs/shoes of people carrying
sacks. A COCO person segmenter (yolo11n-seg, class 0) finds them; any sack polygon with more than
--overlap of its area on a person mask is dropped. Human-verified labels are never touched.

Usage: python -m tools.dataset.person_filter [--overlap 0.5]
"""
from __future__ import annotations

import argparse
import contextlib
import json

import cv2
import numpy as np
from tqdm import tqdm

from tools.dataset.common import ACCEPTED_DIR, LABELS_DIR, ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()
    from ultralytics import YOLO

    with contextlib.chdir(ROOT / "models" / "hub"):
        model = YOLO("yolo11n-seg.pt")
    removed_total, touched = 0, 0
    for lbl in tqdm(sorted(LABELS_DIR.glob("*.txt")), desc="person filter"):
        meta_p = lbl.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        if meta.get("human_verified") or meta.get("person_filtered"):
            continue
        img = cv2.imread(str(ACCEPTED_DIR / f"{lbl.stem}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model.predict(img, classes=[0], conf=0.35, retina_masks=True, verbose=False)[0]
        person = np.zeros((h, w), bool)
        if res.masks is not None:
            for m in res.masks.data.cpu().numpy():
                person |= cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        keep, removed = [], 0
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "0" and person.any():
                poly = (np.array(parts[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [poly], 1)
                area = m.sum()
                if area and (person & m.astype(bool)).sum() / area > args.overlap:
                    removed += 1
                    continue
            keep.append(line)
        lbl.write_text("\n".join(keep) + ("\n" if keep else ""))
        meta.update(person_filtered=True, person_removed=removed)
        meta_p.write_text(json.dumps(meta))
        removed_total += removed
        touched += removed > 0
    print(f"removed {removed_total} sack polygons on people across {touched} images")


if __name__ == "__main__":
    main()
