"""Validate a YOLO-seg dataset directory: label syntax, class ids, coordinate range, pairing."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tools.dataset.common import CLASS_NAMES, DATASET_DIR, IMG_EXTS


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.exists():
            errors.append(f"missing {img_dir}")
            continue
        imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()
        for s in sorted(imgs - lbls):
            errors.append(f"{split}/{s}: image without label")
        for s in sorted(lbls - imgs):
            errors.append(f"{split}/{s}: label without image")
        cls = Counter()
        for lp in lbl_dir.glob("*.txt"):
            for ln, line in enumerate(lp.read_text().splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    c = int(parts[0])
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"{lp.name}:{ln} non-numeric")
                    continue
                if not 0 <= c < len(CLASS_NAMES):
                    errors.append(f"{lp.name}:{ln} bad class {c}")
                if len(vals) < 6 or len(vals) % 2:
                    errors.append(f"{lp.name}:{ln} polygon needs >=3 xy pairs, got {len(vals)} values")
                if any(v < 0 or v > 1 for v in vals):
                    errors.append(f"{lp.name}:{ln} coords outside [0,1]")
                cls[CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else c] += 1
        print(f"{split}: {len(imgs)} images, instances {dict(cls)}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?", default=DATASET_DIR)
    args = ap.parse_args()
    errs = validate(args.root)
    for e in errs[:50]:
        print("ERROR", e)
    print(f"{len(errs)} errors")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
