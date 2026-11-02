"""Step 5c: human correction round-trip through CVAT.

export: packages images the label reviewer marked `fix` + a random sample of `ok` images as an
        "Ultralytics YOLO Segmentation 1.0" zip -> data/scraped/cvat_export.zip
        In CVAT: create a task with the 3 labels (sack, rack_beam, empty_slot), upload the images,
        then Actions -> Upload annotations -> "Ultralytics YOLO Segmentation 1.0" with this zip.
import: takes the zip exported from CVAT in the same format and overwrites the autolabels for
        those images, marking them human_verified (autolabel.py will never overwrite them).

Usage:
    python -m tools.dataset.cvat_roundtrip export [--ok-sample 0.10]
    python -m tools.dataset.cvat_roundtrip import path/to/cvat_export.zip
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

from tools.dataset.common import ACCEPTED_DIR, CLASS_NAMES, LABEL_REVIEW_CSV, LABELS_DIR, SCRAPED, read_csv


def do_export(ok_sample: float, seed: int) -> None:
    rows = read_csv(LABEL_REVIEW_CSV)
    if not rows:
        raise SystemExit("run the label review first (merge_reviews --mode labels)")
    fix = [r["id"] for r in rows if r["verdict"] == "fix"]
    ok = [r["id"] for r in rows if r["verdict"] == "ok"]
    random.Random(seed).shuffle(ok)
    ids = fix + ok[: int(len(ok) * ok_sample)]
    out = SCRAPED / "cvat_export.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
        z.writestr("data.yaml", f"names:\n{names}\npath: .\ntrain: images/train\n")
        z.writestr("train.txt", "\n".join(f"images/train/{i}.jpg" for i in ids) + "\n")
        for i in ids:
            z.write(ACCEPTED_DIR / f"{i}.jpg", f"images/train/{i}.jpg")
            lbl = LABELS_DIR / f"{i}.txt"
            z.writestr(f"labels/train/{i}.txt", lbl.read_text() if lbl.exists() else "")
    (SCRAPED / "cvat_export_ids.json").write_text(json.dumps(ids))
    print(f"{len(fix)} fix + {len(ids) - len(fix)} ok-sample images -> {out}")


def do_import(zip_path: Path) -> None:
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            p = Path(name)
            if p.suffix != ".txt" or "labels" not in p.parts:
                continue
            stem = p.stem
            if not (ACCEPTED_DIR / f"{stem}.jpg").exists():
                continue
            (LABELS_DIR / f"{stem}.txt").write_bytes(z.read(name))
            (LABELS_DIR / f"{stem}.json").write_text(json.dumps({"human_verified": True, "source": "cvat"}))
            n += 1
    print(f"imported {n} human-verified label files")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ok-sample", type=float, default=0.10)
    e.add_argument("--seed", type=int, default=0)
    i = sub.add_parser("import")
    i.add_argument("zip", type=Path)
    args = ap.parse_args()
    if args.cmd == "export":
        do_export(args.ok_sample, args.seed)
    else:
        do_import(args.zip)


if __name__ == "__main__":
    main()
