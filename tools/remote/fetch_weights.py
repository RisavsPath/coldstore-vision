"""Import trained weights from anywhere (Colab Drive folder, a downloaded zip, a run directory).

    python -m tools.remote.fetch_weights "G:/My Drive/coldstore/weights"
    python -m tools.remote.fetch_weights path/to/weights_out.zip

Copies best.pt (+ best.onnx, metrics.json, results.csv) into models/weights/ and checks the model loads
with the expected classes. The pipeline picks best.pt up automatically on the next run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {"sack", "rack_beam", "empty_slot"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    args = ap.parse_args()

    src = args.src
    tmp = None
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    best = next(src.rglob("best.pt"), None) if src.is_dir() else (src if src.name.endswith(".pt") else None)
    if best is None:
        raise SystemExit(f"no best.pt under {args.src}")

    dst = ROOT / "models" / "weights"
    rep = ROOT / "reports" / "training"
    dst.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst / "best.pt")
    for name in ("best.onnx",):
        f = best.parent / name
        if f.exists():
            shutil.copy2(f, dst / name)
    for name in ("metrics.json", "results.csv"):
        f = best.parent / name
        if f.exists():
            shutil.copy2(f, rep / name)

    from ultralytics import YOLO

    names = set(YOLO(str(dst / "best.pt")).names.values())
    if names != EXPECTED:
        print(f"WARNING: model classes {names} != expected {EXPECTED}")
    m = rep / "metrics.json"
    if m.exists():
        print(json.dumps(json.loads(m.read_text()), indent=2))
    print(f"installed {dst / 'best.pt'} - the pipeline will use it automatically")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
