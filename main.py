"""ColdStore-Vision end-to-end CLI.

    python main.py --day1 data/raw/day_01 --day2 data/raw/day_02 [--out outputs] [--skip-video]

Detector: models/weights/best.pt when present, otherwise the YOLOE fallback (see pipeline_config.yaml).
`--detector synthetic` uses the colour-key test detector for tools/synth demo data only.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from src.config import load_config, resolve
from src.pipeline import run_inspection
from src.reporting.output_generator import generate_outputs


def make_detector(kind: str, cfg: dict):
    if kind == "synthetic":
        from tools.synth.synthetic_detector import SyntheticColorDetector

        return SyntheticColorDetector()
    from src.segmentation.sack_detector import SackDetector

    return SackDetector(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", required=True)
    ap.add_argument("--day2", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--detector", choices=["yolo", "synthetic"], default="yolo")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    out = resolve(args.out or cfg["outputs"]["dir"])
    t0 = time.time()
    det = make_detector(args.detector, cfg)
    logging.info("detector: %s", getattr(det, "source", type(det).__name__))
    insp = run_inspection(args.day1, args.day2, cfg, det)
    paths = generate_outputs(insp, cfg, out, skip_video=args.skip_video)
    print(json.dumps(insp.kpis, indent=2))
    for k, p in paths.items():
        print(f"{k:18s} {p}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
