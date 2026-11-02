"""One command: Day 1 + Day 2 recordings in, all five inspection outputs out.

    python process.py --day1 path/to/day1 --day2 path/to/day2 [--notes "removed 2nd sack bottom row"]

Each --dayN accepts any of:
  - a video file (.mp4 / .mov from the iPhone Camera app)
  - a Polycam "Raw Data" export (.zip, or the extracted folder): frames + LiDAR depth + camera poses
  - a folder holding one of the above, or an already-prepared day folder (frames/ or rgb_video.*)
If a folder holds both a video and a Polycam export, the Polycam capture is used (its frames, poses
and depth are time-aligned); the video is kept alongside for reference.

Results go to outputs/run_<timestamp>/ (anomaly_video.mp4, audit_report.html + manifest.json,
side_by_side.mp4, rack_digital_twin.html, panoramic_slider.html) plus run_info.json.
Uses models/weights/best.pt automatically.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve  # noqa: E402
from src.ingestion.polycam_converter import convert as convert_polycam  # noqa: E402
from src.ingestion.polycam_converter import is_polycam_export  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}
log = logging.getLogger("coldstore")


def prepare_day(src: Path, day_dir: Path, prefer_lidar: bool = False) -> dict:
    """Normalise any supported input into data/raw/day_0N/. Returns a description of what was used."""
    src = src.resolve()
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    day_dir.mkdir(parents=True, exist_ok=True)

    candidates = [src] if src.is_file() else sorted(src.iterdir())
    polycam = next((c for c in candidates if is_polycam_export(c)), None)
    if polycam is None and src.is_dir() and is_polycam_export(src):
        polycam = src
    videos = [c for c in candidates if c.is_file() and c.suffix.lower() in VIDEO_EXTS]

    if src.is_dir() and (src / "frames").is_dir() and polycam is None:
        if src != day_dir:
            shutil.copytree(src, day_dir, dirs_exist_ok=True)
        return {"mode": "prepared_folder", "source": str(src)}

    if polycam is not None and (prefer_lidar or not videos):
        info = convert_polycam(polycam, day_dir)
        for v in videos:  # keep the phone video for reference; FrameSource prefers frames/
            shutil.copy2(v, day_dir / f"reference_video{v.suffix.lower()}")
        return {"mode": "polycam", **info}

    if videos:
        v = max(videos, key=lambda p: p.stat().st_size)
        shutil.rmtree(day_dir / "frames", ignore_errors=True)
        for old in day_dir.glob("rgb_video.*"):
            old.unlink()
        shutil.copy2(v, day_dir / f"rgb_video{v.suffix.lower()}")
        for stale in ("camera_poses.csv", "fps.txt"):  # leftovers from an earlier Polycam conversion
            (day_dir / stale).unlink(missing_ok=True)
        shutil.rmtree(day_dir / "depth", ignore_errors=True)
        note = " (Polycam export present but not used; pass --prefer-lidar to use it)" if polycam else ""
        return {"mode": "video", "source": str(v) + note, "poses": False, "depth_frames": 0}

    # last resort: a plain image sequence (e.g. Polycam "Images" export: photos only, no poses/depth)
    for c in [src] + [c for c in candidates if c != src]:
        n = import_image_sequence(c, day_dir)
        if n:
            return {"mode": "images", "source": str(c), "frames": n, "poses": False, "depth_frames": 0}

    raise SystemExit(f"no video, Polycam export or image sequence found in {src}")


def trim_video(day_dir: Path, start: float | None, end: float | None, width: int) -> dict:
    """Cut day_dir/rgb_video.* to [start, end] seconds and downscale to the processing width.

    Lets a run focus on one stack/section of a walk, and decoding a small file is far faster
    than re-decoding 4K footage in every pipeline pass.
    """
    import cv2

    src = next(day_dir.glob("rgb_video.*"))
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    a = int((start or 0) * fps)
    b = int(end * fps) if end else None
    tmp = day_dir / "rgb_video_trim.mp4"
    writer, i, n = None, 0, 0
    while True:
        ok = cap.grab()
        if not ok or (b is not None and i > b):
            break
        if i >= a:
            ok, f = cap.retrieve()
            if ok:
                if f.shape[1] != width:
                    f = cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
                if writer is None:
                    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
                writer.write(f)
                n += 1
        i += 1
    cap.release()
    if writer is None:
        raise SystemExit(f"trim range {start}-{end}s is outside the video ({i / fps:.1f}s long)")
    writer.release()
    src.unlink()
    tmp.rename(day_dir / "rgb_video.mp4")
    return {"trim_s": [start or 0, end or round(i / fps, 2)], "trimmed_frames": n}


def import_image_sequence(src: Path, day_dir: Path, fps: float = 2.0, min_images: int = 5) -> int:
    """Copy a zip/folder of photos (sorted by name) into day_dir/frames. Returns the count (0 = not one)."""
    import tempfile
    import zipfile

    exts = (".jpg", ".jpeg", ".png", ".heic")
    tmp = None
    try:
        if src.is_file() and src.suffix.lower() == ".zip":
            tmp = Path(tempfile.mkdtemp(prefix="imgseq_"))
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            root = tmp
        elif src.is_dir():
            root = src
        else:
            return 0
        imgs = sorted((p for p in root.rglob("*") if p.suffix.lower() in exts), key=lambda p: p.name)
        if len(imgs) < min_images:
            return 0
        if any(p.suffix.lower() == ".heic" for p in imgs):
            raise SystemExit("HEIC photos found - export as JPG (iPhone: Settings > Camera > Formats > "
                             "Most Compatible) or send the video instead")
        shutil.rmtree(day_dir / "frames", ignore_errors=True)
        (day_dir / "frames").mkdir(parents=True)
        for i, p in enumerate(imgs):
            shutil.copy2(p, day_dir / "frames" / f"{i:05d}{p.suffix.lower()}")
        (day_dir / "fps.txt").write_text(str(fps))
        for stale in ("camera_poses.csv",):
            (day_dir / stale).unlink(missing_ok=True)
        shutil.rmtree(day_dir / "depth", ignore_errors=True)
        return len(imgs)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", required=True, type=Path)
    ap.add_argument("--day2", type=Path, default=None, help="omit for a Day 1 baseline + recording check")
    ap.add_argument("--prefer-lidar", action="store_true",
                    help="analyse the Polycam frames/poses/depth instead of the phone video (only for a "
                         "slow side-on Polycam walk; room-scan style captures don't align across days)")
    ap.add_argument("--notes", default="", help="what you changed between the two days (ground truth)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-video", action="store_true", help="skip the two mp4 outputs (faster)")
    for n in (1, 2):
        ap.add_argument(f"--day{n}-start", type=float, default=None, help=f"use day {n} video from this second")
        ap.add_argument(f"--day{n}-end", type=float, default=None, help=f"use day {n} video up to this second")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or ROOT / "outputs" / f"run_{stamp}"
    t0 = time.time()

    days = {}
    for n, src in ((1, args.day1), (2, args.day2)):
        if src is None:
            continue
        day_dir = ROOT / "data" / "raw" / f"day_{n:02d}"
        days[n] = prepare_day(src, day_dir, args.prefer_lidar)
        start, end = getattr(args, f"day{n}_start"), getattr(args, f"day{n}_end")
        if (start is not None or end is not None) and days[n]["mode"] == "video":
            days[n].update(trim_video(day_dir, start, end, cfg["processing"]["width"]))
        log.info("day %d input: %s", n, days[n])
    if args.day2 is None:
        from src.reporting.baseline import run_baseline

        out = args.out or ROOT / "outputs" / f"baseline_{stamp}"
        report = run_baseline(ROOT / "data" / "raw" / "day_01", cfg, out, days[1])
        print(json.dumps(report, indent=2))
        print(f"baseline done in {time.time() - t0:.1f}s -> {out}")
        return
    if days[1]["mode"] != days[2]["mode"]:
        log.warning("day 1 is %s but day 2 is %s - comparable captures give the best alignment",
                    days[1]["mode"], days[2]["mode"])

    from src.pipeline import run_inspection
    from src.reporting.output_generator import generate_outputs
    from src.segmentation.sack_detector import SackDetector

    det = SackDetector(cfg)
    if not det.using_trained:
        log.warning("models/weights/best.pt not found - using the weaker YOLOE fallback detector")
    insp = run_inspection(ROOT / "data" / "raw" / "day_01", ROOT / "data" / "raw" / "day_02", cfg, det)
    if not insp.pairs:
        log.error("no Day 1 / Day 2 keyframes could be aligned - check that both walks cover the same rack "
                  "from a similar side-on viewpoint")
    paths = generate_outputs(insp, cfg, out, skip_video=args.skip_video)

    info = {"run": stamp, "seconds": round(time.time() - t0, 1), "detector": det.source,
            "inputs": days, "notes": args.notes,
            "keyframes": {"day1": len(insp.kf1), "day2": len(insp.kf2), "pairs": len(insp.pairs)},
            "kpis": insp.kpis, "outputs": {k: str(v) for k, v in paths.items()}}
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(insp.kpis, indent=2))
    for k, p in paths.items():
        print(f"{k:18s} {p}")
    print(f"done in {info['seconds']}s -> {out}")


if __name__ == "__main__":
    main()
