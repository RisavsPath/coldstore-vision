"""Convert a Polycam LiDAR "Raw Data" export into the pipeline's day-folder layout.

Polycam raw export (zip or extracted folder) contains, under `keyframes/`:
    images/ or corrected_images/      <id>.jpg      RGB keyframes
    cameras/ or corrected_cameras/    <id>.json     pose t_00..t_23 (3x4 camera-to-world), timestamp, fx, fy...
    depth/                            <id>.png      16-bit depth in millimetres (LiDAR resolution)
Output (what keyframe_extractor.FrameSource expects):
    <out>/frames/00000.jpg ...   <out>/fps.txt   <out>/camera_poses.csv   <out>/depth/00000.png ...

Poses are re-timed onto a regular timeline (i / fps) so frame index and pose time always agree.
Files are matched by stem; anything missing (no cameras, no depth) is simply skipped.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png")


def _find_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:  # first preference wins (corrected_* over raw)
        hits = sorted(p for p in root.rglob(name) if p.is_dir())
        if hits:
            return hits[0]
    return None


def is_polycam_export(path: Path) -> bool:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            return any("keyframes/" in n and n.lower().endswith(IMG_EXTS) for n in z.namelist())
    return path.is_dir() and _find_dir(path, ("keyframes",)) is not None


def _num_key(p: Path):
    try:
        return (0, float(p.stem))
    except ValueError:
        return (1, p.stem)


def _pose(cam: dict) -> tuple[float, float, float] | None:
    try:
        return float(cam["t_03"]), float(cam["t_13"]), float(cam["t_23"])
    except (KeyError, TypeError, ValueError):
        m = cam.get("transform") or cam.get("pose")
        if m is not None:
            a = np.array(m, dtype=float).reshape(-1)
            if a.size >= 12:
                return (a[3], a[7], a[11]) if a.size in (12, 16) else None
    return None


def _timestamp(cam: dict) -> float | None:
    for k in ("timestamp", "time", "t"):
        if k in cam:
            try:
                return float(cam[k])
            except (TypeError, ValueError):
                pass
    return None


def convert(src: str | Path, out_dir: str | Path) -> dict:
    """Convert a Polycam export (zip or folder) into out_dir. Returns a summary dict."""
    src, out_dir = Path(src), Path(out_dir)
    tmp = None
    if src.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="polycam_"))
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        root = tmp
    else:
        root = src
    try:
        kf = _find_dir(root, ("keyframes",)) or root
        img_dir = _find_dir(kf, ("corrected_images", "images"))
        cam_dir = _find_dir(kf, ("corrected_cameras", "cameras"))
        dep_dir = _find_dir(kf, ("depth",))
        if img_dir is None:
            raise FileNotFoundError(f"no keyframe images found in {src}")
        images = sorted((p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS), key=_num_key)

        for sub in ("frames", "depth"):
            shutil.rmtree(out_dir / sub, ignore_errors=True)
        (out_dir / "frames").mkdir(parents=True, exist_ok=True)

        cams, stamps = [], []
        for p in images:
            cam = None
            if cam_dir is not None and (cam_dir / f"{p.stem}.json").exists():
                cam = json.loads((cam_dir / f"{p.stem}.json").read_text())
            cams.append(cam)
            stamps.append(_timestamp(cam) if cam else None)

        # capture rate from timestamps (Polycam stores seconds or nanoseconds); default 2 fps keyframes
        fps = 2.0
        ts = [t for t in stamps if t is not None]
        if len(ts) >= 2:
            dur = max(ts) - min(ts)
            if dur > 1e6:      # nanoseconds
                dur /= 1e9
            elif dur > 1e3:    # milliseconds
                dur /= 1e3
            if dur > 0:
                fps = max(0.5, min(30.0, (len(ts) - 1) / dur))
        (out_dir / "fps.txt").write_text(f"{fps:.4f}")

        rows, n_depth = [], 0
        for i, (p, cam) in enumerate(zip(images, cams)):
            shutil.copy2(p, out_dir / "frames" / f"{i:05d}{p.suffix.lower()}")
            pose = _pose(cam) if cam else None
            if pose is not None:
                rows.append(f"{i / fps:.4f},{pose[0]:.5f},{pose[1]:.5f},{pose[2]:.5f},0,0,0,1")
            if dep_dir is not None:
                d = dep_dir / f"{p.stem}.png"
                if d.exists():
                    (out_dir / "depth").mkdir(exist_ok=True)
                    shutil.copy2(d, out_dir / "depth" / f"{i:05d}.png")
                    n_depth += 1
        poses_ok = len(rows) == len(images) and len(rows) >= 2
        if poses_ok:
            (out_dir / "camera_poses.csv").write_text("timestamp,tx,ty,tz,qx,qy,qz,qw\n" + "\n".join(rows) + "\n")
        elif (out_dir / "camera_poses.csv").exists():
            (out_dir / "camera_poses.csv").unlink()
        return {"frames": len(images), "fps": round(fps, 3), "poses": poses_ok, "depth_frames": n_depth,
                "source": str(src)}
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
