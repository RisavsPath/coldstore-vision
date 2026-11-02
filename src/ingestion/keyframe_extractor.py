"""Module 1: keyframe sampling by camera travel distance with Laplacian blur rejection.

A day directory contains:
    rgb_video.mp4 (or any .mp4/.mov)  OR  frames/*.jpg|png (image sequence)
    camera_poses.csv   (optional) timestamp,tx,ty,tz,qx,qy,qz,qw   (ARKit, metres)
    depth/             (optional) one depth image per frame: 16-bit PNG in mm, .npy or .exr in metres
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi")
IMG_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class Keyframe:
    index: int                 # frame index in the source video
    t: float                   # seconds from start
    image: np.ndarray          # BGR, processing resolution
    pose: np.ndarray | None    # (3,) camera position in metres
    depth: np.ndarray | None   # float32 metres, same HxW as image
    sharpness: float


class FrameSource:
    """Uniform random access to a video file or an image-sequence folder, resized to proc_width."""

    def __init__(self, day_dir: Path, proc_width: int):
        self.day_dir = Path(day_dir)
        self.proc_width = proc_width
        frames_dir = self.day_dir / "frames"
        self.files: list[Path] = []
        self.video: Path | None = None
        if frames_dir.exists():
            self.files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            self.fps = float((self.day_dir / "fps.txt").read_text()) if (self.day_dir / "fps.txt").exists() else 30.0
            self.n = len(self.files)
        else:
            vids = sorted(p for p in self.day_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
            if not vids:
                raise FileNotFoundError(f"no video or frames/ in {self.day_dir}")
            preferred = [v for v in vids if v.stem == "rgb_video"]
            self.video = (preferred or vids)[0]
            cap = cv2.VideoCapture(str(self.video))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        if self.n == 0:
            raise ValueError(f"{self.day_dir} has no frames")

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w == self.proc_width:
            return img
        s = self.proc_width / w
        return cv2.resize(img, (self.proc_width, round(h * s)), interpolation=cv2.INTER_AREA)

    def iter_frames(self, stride: int = 1):
        if self.files:
            for i in range(0, self.n, stride):
                yield i, self._resize(cv2.imread(str(self.files[i])))
            return
        cap = cv2.VideoCapture(str(self.video))
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % stride == 0:
                ok, f = cap.retrieve()
                if ok:
                    yield i, self._resize(f)
            i += 1
        cap.release()

    def get(self, indices: list[int]) -> dict[int, np.ndarray]:
        want = set(indices)
        if self.files:
            return {i: self._resize(cv2.imread(str(self.files[i]))) for i in want}
        out = {}
        for i, f in self.iter_frames():
            if i in want:
                out[i] = f
                if len(out) == len(want):
                    break
        return out


def load_poses(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not csv_path.exists():
        return None
    ts, xyz = [], []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            ts.append(float(r["timestamp"]))
            xyz.append([float(r["tx"]), float(r["ty"]), float(r["tz"])])
    ts_a = np.array(ts)
    return ts_a - ts_a[0], np.array(xyz)


def laplacian_var(img_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def load_depth(day_dir: Path, frame_index: int, n_frames: int, shape: tuple[int, int]) -> np.ndarray | None:
    ddir = Path(day_dir) / "depth"
    if not ddir.exists():
        return None
    files = sorted(p for p in ddir.iterdir() if p.suffix.lower() in (".png", ".npy", ".exr", ".tif", ".tiff"))
    if not files:
        return None
    # depth streams are often at a lower rate than RGB: map proportionally
    k = min(len(files) - 1, round(frame_index * (len(files) - 1) / max(n_frames - 1, 1)))
    p = files[k]
    if p.suffix == ".npy":
        d = np.load(p).astype(np.float32)
    else:
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if d is None:
            return None
        d = d.astype(np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        if p.suffix.lower() in (".png", ".tif", ".tiff") and d.max() > 50:
            d /= 1000.0  # millimetres -> metres
    return cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def extract_keyframes(day_dir: str | Path, cfg: dict) -> tuple[list[Keyframe], FrameSource]:
    day_dir = Path(day_dir)
    icfg = cfg["ingestion"]
    src = FrameSource(day_dir, cfg.get("processing", {}).get("width", 1280))
    poses = load_poses(day_dir / "camera_poses.csv")

    # pass 1: sharpness of every frame (cheap at half processing resolution)
    sharp = np.zeros(src.n)
    for i, f in src.iter_frames():
        if i < src.n:
            sharp[i] = laplacian_var(cv2.resize(f, None, fx=0.5, fy=0.5))
    n = src.n

    # candidate keyframes
    if poses is not None:
        ts, xyz = poses
        frame_t = np.arange(n) / src.fps
        pos = np.stack([np.interp(frame_t, ts, xyz[:, k]) for k in range(3)], axis=1)
        cands, last = [0], pos[0]
        for i in range(1, n):
            if np.linalg.norm(pos[i] - last) >= icfg["keyframe_min_dist_m"]:
                cands.append(i)
                last = pos[i]
    else:
        pos = None
        step = max(1, round(icfg["fallback_keyframe_every_s"] * src.fps))
        cands = list(range(0, n, step))

    # blur rejection: below threshold, take the sharpest frame within ±window
    thr, win = icfg["blur_var_threshold"], icfg["blur_search_window"]
    chosen = []
    for c in cands:
        if sharp[c] < thr:
            lo, hi = max(0, c - win), min(n, c + win + 1)
            c = lo + int(np.argmax(sharp[lo:hi]))
        if not chosen or c != chosen[-1]:
            chosen.append(c)

    frames = src.get(chosen)
    kfs = []
    for c in chosen:
        img = frames[c]
        kfs.append(Keyframe(
            index=c, t=c / src.fps, image=img, pose=None if pos is None else pos[c],
            depth=load_depth(day_dir, c, n, img.shape[:2]), sharpness=float(sharp[c]),
        ))
    return kfs, src
