"""Outputs 1 and 3: temporal anomaly overlay video and synchronized split-screen reasoning video."""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import cv2
import numpy as np

from src.ingestion.keyframe_extractor import FrameSource

log = logging.getLogger("coldstore")

STATUS_BGR = {
    "NORMAL": (80, 200, 80), "WARNING": (0, 210, 255), "CRITICAL": (50, 50, 235),
    "MISSING": (255, 230, 0), "NEW": (230, 80, 230), "UNVERIFIED": (160, 160, 160),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


CODECS = ("mp4v",)  # set via outputs.codecs; add "avc1" first if an H.264 encoder (openh264) is installed


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in CODECS:
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError(f"cannot open video writer for {path}")


def dashed_rect(img, p1, p2, color, thickness=2, dash=12):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for s in range(0, length, 2 * dash):
            t0, t1 = s / max(length, 1), min(s + dash, length) / max(length, 1)
            q0 = (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0))
            q1 = (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1))
            cv2.line(img, q0, q1, color, thickness, cv2.LINE_AA)


def draw_label(img, text, org, color, scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), int(max(th + 4, org[1]))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_item(img, poly, box, status, label, thick=2, fill_alpha=0.25):
    color = STATUS_BGR.get(status, (200, 200, 200))
    if status == "MISSING" or poly is None or len(poly) < 3:
        dashed_rect(img, box[:2], box[2:], color, thick)
    else:
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        if fill_alpha > 0:
            ov = img.copy()
            cv2.fillPoly(ov, [pts], color)
            cv2.addWeighted(ov, fill_alpha, img, 1 - fill_alpha, 0, img)
        cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
    draw_label(img, label, (box[0], box[1] - 2), color)


def _transform_pts(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)


def _box_pts(box):
    x1, y1, x2, y2 = box
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def _pair_items(insp, by_gid) -> dict[int, list]:
    """Per pair: [(poly_in_registered_coords, box_registered, status, label)], one per global sack."""
    items: dict[int, list] = {}
    for o in insp.observations:
        if o.pair is None or o.global_id not in by_gid:
            continue
        rec = by_gid[o.global_id]
        if rec.status == "MISSING":
            poly, box = None, o.inst.box
        elif o.match is not None:
            poly, box = o.match.polygon, o.match.box
        else:
            poly, box = o.inst.polygon, o.inst.box
        items.setdefault(o.pair, []).append((poly, box, rec.status, f"{rec.sack_id} {rec.status}", o.global_id,
                                             o.completeness))
    return items


def _frame_to_kf(frame, kf_img, cfg) -> np.ndarray | None:
    """Homography frame -> keyframe. SIFT + Lowe ratio (not ORB cross-check): stacked sacks are a
    repetitive texture, and only the ratio test rejects matches to the wrong, identical-looking sack."""
    from src.registration.homography_warper import register

    s = 0.5
    r = register(cv2.resize(kf_img, None, fx=s, fy=s), cv2.resize(frame, None, fx=s, fy=s), cfg)
    if not r.ok:
        return None
    S = np.diag([s, s, 1.0])
    return np.linalg.inv(S) @ r.H @ S


def render_overlay_video(insp, cfg: dict, out_path: Path) -> Path:
    """Output 1: full Day 2 replay with live status masks (statuses projected from the analysed pairs)."""
    by_gid = {s.obs.global_id: s for s in insp.sacks}
    items = _pair_items(insp, by_gid)
    pairs_by_j: dict[int, list[int]] = {}
    for p_idx, pr in enumerate(insp.pairs):
        pairs_by_j.setdefault(pr.j2, []).append(p_idx)
    src = FrameSource(insp.src2_dir, cfg["processing"]["width"])
    stride = max(1, round(src.fps / cfg["outputs"]["video_fps"]))
    kf_idx = [(insp.kf2[j].index, j) for j in pairs_by_j]
    writer = None
    counts = insp.kpis.get("status_counts", {})
    hud = "  ".join(f"{k}:{v}" for k, v in counts.items())
    for f_idx, frame in src.iter_frames(stride):
        if writer is None:
            writer = _writer(out_path, src.fps / stride, (frame.shape[1], frame.shape[0]))
        out = frame.copy()
        # the two nearest analysed keyframes; per sack, draw the least-truncated observation
        best: dict[int, tuple] = {}
        for src_index, j in sorted(kf_idx, key=lambda t: abs(t[0] - f_idx))[:2]:
            G = np.eye(3) if src_index == f_idx else _frame_to_kf(frame, insp.kf2[j].image, cfg)
            if G is None:
                continue
            for p_idx in pairs_by_j[j]:
                M = np.linalg.inv(G) @ np.linalg.inv(insp.pairs[p_idx].reg.H)  # registered -> frame
                for poly, box, status, label, gid, comp in items.get(p_idx, []):
                    if gid not in best or comp > best[gid][0]:
                        best[gid] = (comp, poly, box, status, label, M)
        for comp, poly, box, status, label, M in best.values():
            bp = _transform_pts(_box_pts(box), M)
            fb = np.array([*bp.min(0), *bp.max(0)])
            if fb[2] < 0 or fb[0] > out.shape[1] or fb[3] < 0 or fb[1] > out.shape[0]:
                continue
            fp = _transform_pts(poly, M) if poly is not None and len(poly) >= 3 else None
            draw_item(out, fp, fb, status, label)
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(out, f"ColdStore-Vision | Day 2 replay | frame {f_idx}   {hud}", (10, 21), FONT, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(out)
    if writer is not None:
        writer.release()
    return out_path


def _banner(width: int, lines: list[str], height: int = 150) -> np.ndarray:
    b = np.full((height, width, 3), 24, np.uint8)
    y = 34
    for ln in lines:
        for sub in textwrap.wrap(ln, width=int(width / 11.5)) or [""]:
            cv2.putText(b, sub, (18, y), FONT, 0.68, (235, 235, 235), 1, cv2.LINE_AA)
            y += 28
            if y > height - 8:
                return b
    return b


def render_side_by_side(insp, cfg: dict, out_path: Path) -> Path:
    """Output 3: Day 1 keyframe | registered Day 2 keyframe + lower-third reasoning per flagged sack."""
    width = cfg["outputs"]["split_screen_width"]
    fps = cfg["outputs"].get("side_by_side_fps", 5)  # mostly still frames: low fps keeps the file small
    pane_w = width // 2
    by_gid = {s.obs.global_id: s for s in insp.sacks}
    items = _pair_items(insp, by_gid)
    writer = None

    def pane(img):
        s = pane_w / img.shape[1]
        return cv2.resize(img, (pane_w, round(img.shape[0] * s))), s

    for p_idx, pr in enumerate(insp.pairs):
        flagged = [s for s in insp.sacks if s.obs.pair == p_idx and s.status not in ("NORMAL", "UNVERIFIED")]
        segments = [(None, 1.0)] + [(s, 2.5) for s in flagged]
        for focus, dur in segments:
            left = pr.img1.copy()
            right = pr.img2n.copy()
            for s1 in pr.sacks1:
                cv2.polylines(left, [s1.polygon.astype(np.int32).reshape(-1, 1, 2)], True, (230, 230, 230), 1)
            for poly, box, status, label, gid, _ in items.get(p_idx, []):
                hl = focus is not None and focus.obs.global_id == gid
                draw_item(right, poly, box, status, label, thick=4 if hl else 2, fill_alpha=0.35 if hl else 0.15)
                if hl:
                    b1 = focus.obs.inst.box
                    cv2.rectangle(left, tuple(map(int, b1[:2])), tuple(map(int, b1[2:])), (0, 255, 255), 3)
            lp, _ = pane(left)
            rp, _ = pane(right)
            for img, t in ((lp, "DAY 1 BASELINE"), (rp, "DAY 2 REGISTERED")):
                cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
                cv2.putText(img, f"{t}  keyframe {pr.i1 if t.startswith('DAY 1') else pr.j2}", (10, 21), FONT, 0.6,
                            (240, 240, 240), 1, cv2.LINE_AA)
            top = np.hstack([lp, rp])
            if focus is None:
                n = len([o for o in insp.observations if o.pair == p_idx])
                lines = [f"Keyframe pair {p_idx + 1}/{len(insp.pairs)}: {n} sacks analysed, "
                         f"{len(flagged)} flagged. Registration inliers: {pr.reg.inliers}. "
                         f"Facility hue baseline shift {pr.global_hue:+.1f} deg."]
            else:
                lines = [focus.reasoning, f"Action: {focus.action}"]
            frame = np.vstack([top, _banner(top.shape[1], lines)])
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                writer = _writer(out_path, fps, size)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            for _ in range(int(dur * fps)):
                writer.write(frame)
    if writer is not None:
        writer.release()
    return out_path
