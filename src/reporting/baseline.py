"""Day 1 baseline (no Day 2 yet): sack inventory + recording-quality check.

Outputs in <out>/:
  baseline.html            summary, recording checks, aisle mosaic with every sack, keyframe gallery
  baseline.json            counts + checks (machine-readable)
  mosaic_day1_sacks.jpg    aisle panorama with numbered sacks
  keyframes/kf_XXX.jpg     each keyframe with its YOLO masks
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.ingestion.keyframe_extractor import extract_keyframes, laplacian_var
from src.registration.mosaicking import build_mosaic, project_box
from src.reporting.dashboard_builder import BASE_CSS, b64img
from src.segmentation.sack_detector import SackDetector

log = logging.getLogger("coldstore")


def _iomin(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    amin = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / amin if amin > 0 else 0.0


def run_baseline(day_dir: Path, cfg: dict, out: Path, input_info: dict) -> dict:
    out = Path(out)
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    kfs, src = extract_keyframes(day_dir, cfg)
    det = SackDetector(cfg)
    log.info("baseline: %d keyframes, detector %s", len(kfs), det.source)

    mosaic = build_mosaic([k.image for k in kfs], cfg)
    boxes, per_kf, gallery = [], [], []
    for i, k in enumerate(kfs):
        inst = [d for d in det.detect(k.image) if d.cls == "sack"]
        per_kf.append(len(inst))
        vis = k.image.copy()
        ov = vis.copy()
        for d in inst:
            cv2.fillPoly(ov, [d.polygon.astype(np.int32).reshape(-1, 1, 2)], (80, 200, 80))
            h, w = k.image.shape[:2]
            b = d.box
            comp = min(b[0], b[1], w - b[2], h - b[3])
            boxes.append((project_box(b, mosaic.transforms[i]), comp, d.conf))
        vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
        for d in inst:
            cv2.polylines(vis, [d.polygon.astype(np.int32).reshape(-1, 1, 2)], True, (80, 200, 80), 2)
        cv2.putText(vis, f"keyframe {i}  t={k.t:.1f}s  sacks={len(inst)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        p = out / "keyframes" / f"kf_{i:03d}.jpg"
        cv2.imwrite(str(p), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        gallery.append(p)

    # unique sacks along the aisle: cluster keyframe detections in mosaic space, least-truncated first
    uniq: list = []
    for b, comp, conf in sorted(boxes, key=lambda t: -t[1]):
        if not any(_iomin(u, b) > 0.6 for u in uniq):
            uniq.append(b)
    m = mosaic.image.copy()
    for n, b in enumerate(sorted(uniq, key=lambda b: (b[0], b[1])), 1):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(m, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(m, str(n), (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "mosaic_day1_sacks.jpg"), m, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # recording checks (what matters for matching Day 2 to Day 1)
    sharp = [laplacian_var(k.image) for k in kfs]
    bright = [float(cv2.cvtColor(k.image, cv2.COLOR_BGR2GRAY).mean()) for k in kfs]
    duration = src.n / src.fps
    checks = {
        "duration_s": round(duration, 1),
        "keyframes": len(kfs),
        "sharp_keyframes_pct": round(100 * float(np.mean(np.array(sharp) >= cfg["ingestion"]["blur_var_threshold"])), 1),
        "mean_brightness": round(float(np.mean(bright)), 1),
        "brightness_drift": round(float(np.max(bright) - np.min(bright)), 1),
        "mosaic_px": list(mosaic.image.shape[1::-1]),
    }
    advice = []
    if duration < 20:
        advice.append(f"Video is {duration:.0f}s. 20-30s covering the whole rack length gives more overlap to match Day 2.")
    if checks["sharp_keyframes_pct"] < 70:
        advice.append("Many blurry frames: walk slower and hold the phone steadier.")
    if checks["brightness_drift"] > 40:
        advice.append("Brightness changes a lot between frames: long-press to set AE/AF LOCK before recording.")
    if checks["mean_brightness"] < 60:
        advice.append("Footage is dark: turn on more lights (and keep them the same on Day 2).")
    if not advice:
        advice.append("Recording looks good for Day 2 matching. Repeat the same route, height, speed and lighting.")

    report = {"input": input_info, "detector": det.source, "unique_sacks_in_aisle": len(uniq),
              "sacks_per_keyframe": per_kf, "checks": checks, "advice": advice}
    (out / "baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    thumbs = "".join(f'<figure><img src="{b64img(cv2.resize(cv2.imread(str(g)), (480, int(480 * kfs[0].image.shape[0] / kfs[0].image.shape[1]))), 78)}" '
                     f'alt="keyframe {i}"><figcaption>kf {i} · {per_kf[i]} sacks</figcaption></figure>'
                     for i, g in enumerate(gallery))
    mz = cv2.imread(str(out / "mosaic_day1_sacks.jpg"))
    s = min(1.0, 2400 / mz.shape[1])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Day 1 Baseline</title><style>{BASE_CSS}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.card .v{{font-size:26px;font-weight:700}}.mz{{overflow-x:auto;border:1px solid var(--line);border-radius:10px}}.mz img{{display:block;max-width:none}}
.gal{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}}figure{{margin:0}}figure img{{width:100%;border-radius:6px}}figcaption{{font-size:12px;color:var(--muted)}}
li{{margin:4px 0}}</style></head><body><header><h1>Day 1 Baseline</h1><div class="muted">{html.escape(str(input_info.get('source', '')))} · detector {html.escape(det.source)}</div></header><main>
<div class="cards"><div class="card"><div class="muted">Unique sacks in aisle</div><div class="v">{len(uniq)}</div></div>
<div class="card"><div class="muted">Duration</div><div class="v">{checks['duration_s']} s</div></div>
<div class="card"><div class="muted">Keyframes</div><div class="v">{checks['keyframes']}</div></div>
<div class="card"><div class="muted">Sharp keyframes</div><div class="v">{checks['sharp_keyframes_pct']}%</div></div>
<div class="card"><div class="muted">Brightness drift</div><div class="v">{checks['brightness_drift']}</div></div></div>
<h2>Recording advice for Day 2</h2><ul>{''.join(f'<li>{html.escape(a)}</li>' for a in advice)}</ul>
<h2>Aisle mosaic with detected sacks</h2><div class="mz"><img src="{b64img(cv2.resize(mz, None, fx=s, fy=s), 82)}" alt="Day 1 mosaic"></div>
<h2>Keyframes</h2><div class="gal">{thumbs}</div></main></body></html>"""
    (out / "baseline.html").write_text(page, encoding="utf-8")
    return report
