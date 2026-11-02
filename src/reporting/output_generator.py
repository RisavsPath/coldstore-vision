"""Master orchestrator for the 5 deliverables."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

from src.reporting import dashboard_builder as db
from src.reporting import video_renderer as vr

log = logging.getLogger("coldstore")


def generate_outputs(insp, cfg: dict, out_dir: str | Path, skip_video: bool = False) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    vr.CODECS = tuple(cfg["outputs"].get("codecs", vr.CODECS))
    if not skip_video:
        log.info("output 1: anomaly overlay video")
        paths["anomaly_video"] = vr.render_overlay_video(insp, cfg, out / "anomaly_video.mp4")
        log.info("output 3: side-by-side video")
        paths["side_by_side"] = vr.render_side_by_side(insp, cfg, out / "side_by_side.mp4")
    log.info("output 2: audit report + manifest")
    paths["manifest"] = db.write_manifest(insp, out)
    paths["audit_report"] = db.write_audit_report(insp, out)
    log.info("output 4: digital twin")
    paths["digital_twin"] = db.write_digital_twin(insp, out)
    log.info("output 5: curtain slider")
    paths["panoramic_slider"] = db.write_panoramic_slider(insp, out)
    cv2.imwrite(str(out / "mosaic_day1.jpg"), insp.mosaic1.image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(out / "mosaic_day2.jpg"), insp.mosaic2.image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return paths
