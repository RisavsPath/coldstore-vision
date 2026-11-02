"""Render HTML/CSS/SVG scenes to MP4, frame-exact.

Every CSS animation is paused and its clock set to t = frame / fps before each screenshot, so the
output is deterministic (no dropped frames, no timing jitter) regardless of machine speed.
Each scene's length comes from <body data-duration="seconds">.

    python video/render_html.py                      # all scenes in video/web/s*.html
    python video/render_html.py video/web/s4_results.html --fps 30
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "renders"


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    hits = glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\*\bin\ffmpeg.exe"))
    if hits:
        return hits[0]
    raise SystemExit("ffmpeg not found - install it (winget install Gyan.FFmpeg --source winget)")


def render(page, html: Path, fps: int) -> Path:
    page.goto(html.resolve().as_uri())
    page.wait_for_load_state("networkidle")
    page.evaluate("document.fonts.ready")
    duration = float(page.evaluate("document.body.dataset.duration || 8"))
    n = int(round(duration * fps))
    out = OUT / f"{html.stem}.mp4"
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", str(out)],
        stdin=subprocess.PIPE)
    page.evaluate("document.getAnimations().forEach(a => a.pause())")
    for i in range(n):
        t = i * 1000.0 / fps
        page.evaluate(f"document.getAnimations().forEach(a => {{ a.currentTime = {t}; }})")
        proc.stdin.write(page.screenshot(type="png"))
    proc.stdin.close()
    proc.wait()
    print(f"{html.name}: {n} frames ({duration}s) -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    scenes = args.scenes or sorted((ROOT / "web").glob("s*.html"))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        for s in scenes:
            render(page, s, args.fps)
        browser.close()


if __name__ == "__main__":
    main()
