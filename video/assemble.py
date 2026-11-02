"""Assemble the prototype-phase segment: rendered HTML scenes + real footage + pipeline outputs.

    python video/render_html.py        # renders video/renders/s*.mp4 first
    python video/assemble.py           # -> video/renders/prototype_phase.mp4

Footage clips get a caption bar (drawtext) and are normalised to 1920x1080 @30 fps so the
final concat is a straight stream copy.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from render_html import ffmpeg_bin

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
R = ROOT / "renders"
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
FOOT1 = REPO / "footage/day1/IMG_3866.MOV"
FOOT2 = REPO / "footage/day2/IMG_3939.MOV"
RUN = REPO / "outputs/onion_day1_vs_day2"


def run(args: list[str]) -> None:
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", *args], check=True)


def caption(text: str, sub: str = "") -> str:
    t = text.replace(":", "\\:").replace("'", "’")
    s = sub.replace(":", "\\:").replace("'", "’")
    f = (f"drawbox=x=0:y=ih-150:w=iw:h=150:color=0x0b1220@0.82:t=fill,"
         f"drawtext=fontfile='{FONT}':text='{t}':fontcolor=white:fontsize=46:x=80:y=h-122")
    if s:
        f += f",drawtext=fontfile='{FONT}':text='{s}':fontcolor=0x93a1b8:fontsize=30:x=80:y=h-62"
    return f


NORM = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0b1220,fps=30,format=yuv420p"


def clip_days() -> Path:
    """Raw Day 1 | Day 2 walks side by side at 2x speed."""
    out = R / "f1_days.mp4"
    half = "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:color=0x0b1220"
    lab = lambda txt, col: (f"drawtext=fontfile='{FONT}':text='{txt}':fontcolor=white:fontsize=34:"  # noqa: E731
                            f"box=1:boxcolor={col}@0.85:boxborderw=14:x=24:y=24")
    fc = (f"[0:v]trim=0:15,setpts=0.5*PTS,{half},{lab('DAY 1  ·  21 Oct', '0x4fa3ff')}[a];"
          f"[1:v]trim=0:15,setpts=0.5*PTS,{half},{lab('DAY 2  ·  22 Oct', '0x14b8c8')}[b];"
          f"[a][b]hstack,pad=1920:1080:0:200:color=0x0b1220,"
          f"drawtext=fontfile='{FONT}':text='The raw input  -  two handheld walks (2x speed)':fontcolor=white:fontsize=52:x=80:y=90,"
          f"{caption('Different day, angle, resolution and lighting', 'Potato stack first, then the onion stack')},fps=30,format=yuv420p[v]")
    run(["-i", str(FOOT1), "-i", str(FOOT2), "-filter_complex", fc, "-map", "[v]", "-an", "-t", "7.5",
         "-c:v", "libx264", "-crf", "18", str(out)])
    return out


def clip_output(src: Path, name: str, start: float, dur: float, title: str, sub: str) -> Path:
    out = R / f"{name}.mp4"
    run(["-ss", str(start), "-t", str(dur), "-i", str(src), "-vf", f"{NORM},{caption(title, sub)}", "-an",
         "-c:v", "libx264", "-crf", "18", str(out)])
    return out


def clip_drone() -> Path | None:
    """Blender frames (video/blender/frames/drone_####.png) -> clip with a 'concept' tag."""
    frames = ROOT / "blender" / "frames"
    if not any(frames.glob("drone_*.png")):
        return None
    out = R / "f0_drone.mp4"
    tag = (f"drawtext=fontfile='{FONT}':text='CONCEPT VISUALISATION  ·  inspection drone':fontcolor=white:"
           f"fontsize=28:box=1:boxcolor=0x0b1220@0.7:boxborderw=12:x=40:y=h-80")
    run(["-framerate", "30", "-start_number", "1", "-i", str(frames / "drone_%04d.png"), "-vf", f"{tag},format=yuv420p",
         "-c:v", "libx264", "-crf", "18", str(out)])
    return out


def concat(parts: list[Path], final: Path) -> None:
    lst = R / f"{final.stem}_concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         "-r", "30", str(final)])
    print(f"-> {final}")


def main() -> None:
    R.mkdir(exist_ok=True)
    f1 = clip_days()
    f2 = clip_output(RUN / "side_by_side.mp4", "f2_side_by_side", 0, 12,
                     "Day 1 vs registered Day 2 - with the reason for every flag",
                     "Output 3: synchronized split-screen, generated automatically")
    f3 = clip_output(RUN / "anomaly_video.mp4", "f3_anomaly", 0, 7.2,
                     "Day 2 replay with live per-sack status",
                     "Output 1: green normal - yellow warning - cyan missing - grey unverified")
    order = [R / "s1_title.mp4", R / "s2_setup.mp4", f1, R / "s3_pipeline.mp4", f2, f3,
             R / "s4_results.mp4", R / "s5_next.mp4"]
    missing = [p for p in order if not p.exists()]
    if missing:
        raise SystemExit(f"render the HTML scenes first: {missing}")
    # re-encode on concat: inputs share size/fps/pix_fmt but come from different encoders
    concat(order, R / "prototype_phase.mp4")
    drone = clip_drone()
    if drone and (R / "s0_intro.mp4").exists():
        concat([R / "s0_intro.mp4", drone, R / "prototype_phase.mp4"], R / "coldstore_vision_explainer.mp4")
    else:
        print("drone frames or intro not rendered yet - full explainer skipped")


if __name__ == "__main__":
    main()
