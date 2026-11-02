"""Narration for the explainer: synthesise each line, place it on the timeline, mix, mux.

    python video/voiceover.py                 # needs video/renders/coldstore_vision_explainer.mp4
    python video/voiceover.py --voice en-IN-PrabhatNeural

Each line is anchored to a segment (start = sum of the real durations of earlier segments, read
with ffprobe) plus an offset, and must fit inside max_s. A line that runs long is re-synthesised
faster (up to +30%). Uses edge-tts (Microsoft neural voices; sends only the narration text).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

import edge_tts

from render_html import ffmpeg_bin

ROOT = Path(__file__).resolve().parent
R = ROOT / "renders"
VO = R / "vo"
SEGMENTS = ["s0_intro", "f0_drone", "s1_title", "s2_setup", "f1_days", "s3_pipeline",
            "f2_side_by_side", "f3_anomaly", "s4_results", "s5_next"]

# (segment, offset s, max s, text)
LINES = [
    ("s0_intro", 0.4, 7.4, "ColdStore Vision: automated cold storage inspection, where every sack is compared with its last scan."),
    ("f0_drone", 0.5, 4.8, "An inspection drone waits in its charging dock."),
    ("f0_drone", 5.6, 9.4, "It carries a 4K camera, LiDAR depth sensing, visual-inertial tracking, and an edge AI computer."),
    ("f0_drone", 15.2, 4.3, "At each bay, it reads the rack's QR tag."),
    ("f0_drone", 19.6, 4.1, "Then it sweeps the rack, checking every sack."),
    ("f0_drone", 23.8, 6.0, "Against the last scan: spoilage in red, a missing sack in cyan, the rest normal."),
    ("s1_title", 0.3, 7.8, "That is the vision. Today, we have a working prototype, tested in a real godown."),
    ("s2_setup", 0.5, 11.6, "We recorded two handheld walkthroughs of the same onion and potato stacks, on two different days, "
                           "with different angles and lighting. A deliberately hard test."),
    ("f1_days", 0.4, 6.9, "This is the raw input. Day one on the left, day two on the right."),
    ("s3_pipeline", 0.6, 19.2, "The software picks sharp keyframes, aligns the two days across new viewpoints and light, "
                              "outlines every sack with our trained YOLO model, matches each sack to itself across days, "
                              "then checks for missing stock, spoilage, texture change and settling, and writes the reports."),
    ("f2_side_by_side", 0.4, 11.3, "The side by side output shows day one against the aligned day two, "
                                  "with the reason for every flag written underneath."),
    ("f3_anomaly", 0.3, 6.8, "And the day two replay marks every sack live, by status."),
    ("s4_results", 0.5, 24.2, "The detector was trained on nearly twenty eight hundred sack outlines, labelled by AI agents "
                             "and corrected with SAM 2, reaching eighty percent mask precision. On real footage it separated "
                             "forty four touching sacks in one frame. Between the two days, it caught the front sack that "
                             "was swapped, and it refused to call sacks hidden behind a pillar missing."),
    ("s5_next", 0.5, 21.2, "Thirteen of the twenty two onion sacks stayed unverified, because the footage was handheld "
                          "from different angles. A fixed drone route, rack tags and LiDAR remove exactly that limit. "
                          "The software is ready. Consistent capture unlocks full rack verdicts."),
]


def ffprobe_duration(path: Path) -> float:
    probe = Path(ffmpeg_bin()).with_name("ffprobe.exe")
    out = subprocess.run([str(probe), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


async def synth(text: str, voice: str, rate: int, out: Path) -> None:
    await edge_tts.Communicate(text, voice, rate=f"{rate:+d}%").save(str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-IN-NeerjaNeural")
    ap.add_argument("--video", type=Path, default=R / "coldstore_vision_explainer.mp4")
    args = ap.parse_args()
    VO.mkdir(parents=True, exist_ok=True)

    starts, t = {}, 0.0
    for seg in SEGMENTS:
        starts[seg] = t
        t += ffprobe_duration(R / f"{seg}.mp4" if seg != "f0_drone" else R / "f0_drone.mp4")
    total = ffprobe_duration(args.video)
    print(f"timeline {t:.1f}s, video {total:.1f}s")

    clips = []
    for i, (seg, off, max_s, text) in enumerate(LINES):
        out = VO / f"line_{i:02d}.mp3"
        rate = 0
        while True:
            asyncio.run(synth(text, args.voice, rate, out))
            d = ffprobe_duration(out)
            if d <= max_s or rate >= 30:
                break
            rate = min(30, rate + max(5, int((d / max_s - 1) * 100) + 2))
        start = starts[seg] + off
        flag = "  (TOO LONG)" if d > max_s else ""
        print(f"{i:02d} {seg:16s} @{start:6.1f}s  {d:4.1f}s / {max_s}s  rate {rate:+d}%{flag}")
        clips.append((out, start))

    inputs, chains = [], []
    for k, (p, start) in enumerate(clips):
        inputs += ["-i", str(p)]
        ms = int(start * 1000)
        chains.append(f"[{k + 1}:a]aresample=48000,adelay={ms}|{ms}[a{k}]")
    mix = "".join(f"[a{k}]" for k in range(len(clips)))
    fc = ";".join(chains) + f";{mix}amix=inputs={len(clips)}:normalize=0,loudnorm=I=-16:TP=-1.5,apad[aout]"
    final = args.video.with_name(args.video.stem + "_voiceover.mp4")
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(args.video), *inputs, "-filter_complex", fc,
                    "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    str(final)], check=True)
    print(f"-> {final}")


if __name__ == "__main__":
    main()
