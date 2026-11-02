"""Polycam raw-export conversion: fake export built from the synthetic walk -> day folder -> pipeline."""
import json
import zipfile

from src.ingestion.keyframe_extractor import extract_keyframes
from src.ingestion.polycam_converter import convert, is_polycam_export


def make_fake_polycam(day_dir, out_zip):
    """Mimic Polycam's keyframes/{images,cameras,depth} layout (ns timestamps, 3x4 t_ij pose)."""
    poses = [ln.split(",") for ln in (day_dir / "camera_poses.csv").read_text().splitlines()[1:]]
    with zipfile.ZipFile(out_zip, "w") as z:
        for i, f in enumerate(sorted((day_dir / "frames").glob("*.jpg"))):
            stem = str(1000 + i)
            z.write(f, f"capture/keyframes/images/{stem}.jpg")
            _, tx, ty, tz, *_ = (float(v) for v in poses[i])
            cam = {"timestamp": int(i / 30 * 1e9), "fx": 900, "fy": 900, "cx": 640, "cy": 360,
                   "t_00": 1, "t_01": 0, "t_02": 0, "t_03": tx, "t_10": 0, "t_11": 1, "t_12": 0, "t_13": ty,
                   "t_20": 0, "t_21": 0, "t_22": 1, "t_23": tz}
            z.writestr(f"capture/keyframes/cameras/{stem}.json", json.dumps(cam))
            z.write(day_dir / "depth" / f"{f.stem}.png", f"capture/keyframes/depth/{stem}.png")


def test_polycam_roundtrip(synth_pair, cfg, tmp_path):
    out, _ = synth_pair
    z = tmp_path / "polycam_day1.zip"
    make_fake_polycam(out / "day_01", z)
    assert is_polycam_export(z)
    info = convert(z, tmp_path / "day_01")
    n = len(list((out / "day_01" / "frames").glob("*.jpg")))
    assert info["frames"] == n and info["poses"] and info["depth_frames"] == n
    assert abs(info["fps"] - 30) < 1
    kfs, _ = extract_keyframes(tmp_path / "day_01", cfg)
    assert len(kfs) >= 8 and all(k.pose is not None and k.depth is not None for k in kfs)
