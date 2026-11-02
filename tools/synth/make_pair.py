"""Synthetic Day 1 / Day 2 aisle walkthroughs with known ground-truth anomalies.

Renders a 4-bay x 3-tier rack of textured sacks (jute / onion mesh / garlic net) on a long canvas,
then "walks" a virtual camera along it to produce frames, ARKit-style poses and LiDAR-style depth.
Day 2 has: a different camera path (offset, roll, scale), a global lighting/white-balance change,
one MISSING sack, two rotting sacks (CRITICAL), one slumped sack, and one NEW sack in a slot that
was empty on Day 1.

    python -m tools.synth.make_pair --out data/synth

Writes <out>/day_01 and <out>/day_02 (frames/, fps.txt, camera_poses.csv, depth/) + ground_truth.json.
This exercises the full pipeline before real footage exists; it is NOT training data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H = 4400, 900
PX_PER_M = 400.0
BAYS, TIERS, SLOTS = 4, 3, 3
UPRIGHT_W, BEAM_H = 26, 18
WALL = (35, 30, 28)
UPRIGHT = (170, 95, 30)
BEAM = (0, 140, 255)
MATERIALS = {"jute": (70, 120, 160), "onion": (75, 55, 165), "garlic": (200, 205, 212)}
ROT = (35, 60, 95)
VIEW_W = 1600                      # canvas pixels covered by one frame (-> 1280x720)
FRAME = (1280, 720)

DAY2_CHANGES = {
    "missing": [(2, 1, 2)],
    "rot": [(1, 2, 1), (4, 1, 3)],
    "slump": [(3, 2, 2)],
    "new": [(3, 3, 3)],             # empty on Day 1, filled on Day 2
}


def slot_box(bay: int, tier: int, slot: int) -> tuple[int, int, int, int]:
    bay_w = W / BAYS
    tier_h = (H - BEAM_H) / TIERS
    x0 = (bay - 1) * bay_w + UPRIGHT_W
    slot_w = (bay_w - UPRIGHT_W) / SLOTS
    y_top = H - BEAM_H - tier * tier_h + BEAM_H   # tier 1 = bottom
    x1 = int(x0 + (slot - 1) * slot_w + 12)
    x2 = int(x0 + slot * slot_w - 12)
    y2 = int(y_top + tier_h - BEAM_H - 6)
    y1 = int(y2 - (tier_h - BEAM_H - 46))
    return x1, y1, x2, y2


def sack_texture(w: int, h: int, material: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx / w) * 2 - 1, (yy / h) * 2 - 1
    mask = (np.abs(nx) ** 4 + np.abs(ny) ** 3.2) <= 1.0
    base = np.array(MATERIALS[material], np.float32)
    if material == "jute":
        pat = 16 * np.sin(xx * 2 * np.pi / 6) * np.sin(yy * 2 * np.pi / 6)
    elif material == "onion":
        pat = 28 * ((((xx + yy) % 14) < 3) | (((xx - yy) % 14) < 3))
    else:
        pat = 20 * ((((xx + yy) % 10) < 2) | (((xx - yy) % 10) < 2))
    noise = cv2.GaussianBlur(rng.normal(0, 14, (h, w)).astype(np.float32), (0, 0), 1.2)
    lumps = cv2.GaussianBlur(rng.normal(0, 40, (h, w)).astype(np.float32), (0, 0), 18)
    shade = 0.72 + 0.28 * (1 - np.clip(nx ** 2 + ny ** 2, 0, 1))
    tex = (base[None, None, :] + (pat + noise + lumps)[..., None]) * shade[..., None]
    return np.clip(tex, 0, 255).astype(np.uint8), mask


def add_rot(tex: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 999)
    h, w = mask.shape
    rot = np.zeros((h, w), np.float32)
    for _ in range(6):
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        cv2.ellipse(rot, (int(cx), int(cy)), (int(rng.uniform(0.1, 0.22) * w), int(rng.uniform(0.08, 0.18) * h)),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
    rot = cv2.GaussianBlur(rot, (0, 0), 4) > 0.5
    speck = rng.normal(0, 38, (h, w, 1))
    out = tex.astype(np.float32)
    rotted = np.clip(np.array(ROT, np.float32) + speck, 0, 255)
    out[rot & mask] = rotted[rot & mask]
    return out.astype(np.uint8)


def render(day: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = WALL
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, (H, W, 1)).astype(np.int16), 0, 255).astype(np.uint8)
    for _ in range(900):  # wall marks/bolts so the background has registrable features
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
        cv2.circle(img, (x, y), int(rng.uniform(2, 5)), (60, 55, 52), -1)
    depth = np.full((H, W), 2.2, np.float32)
    gt = []
    mats = list(MATERIALS)
    for bay in range(1, BAYS + 1):
        for tier in range(1, TIERS + 1):
            for slot in range(1, SLOTS + 1):
                key = (bay, tier, slot)
                sid = f"b{bay}t{tier}s{slot}"
                kind = "normal"
                if key in DAY2_CHANGES["new"]:
                    kind = "new"
                    if day == 1:
                        gt.append({"id": sid, "bay": bay, "tier": tier, "slot": slot, "kind": "new", "box": None})
                        continue
                if key in DAY2_CHANGES["missing"]:
                    kind = "missing"
                    if day == 2:
                        gt.append({"id": sid, "bay": bay, "tier": tier, "slot": slot, "kind": kind,
                                   "box": list(slot_box(*key))})
                        continue
                x1, y1, x2, y2 = slot_box(*key)
                s_seed = seed * 1000 + bay * 100 + tier * 10 + slot
                mat = mats[(bay + tier + slot) % 3]
                tex, m = sack_texture(x2 - x1, y2 - y1, mat, s_seed)
                if key in DAY2_CHANGES["rot"]:
                    kind = "rot"
                    if day == 2:
                        tex = add_rot(tex, m, s_seed)
                if key in DAY2_CHANGES["slump"]:
                    kind = "slump"
                    if day == 2:  # top drops 28 px, sack bulges sideways
                        nh, nw = (y2 - y1) - 28, (x2 - x1) + 10
                        tex = cv2.resize(tex, (nw, nh))
                        m = cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
                        x1, x2, y1 = x1 - 5, x2 + 5, y1 + 28
                roi = img[y1:y2, x1:x2]
                roi[m] = tex[m]
                depth[y1:y2, x1:x2][m] = 1.2
                gt.append({"id": sid, "bay": bay, "tier": tier, "slot": slot, "kind": kind, "material": mat,
                           "box": [x1, y1, x2, y2]})
    # rack structure drawn over the sacks' edges
    for b in range(BAYS + 1):
        x = int(b * W / BAYS)
        x = min(max(x, UPRIGHT_W // 2), W - UPRIGHT_W // 2)
        cv2.rectangle(img, (x - UPRIGHT_W // 2, 0), (x + UPRIGHT_W // 2, H), UPRIGHT, -1)
        for yy in range(20, H, 60):
            cv2.circle(img, (x, yy), 4, (90, 50, 20), -1)
        depth[:, x - UPRIGHT_W // 2: x + UPRIGHT_W // 2] = 1.1
    tier_h = (H - BEAM_H) / TIERS
    for t in range(TIERS + 1):
        y = int(H - BEAM_H - t * tier_h)
        cv2.rectangle(img, (0, y), (W, y + BEAM_H), BEAM, -1)
        depth[y:y + BEAM_H] = 1.1
    return img, depth, gt


def relight(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    f = np.power(f, 1.08) * 0.84
    f *= np.array([0.95, 1.0, 1.05], np.float32)  # warmer white balance (BGR)
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def walk(canvas: np.ndarray, depth: np.ndarray, out: Path, day: int, fps: int = 30) -> None:
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "fps.txt").write_text(str(fps))
    step = 40 if day == 1 else 37
    x0 = 0 if day == 1 else 30
    roll = 0.0 if day == 1 else 1.2
    scale = 1.0 if day == 1 else 1.03
    dy = 0 if day == 1 else 14
    s = FRAME[0] / VIEW_W
    rows = ["timestamp,tx,ty,tz,qx,qy,qz,qw"]
    i = 0
    x = x0
    while x + VIEW_W <= W:
        cx, cy = x + VIEW_W / 2, H / 2 + dy
        # canvas -> frame: translate centre to origin, roll/scale, scale to frame, re-centre
        R = cv2.getRotationMatrix2D((cx, cy), roll, scale * s)
        R[0, 2] += FRAME[0] / 2 - cx
        R[1, 2] += FRAME[1] / 2 - cy
        frame = cv2.warpAffine(canvas, R, FRAME, flags=cv2.INTER_LINEAR, borderValue=WALL)
        noise = np.random.default_rng(day * 10000 + i).normal(0, 2.5, frame.shape)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "frames" / f"{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        d = cv2.warpAffine(depth, R, FRAME, flags=cv2.INTER_NEAREST, borderValue=2.2)
        d = cv2.resize(d, (256, 144), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / "depth" / f"{i:05d}.png"), (d * 1000).astype(np.uint16))
        rows.append(f"{i / fps:.4f},{cx / PX_PER_M:.4f},{(dy) / PX_PER_M:.4f},0,0,0,0,1")
        i += 1
        x += step
    (out / "camera_poses.csv").write_text("\n".join(rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synth"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    c1, d1, gt1 = render(1, args.seed)
    c2, d2, gt2 = render(2, args.seed)
    c2 = relight(c2)
    walk(c1, d1, args.out / "day_01", 1)
    walk(c2, d2, args.out / "day_02", 2)
    cv2.imwrite(str(args.out / "canvas_day1.jpg"), c1)
    cv2.imwrite(str(args.out / "canvas_day2.jpg"), c2)
    expected = {f"b{b}t{t}s{s}": k for k, lst in DAY2_CHANGES.items() for (b, t, s) in lst}
    (args.out / "ground_truth.json").write_text(json.dumps(
        {"bays": BAYS, "tiers": TIERS, "slots_per_tier": SLOTS, "changes": DAY2_CHANGES,
         "expected": expected, "day1": gt1, "day2": gt2}, indent=1))
    print(f"synthetic pair -> {args.out}")


if __name__ == "__main__":
    main()
