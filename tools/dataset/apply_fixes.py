"""Stage 1b: apply reviewer-agent fix instructions with SAM 2.1 point prompts.

Reads data/scraped/reviews/fix/*.json ({"delete", "split", "add", "drop_image"} per image):
  - removes deleted and split masks (1-based numbers as drawn by render_overlays.draw)
  - converts each grid point to pixels and asks SAM 2.1 for one mask per point
  - keeps new masks with a sack-like area/solidity that don't duplicate an existing mask
Updates data/scraped/autolabels/<id>.txt and marks meta agent_fixed=true (originals backed up as
<id>.txt.bak). Images with drop_image=true are marked agent_dropped and excluded from packaging.

Usage: python -m tools.dataset.apply_fixes [--ids ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from tools.dataset.autolabel import solidity, to_polygon
from tools.dataset.common import ACCEPTED_DIR, LABELS_DIR, REVIEW_DIR, ROOT
from tools.dataset.render_fix_sheets import cell_to_xy

MIN_AREA, MAX_AREA, MIN_SOLIDITY, DUP_IOU = 0.0005, 0.35, 0.70, 0.5


def poly_mask(line: str, w: int, h: int) -> np.ndarray:
    pts = (np.array(line.split()[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def load_fixes(fix_dir: str) -> dict[str, dict]:
    out = {}
    for p in sorted((REVIEW_DIR / fix_dir).glob("*.json")):
        out.update(json.loads(p.read_text(encoding="utf-8")).get("results", {}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fix-dir", default="fix", help="review round folder, e.g. fix2 for a second pass")
    args = ap.parse_args()
    from ultralytics import SAM

    with contextlib.chdir(ROOT / "models" / "hub"):
        sam = SAM("sam2.1_b.pt")
    fixes = load_fixes(args.fix_dir)
    if args.ids:
        fixes = {k: v for k, v in fixes.items() if k in set(args.ids)}
    stats = {"images": 0, "dropped": 0, "deleted": 0, "added": 0, "rejected_points": 0}
    for img_id, fx in tqdm(fixes.items(), desc="apply fixes"):
        lbl = LABELS_DIR / f"{img_id}.txt"
        meta_p = LABELS_DIR / f"{img_id}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        done = meta.get(f"applied_{args.fix_dir}") or (args.fix_dir == "fix" and meta.get("agent_fixed"))
        if meta.get("human_verified") or done:
            continue
        if fx.get("drop_image"):
            meta["agent_dropped"] = True
            stats["dropped"] += 1
            if not args.dry_run:
                meta_p.write_text(json.dumps(meta))
            continue
        img = cv2.imread(str(ACCEPTED_DIR / f"{img_id}.jpg"))
        h, w = img.shape[:2]
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
        remove = {int(k) for k in fx.get("delete", []) + fx.get("split", [])}
        keep = [ln for k, ln in enumerate(lines, 1) if k not in remove]
        stats["deleted"] += len(lines) - len(keep)
        existing = [poly_mask(ln, w, h) for ln in keep if ln.startswith("0 ")]

        pts = []
        for p in fx.get("add", []):
            try:
                pts.append(cell_to_xy(p, w, h))
            except (ValueError, IndexError):
                stats["rejected_points"] += 1
        new_lines = []
        if pts:
            res = sam.predict(img, points=[[float(x), float(y)] for x, y in pts], labels=[1] * len(pts),
                              verbose=False)[0]
            masks = [] if res.masks is None else list(res.masks.data.cpu().numpy() > 0.5)
            for m in masks:
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                a = m.sum() / (h * w)
                if not MIN_AREA <= a <= MAX_AREA or solidity(m) < MIN_SOLIDITY:
                    stats["rejected_points"] += 1
                    continue
                if any((m & e).sum() / max((m | e).sum(), 1) > DUP_IOU for e in existing):
                    stats["rejected_points"] += 1
                    continue
                poly = to_polygon(m)
                if poly is None:
                    continue
                existing.append(m)
                new_lines.append("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly))
        stats["added"] += len(new_lines)
        stats["images"] += 1
        if args.dry_run:
            continue
        shutil.copy2(lbl, lbl.with_name(f"{lbl.stem}.{args.fix_dir}.bak"))
        lbl.write_text("\n".join(keep + new_lines) + "\n")
        meta.update(agent_fixed=True, **{f"applied_{args.fix_dir}": True},
                    fix_deleted=len(lines) - len(keep), fix_added=len(new_lines))
        meta_p.write_text(json.dumps(meta))
    print(stats)


if __name__ == "__main__":
    main()
