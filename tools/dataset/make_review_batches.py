"""Step 3a: build contact sheets + batch files for the sack-image-reviewer agent.

Each sheet is a 3x3 grid of numbered tiles (one Read call shows the agent 9 images).
Each batch = up to --sheets-per-batch sheets and is described by a JSON file:

  data/scraped/review_sheets/batch_003.json
  {"batch": "batch_003", "mode": "images", "sheets": [{"sheet": "...png", "tiles": {"1": "<id>", ...}}],
   "images": {"<id>": "data/scraped/candidates/<id>.jpg"}, "output": "data/scraped/reviews/batch_003.json"}

Mode `labels` builds sheets from data/scraped/overlays/ instead (label QA pass).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.dataset.common import (ACCEPTED_CSV, CANDIDATES_CSV, OVERLAYS_DIR, REVIEW_DIR, ROOT, SHEETS_DIR,
                                  read_csv)

TILE = 520
GRID = 3


def font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_sheet(items: list[tuple[str, str]], out_path) -> dict:
    sheet = Image.new("RGB", (TILE * GRID, TILE * GRID), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    f = font(48)
    tiles = {}
    for i, (img_id, rel) in enumerate(items):
        im = Image.open(ROOT / rel).convert("RGB")
        im.thumbnail((TILE - 8, TILE - 8))
        x, y = (i % GRID) * TILE, (i // GRID) * TILE
        sheet.paste(im, (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2))
        n = str(i + 1)
        draw.rectangle([x + 4, y + 4, x + 70, y + 66], fill=(255, 215, 0))
        draw.text((x + 18, y + 8), n, fill=(0, 0, 0), font=f)
        tiles[n] = img_id
    sheet.save(out_path, quality=88)
    return tiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["images", "labels"], default="images")
    ap.add_argument("--sheets-per-batch", type=int, default=4)
    ap.add_argument("--ids-from", choices=["agent_fixed"], default=None,
                    help="labels mode: re-review only images whose labels were changed by apply_fixes")
    ap.add_argument("--ids-file", type=Path, default=None, help="labels mode: re-review exactly these ids")
    ap.add_argument("--prefix", default=None, help="batch name prefix (must sort after earlier rounds)")
    args = ap.parse_args()

    if args.mode == "images":
        rows = read_csv(CANDIDATES_CSV)
        items = [(r["id"], r["candidate"]) for r in rows]
        prefix = "batch"
    else:
        rows = read_csv(ACCEPTED_CSV)
        items = []
        for r in rows:
            ov = OVERLAYS_DIR / f"{r['id']}.jpg"
            if ov.exists():
                items.append((r["id"], str(ov.relative_to(ROOT)).replace("\\", "/")))
        prefix = "labels"
        if args.ids_from:
            from tools.dataset.common import LABELS_DIR

            def changed(i: str) -> bool:
                p = LABELS_DIR / f"{i}.json"
                return p.exists() and json.loads(p.read_text()).get("agent_fixed", False)

            items = [it for it in items if changed(it[0])]
            prefix = "relabel"  # sorts after labels_*, so merge_reviews lets these verdicts win
        if args.ids_file:
            wanted = set(args.ids_file.read_text().split())
            items = [it for it in items if it[0] in wanted]
        if args.prefix:
            prefix = args.prefix

    out_dir = SHEETS_DIR / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    # incremental: skip ids already in an existing batch, number new batches after the existing ones
    existing = sorted(out_dir.glob(f"{prefix}_*.json"))
    batched = set()
    for spec_path in existing:
        batched.update(json.loads(spec_path.read_text())["images"])
    items = [it for it in items if it[0] not in batched]
    start = len(existing)
    per_sheet = GRID * GRID
    per_batch = per_sheet * args.sheets_per_batch
    n_batches = 0
    for b in range(0, len(items), per_batch):
        name = f"{prefix}_{start + b // per_batch:03d}"
        out_json = REVIEW_DIR / args.mode / f"{name}.json"
        sheets, images = [], {}
        chunk = items[b: b + per_batch]
        for s in range(0, len(chunk), per_sheet):
            sp = out_dir / f"{name}_s{s // per_sheet}.jpg"
            tiles = make_sheet(chunk[s: s + per_sheet], sp)
            sheets.append({"sheet": str(sp.relative_to(ROOT)).replace("\\", "/"), "tiles": tiles})
        images = {i: rel for i, rel in chunk}
        spec = {"batch": name, "mode": args.mode, "sheets": sheets, "images": images,
                "output": str(out_json.relative_to(ROOT)).replace("\\", "/")}
        (out_dir / f"{name}.json").write_text(json.dumps(spec, indent=1))
        n_batches += 1
    print(f"{len(items)} new images -> {n_batches} new batches ({start} existing) in {out_dir}")


if __name__ == "__main__":
    main()
