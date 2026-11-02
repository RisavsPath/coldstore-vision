"""Step 2: cheap automatic filtering before the reviewer agent.

- drops corrupt / small / extreme-aspect images
- perceptual-hash dedup (Hamming <= --hash-dist)
- OpenCLIP zero-shot relevance score; very low scores are dropped, the rest go to review

Writes data/scraped/candidates/<id>.jpg and data/scraped/candidates.csv (sorted by clip score).
"""
from __future__ import annotations

import argparse
import shutil

import imagehash
import torch
from PIL import Image
from tqdm import tqdm

from tools.dataset.common import CANDIDATES_CSV, CANDIDATES_DIR, MANIFEST, ROOT, read_csv, write_csv

POSITIVE = [
    "a photo of many sacks stacked on top of each other",
    "a photo of jute sacks of potatoes stacked in a warehouse",
    "a photo of mesh bags of onions piled up",
    "a photo of net bags of garlic stacked",
    "a photo of gunny bags in a cold storage",
    "a photo of sacks on a pallet rack",
]
NEGATIVE = [
    "a product photo of a single bag on a white background",
    "a logo or an illustration",
    "a cartoon or a 3D render",
    "a photo of a person's face",
    "a photo of a handbag or fashion accessory",
    "a photo of loose vegetables without bags",
    "a screenshot of a website or text document",
    "a photo of sandbags in a flood",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-side", type=int, default=480)
    ap.add_argument("--max-aspect", type=float, default=3.0)
    ap.add_argument("--hash-dist", type=int, default=6)
    ap.add_argument("--min-clip", type=float, default=0.20, help="drop images whose positive prob is below this")
    args = ap.parse_args()

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    with torch.no_grad():
        txt = model.encode_text(tok(POSITIVE + NEGATIVE))
        txt /= txt.norm(dim=-1, keepdim=True)

    rows = read_csv(MANIFEST)
    stats = {"input": len(rows), "bad": 0, "small": 0, "aspect": 0, "dup": 0, "low_clip": 0}
    hashes: list[tuple[imagehash.ImageHash, str]] = []
    kept = []
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    for r in tqdm(rows, desc="prefilter"):
        p = ROOT / r["file"]
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            stats["bad"] += 1
            continue
        w, h = im.size
        if min(w, h) < args.min_side:
            stats["small"] += 1
            continue
        if max(w, h) / min(w, h) > args.max_aspect:
            stats["aspect"] += 1
            continue
        ph = imagehash.phash(im)
        if any(ph - other <= args.hash_dist for other, _ in hashes):
            stats["dup"] += 1
            continue
        with torch.no_grad():
            f = model.encode_image(preprocess(im).unsqueeze(0))
            f /= f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ txt.T).softmax(dim=-1)[0]
        pos = float(probs[: len(POSITIVE)].sum())
        if pos < args.min_clip:
            stats["low_clip"] += 1
            continue
        hashes.append((ph, p.stem))
        dst = CANDIDATES_DIR / f"{p.stem}.jpg"
        if not dst.exists():
            shutil.copy2(p, dst)
        kept.append({**r, "id": p.stem, "phash": str(ph), "clip_pos": round(pos, 4),
                     "candidate": str(dst.relative_to(ROOT)).replace("\\", "/")})

    kept.sort(key=lambda r: -r["clip_pos"])
    write_csv(CANDIDATES_CSV, kept)
    print({**stats, "kept": len(kept)})


if __name__ == "__main__":
    main()
