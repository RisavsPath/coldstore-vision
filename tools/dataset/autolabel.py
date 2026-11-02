"""Step 4: draft polygon labels for accepted images.

sack:      FastSAM "segment everything" proposals -> geometric filters (area, solidity, nesting)
           -> OpenCLIP crop classifier (sack vs wall/floor/person/clothes/...) with a lower bar when
           the open-vocabulary YOLOE detector agrees. Tested: far higher recall on dense stacks than
           YOLOE text prompts alone (which miss most sacks in a pile).
rack_beam / empty_slot: YOLOE text prompts at a high confidence (rare in web images; these classes
           mostly come from the Day 1 footage later).

Writes data/scraped/autolabels/<id>.txt (YOLO-seg) + <id>.json meta. Human-verified labels
(from cvat_roundtrip import) are never overwritten.

Usage: python -m tools.dataset.autolabel [--limit N] [--overwrite] [--clip-thr 0.6]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from tools.dataset.common import ACCEPTED_CSV, LABELS_DIR, ROOT, read_csv

HUB = ROOT / "models" / "hub"
SACK_TEXT = ["a photo of a sack", "a photo of a jute sack", "a photo of a mesh bag of onions",
             "a photo of a white woven polypropylene sack", "a photo of a bag of potatoes",
             "a photo of a net bag of garlic", "a photo of a gunny bag"]
NEG_TEXT = ["a photo of a wall", "a photo of the floor", "a photo of the ground", "a photo of the sky",
            "a photo of a person", "a photo of a face", "a photo of clothes", "a photo of shoes",
            "a photo of a wooden pallet", "a photo of a metal fence", "a photo of a door",
            "a photo of a pile of many sacks", "a photo of a building", "a photo of a vehicle",
            "a photo of a tree", "a photo of loose vegetables", "a photo of text",
            "a photo of a man carrying a sack on his head", "a photo of a dark doorway"]
YOLOE_PROMPTS = {"sack": 0, "jute sack": 0, "gunny bag": 0, "mesh bag of onions": 0, "bag of potatoes": 0,
                 "net bag of garlic": 0, "steel rack beam": 1, "metal shelf upright": 1, "empty shelf space": 2}
CLASS_CONF = {1: 0.35, 2: 0.45}


def solidity(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(cv2.contourArea(c) / hull) if hull > 0 else 0.0


def to_polygon(mask: np.ndarray, max_pts: int = 100) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    eps = 0.002 * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(c) > max_pts:
        c = c[np.linspace(0, len(c) - 1, max_pts).astype(int)]
    return c if len(c) >= 3 else None


class AutoLabeler:
    def __init__(self, clip_thr: float, yoloe_bonus: float, device: str = "cpu"):
        import open_clip
        from ultralytics import YOLOE, FastSAM

        os.chdir(HUB)  # ultralytics downloads weights into the CWD
        HUB.mkdir(parents=True, exist_ok=True)
        self.fastsam = FastSAM("FastSAM-x.pt")
        self.yoloe = YOLOE("yoloe-11s-seg.pt")
        self.ye_names = list(YOLOE_PROMPTS)
        self.yoloe.set_classes(self.ye_names, self.yoloe.get_text_pe(self.ye_names))
        self.clip, _, self.pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.clip.eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            t = self.clip.encode_text(tok(SACK_TEXT + NEG_TEXT))
            self.text = t / t.norm(dim=-1, keepdim=True)
        self.clip_thr, self.bonus, self.device = clip_thr, yoloe_bonus, device
        self.proto_sim, self.proto_min_score = 0.86, 0.20

    def clip_scores(self, img: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Per-mask P(sack) over the text prompts, and the normalised image embeddings."""
        crops = []
        for m in masks:
            ys, xs = np.nonzero(m)
            y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            c = img[y1:y2, x1:x2].copy()
            c[~m[y1:y2, x1:x2]] = 127
            crops.append(self.pre(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))))
        out, feats = [], []
        with torch.no_grad():
            for b in range(0, len(crops), 64):
                f = self.clip.encode_image(torch.stack(crops[b:b + 64]))
                f /= f.norm(dim=-1, keepdim=True)
                p = (100.0 * f @ self.text.T).softmax(dim=-1)
                out.append(p[:, :len(SACK_TEXT)].sum(1).numpy())
                feats.append(f.numpy())
        if not out:
            return np.zeros(0), np.zeros((0, 512))
        return np.concatenate(out), np.concatenate(feats)

    def label(self, path: Path) -> tuple[list[str], dict]:
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        area = H * W
        lines: list[str] = []
        stats = {"proposals": 0, "candidates": 0, "sacks": 0, "beams": 0, "empty": 0}

        # YOLOE pass: structure classes + agreement signal for sacks
        ye = self.yoloe.predict(img, conf=0.05, imgsz=960, iou=0.5, agnostic_nms=True, retina_masks=True,
                                device=self.device, verbose=False)[0]
        ye_sack_masks = []
        if ye.masks is not None:
            ye_masks = ye.masks.data.cpu().numpy() > 0.5
            for k in range(len(ye.boxes)):
                cls = YOLOE_PROMPTS[self.ye_names[int(ye.boxes.cls[k])]]
                conf = float(ye.boxes.conf[k])
                if cls == 0:
                    if conf >= 0.08:
                        ye_sack_masks.append(ye_masks[k])
                elif conf >= CLASS_CONF[cls]:
                    poly = to_polygon(ye_masks[k])
                    if poly is not None:
                        lines.append(f"{cls} " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in poly))
                        stats["beams" if cls == 1 else "empty"] += 1

        # FastSAM proposals
        fs = self.fastsam(img, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, device=self.device, verbose=False)[0]
        if fs.masks is None:
            return lines, stats
        masks = list(fs.masks.data.cpu().numpy() > 0.5)
        stats["proposals"] = len(masks)
        cand = []
        for m in masks:
            a = m.sum() / area
            if a < 0.002 or a > 0.25:
                continue
            if solidity(m) < 0.80:
                continue
            cand.append(m)
        cand.sort(key=lambda m: -m.sum())
        kept_geom = []
        for m in cand:  # drop masks mostly inside a bigger candidate (logos, patches on a sack)
            if any((m & k).sum() > 0.7 * m.sum() for k in kept_geom):
                continue
            kept_geom.append(m)
        stats["candidates"] = len(kept_geom)
        if not kept_geom:
            return lines, stats
        scores, feats = self.clip_scores(img, kept_geom)
        accept = np.zeros(len(kept_geom), bool)
        for i, (m, s) in enumerate(zip(kept_geom, scores)):
            agree = any((m & y).sum() > 0.5 * min(m.sum(), y.sum()) for y in ye_sack_masks)
            accept[i] = s >= self.clip_thr or (agree and s >= self.clip_thr - self.bonus)
        # sacks in one pile look alike: accept candidates close to this image's confident-sack prototype
        if accept.sum() >= 3:
            proto = feats[accept].mean(0)
            proto /= np.linalg.norm(proto)
            sim = feats @ proto
            accept |= (sim >= self.proto_sim) & (scores >= self.proto_min_score)
        for m, ok in zip(kept_geom, accept):
            if not ok:
                continue
            poly = to_polygon(m)
            if poly is not None:
                lines.append("0 " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in poly))
                stats["sacks"] += 1
        return lines, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--clip-thr", type=float, default=0.60)
    ap.add_argument("--yoloe-bonus", type=float, default=0.20)
    ap.add_argument("--ids", nargs="*", help="only these image ids")
    args = ap.parse_args()

    rows = read_csv(ACCEPTED_CSV)
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    rows = rows[: args.limit or None]
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    labeler = AutoLabeler(args.clip_thr, args.yoloe_bonus)
    total = {"images": 0, "skipped_human": 0, "sacks": 0, "beams": 0, "empty_slots": 0, "no_instances": 0}
    for r in tqdm(rows, desc="autolabel"):
        out = LABELS_DIR / f"{r['id']}.txt"
        meta = LABELS_DIR / f"{r['id']}.json"
        if meta.exists() and json.loads(meta.read_text()).get("human_verified"):
            total["skipped_human"] += 1
            continue
        if out.exists() and not args.overwrite:
            continue
        lines, st = labeler.label(ROOT / r["accepted"])
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        meta.write_text(json.dumps({"human_verified": False, "source": "fastsam+clip+yoloe", **st}))
        total["images"] += 1
        total["sacks"] += st["sacks"]
        total["beams"] += st["beams"]
        total["empty_slots"] += st["empty"]
        total["no_instances"] += not lines
    print(total)


if __name__ == "__main__":
    main()
