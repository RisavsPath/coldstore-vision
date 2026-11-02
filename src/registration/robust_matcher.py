"""Viewpoint/lighting-robust Day1<->Day2 frame matching for handheld walks without poses.

1. Retrieval: OpenCLIP image embeddings shortlist the top-k most similar Day 2 keyframes for each
   Day 1 keyframe (cheap, tolerant to big viewpoint changes).
2. Verification: LoFTR (kornia, detector-free dense matcher) + MAGSAC homography on each shortlisted
   pair. SIFT + ratio test fails across days because stacked mesh sacks are repetitive and the two
   handheld walks differ in distance, angle and exposure; LoFTR still finds reliable matches.
Both models run on CPU. Returns None-safe results so the caller can fall back to SIFT.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np

from src.registration.homography_warper import Registration, _sane

log = logging.getLogger("coldstore")
LOFTR_WIDTH = 640


@lru_cache(maxsize=1)
def _loftr():
    import kornia.feature as KF

    return KF.LoFTR(pretrained="outdoor").eval()


@lru_cache(maxsize=1)
def _clip():
    import open_clip

    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    return model.eval(), pre


def available() -> bool:
    try:
        import kornia  # noqa: F401
        import open_clip  # noqa: F401
        return True
    except ImportError:
        return False


def embed(images: list[np.ndarray]) -> np.ndarray:
    import torch
    from PIL import Image

    model, pre = _clip()
    batch = torch.stack([pre(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in images])
    with torch.inference_mode():
        f = model.encode_image(batch)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.numpy()


def _gray_tensor(img: np.ndarray):
    import torch

    h, w = img.shape[:2]
    nw = LOFTR_WIDTH
    nh = int(h * nw / w) // 8 * 8
    g = cv2.cvtColor(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(g)[None, None].float() / 255.0, (w / nw, h / nh)


def loftr_register(ref: np.ndarray, mov: np.ndarray, min_inliers: int = 40, conf: float = 0.5) -> Registration:
    """Homography mapping mov pixels -> ref pixels (full resolution)."""
    import torch

    t0, s0 = _gray_tensor(ref)
    t1, s1 = _gray_tensor(mov)
    with torch.inference_mode():
        out = _loftr()({"image0": t0, "image1": t1})
    k0 = out["keypoints0"].numpy()
    k1 = out["keypoints1"].numpy()
    keep = out["confidence"].numpy() > conf
    k0, k1 = k0[keep] * s0, k1[keep] * s1
    if len(k0) < 8:
        return Registration(None, 0, len(k0), False)
    H, inl = cv2.findHomography(k1.astype(np.float32), k0.astype(np.float32), cv2.USAC_MAGSAC, 4.0)
    n_in = int(inl.sum()) if inl is not None else 0
    ok = H is not None and n_in >= min_inliers and _sane(H, ref.shape)
    return Registration(H, n_in, len(k0), ok)


def pair_keyframes_robust(kf1: list, kf2: list, cfg: dict) -> list[tuple[int, int, Registration]]:
    rc = cfg["registration"]
    topk = rc.get("retrieval_topk", 4)
    min_in = rc.get("loftr_min_inliers", 40)
    e1 = embed([k.image for k in kf1])
    e2 = embed([k.image for k in kf2])
    sim = e1 @ e2.T
    pairs = []
    for i, k1 in enumerate(kf1):
        cand = list(np.argsort(-sim[i])[:topk])
        best = None
        for j in cand:
            r = loftr_register(k1.image, kf2[j].image, min_in)
            if r.ok and (best is None or r.inliers > best[1].inliers):
                best = (int(j), r)
        if best is not None:
            pairs.append((i, best[0], best[1]))
        log.debug("kf %d: best %s", i, None if best is None else (best[0], best[1].inliers))
    return pairs
