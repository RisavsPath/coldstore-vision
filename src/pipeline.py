"""End-to-end inspection: Day 1 vs Day 2 -> per-sack diagnoses in a common aisle frame.

  keyframes (both days) -> Day 1 mosaic (+ per-keyframe transforms) -> keyframe pairing + homography
  -> YOLO on Day 1 keyframes and registered Day 2 frames -> lighting normalisation -> Farneback flow
  -> Hungarian IoU matching -> diagnostics -> cross-keyframe dedup in mosaic space -> bays/tiers
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.analytics.anomaly_engine import ACTIONS, MISSING, NEW, NORMAL, SEVERITY, diagnose, reasoning_text
from src.analytics.bay_indexer import index_bays
from src.analytics.lighting_normalizer import normalize_pair
from src.analytics.tracker import box_iou, match_by_centroid
from src.ingestion.keyframe_extractor import Keyframe, extract_keyframes
from src.registration.homography_warper import Registration, pair_keyframes, register
from src.registration.mosaicking import Mosaic, build_mosaic, composite_onto, project_box
from src.registration.optical_flow import farneback

log = logging.getLogger("coldstore")
UNVERIFIED = "UNVERIFIED"
SEVERITY.setdefault(UNVERIFIED, 1)
ACTIONS.setdefault(UNVERIFIED, "Re-scan: Day 2 walkthrough did not cover this sack")


@dataclass
class PairResult:
    i1: int
    j2: int
    reg: Registration
    img1: np.ndarray
    img2n: np.ndarray
    valid: np.ndarray
    sacks1: list
    sacks2: list
    global_hue: float
    flow: np.ndarray


@dataclass
class Observation:
    kf: int                       # Day 1 keyframe index
    pair: int | None              # index into Inspection.pairs, None if keyframe unpaired
    inst: object                  # Day 1 Instance (or Day 2 Instance for NEW)
    match: object | None          # matched Day 2 instance (registered coords)
    diag: dict | None
    mosaic_box: np.ndarray
    completeness: float           # min distance to frame border, px (higher = less truncated)
    global_id: int = -1


@dataclass
class SackRecord:
    sack_id: str
    bay: int
    tier: int
    slot: str
    status: str
    metrics: dict
    reasoning: str
    action: str
    obs: Observation
    crop1: np.ndarray | None = None
    crop2: np.ndarray | None = None


@dataclass
class Inspection:
    kf1: list[Keyframe]
    kf2: list[Keyframe]
    pairs: list[PairResult]
    mosaic1: Mosaic
    mosaic2: Mosaic
    sacks: list[SackRecord]
    observations: list[Observation]
    beams_mosaic: list[np.ndarray]
    detector_source: str
    src2_dir: Path
    kpis: dict = field(default_factory=dict)


def _completeness(box: np.ndarray, shape) -> float:
    h, w = shape[:2]
    return float(min(box[0], box[1], w - box[2], h - box[3]))


def _crop(img: np.ndarray, box: np.ndarray, pad: float = 0.15, size: int = 220) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    s = size / max(c.shape[:2])
    return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def box_iomin(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller box: a sack truncated at a frame edge still matches its full view."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    amin = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / amin) if amin > 0 else 0.0


def _match_confirmed(s1, s2, d_local: dict, cfg: dict) -> bool:
    """A centroid-only pairing (masks barely overlap in the rack-aligned frame) must also look the
    same after local alignment; otherwise a removed sack gets 'matched' to its neighbour."""
    inter, union = (s1.mask & s2.mask).sum(), (s1.mask | s2.mask).sum()
    global_iou = inter / union if union else 0.0
    if global_iou >= cfg["tracking"]["iou_match_threshold"]:
        return True
    return d_local.get("diff_area_frac", 1.0) <= cfg["tracking"].get("centroid_match_max_diff", 0.3)


def _truncated(box: np.ndarray, shape, edge_px: int = 4, sliver_frac: float = 0.06) -> bool:
    """Cut off by the pan direction (left/right frame edge) or a thin sliver. Sacks touching the
    top/bottom edge are normal for tall stacks and are NOT treated as truncated."""
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    at_side = x1 <= edge_px or x2 >= w - edge_px
    sliver = (x2 - x1) < sliver_frac * w or (y2 - y1) < sliver_frac * h
    return bool(at_side or sliver)


def diagnose_local(img1, img2n, s1, s2, ghue: float, cfg: dict, valid: np.ndarray, union2: np.ndarray,
                   flow_global: np.ndarray | None = None) -> dict:
    """Per-sack comparison after LOCAL alignment of the matched sack's own crop.

    A whole-frame homography cannot remove 3-D parallax across a stack; re-registering just the
    region around this sack (Day 2 crop -> Day 1 crop) does, so colour/texture/shape deltas
    compare the same fabric. Falls back to the globally aligned crop if local matching fails.
    """
    # structure (slump) in the rack-aligned frame, before any per-sack warping
    from src.analytics.anomaly_engine import erode
    from src.registration.optical_flow import slump_metrics

    em = erode(s1.mask, cfg["anomaly"]["mask_erode_px"])
    sm = slump_metrics(flow_global, em) if flow_global is not None else {"flow_vy": 0.0, "flow_div": 0.0}
    t1, t2 = np.nonzero(s1.mask.any(axis=1))[0], np.nonzero(s2.mask.any(axis=1))[0]
    structure = {"flow_vy": sm["flow_vy"], "flow_div": sm["flow_div"],
                 "top_drop_px": float(t2.min() - t1.min()) if len(t1) and len(t2) else 0.0}
    h, w = img1.shape[:2]
    bx = np.array([min(s1.box[0], s2.box[0]), min(s1.box[1], s2.box[1]),
                   max(s1.box[2], s2.box[2]), max(s1.box[3], s2.box[3])])
    pad = 0.15 * max(bx[2] - bx[0], bx[3] - bx[1])
    x1, y1 = int(max(0, bx[0] - pad)), int(max(0, bx[1] - pad))
    x2, y2 = int(min(w, bx[2] + pad)), int(min(h, bx[3] + pad))
    c1, c2 = img1[y1:y2, x1:x2], img2n[y1:y2, x1:x2]
    m1 = s1.mask[y1:y2, x1:x2]
    m2 = s2.mask[y1:y2, x1:x2].astype(np.uint8)
    v2 = valid[y1:y2, x1:x2].astype(np.uint8)
    u2 = union2[y1:y2, x1:x2].astype(np.uint8)
    lcfg = {**cfg, "registration": {**cfg["registration"], "min_inliers": 12}}
    reg = register(c1, c2, lcfg) if min(c1.shape[:2]) >= 40 else None
    if reg is not None and reg.ok:
        sh = c1.shape[:2]
        c2 = reg.warp(c2, sh)
        m2, v2, u2 = (reg.warp(x, sh, nearest=True) for x in (m2, v2, u2))
        v2 = v2 & reg.valid_mask(c2.shape, sh).astype(np.uint8)
    m2, v2, u2 = m2.astype(bool), v2.astype(bool), u2.astype(bool)
    inter, union = (m1 & m2).sum(), (m1 | m2).sum()
    local_iou = float(inter / union) if union else 0.0
    flow = farneback(c1, c2, cfg)
    d = diagnose(c1, c2, m1, m2, max(local_iou, cfg["tracking"]["iou_match_threshold"]), flow, ghue, cfg,
                 valid2=v2, day2_sacks=u2, structure=structure).to_dict()
    d["match_iou"] = round(local_iou, 4)
    d["local_aligned"] = bool(reg is not None and reg.ok)
    return d


def _group(obs: list[Observation], thr: float = 0.6) -> list[list[Observation]]:
    """Cluster observations of the same physical sack across keyframes (mosaic-space IoMin).

    Most complete (least truncated) observation first, so it becomes the group's anchor.
    """
    groups: list[list[Observation]] = []
    for o in sorted(obs, key=lambda o: -o.completeness):
        for g in groups:
            if box_iomin(g[0].mosaic_box, o.mosaic_box) >= thr:
                g.append(o)
                break
        else:
            groups.append([o])
    return groups


def run_inspection(day1_dir: str | Path, day2_dir: str | Path, cfg: dict, detector) -> Inspection:
    log.info("extracting keyframes")
    kf1, _ = extract_keyframes(day1_dir, cfg)
    kf2, _ = extract_keyframes(day2_dir, cfg)
    log.info("keyframes: day1=%d day2=%d", len(kf1), len(kf2))

    mosaic1 = build_mosaic([k.image for k in kf1], cfg)
    have_pose = all(k.pose is not None for k in kf1 + kf2)
    from src.registration import robust_matcher

    if cfg["registration"].get("matcher", "loftr") == "loftr" and not have_pose and robust_matcher.available():
        log.info("pairing keyframes with CLIP retrieval + LoFTR")
        pairs_raw = robust_matcher.pair_keyframes_robust(kf1, kf2, cfg)
        cfg = {**cfg, "anomaly": {**cfg["anomaly"], "slump_cross_view": True}}
    else:
        pairs_raw = pair_keyframes(kf1, kf2, cfg)
    log.info("registered pairs: %d / %d", len(pairs_raw), len(kf1))

    det1 = [detector.detect(k.image) for k in kf1]
    pair_of_kf = {i: p for p, (i, _, _) in enumerate(pairs_raw)}
    a = cfg["anomaly"]
    thr = cfg["tracking"]["iou_match_threshold"]

    pairs: list[PairResult] = []
    obs: list[Observation] = []
    new_obs: list[Observation] = []
    beams_mosaic: list[np.ndarray] = []

    for i, k in enumerate(kf1):
        for inst in det1[i]:
            if inst.cls == "rack_beam":
                beams_mosaic.append(project_box(inst.box, mosaic1.transforms[i]))

    for p_idx, (i, j, reg) in enumerate(pairs_raw):
        img1 = kf1[i].image
        shape = img1.shape[:2]
        img2r = reg.warp(kf2[j].image, shape)
        valid = cv2.erode(reg.valid_mask(kf2[j].image.shape, shape).astype(np.uint8),
                          np.ones((9, 9), np.uint8)).astype(bool)
        det2 = [d for d in detector.detect(img2r) if (d.mask & valid).sum() >= 0.6 * max(d.area, 1)]
        sacks1 = [d for d in det1[i] if d.cls == "sack"]
        sacks2 = [d for d in det2 if d.cls == "sack"]
        matches, un1, un2 = match_by_centroid(sacks1, sacks2, cfg["tracking"].get("centroid_max_cost", 0.6))
        mdict = {m[0]: (m[1], m[2]) for m in matches}
        # reliability gate: if most Day 1 sacks inside the Day 2 view found no partner, the two
        # viewpoints are too different (or alignment failed) - never turn that into MISSING/CRITICAL
        in_view = [k for k, s in enumerate(sacks1) if (s.mask & valid).sum() >= 0.5 * max(s.area, 1)]
        matched_in_view = sum(1 for k in in_view if k in mdict)
        reliable = bool(in_view) and matched_in_view / len(in_view) >= cfg["tracking"].get("min_pair_match_frac", 0.5)
        img2n, _, ghue = normalize_pair(img1, img2r, valid, [s.mask for s in sacks1], [s.mask for s in sacks2],
                                        [sacks1[u].mask for u in un1] + [sacks2[u].mask for u in un2],
                                        cfg["lighting"]["histogram_match"])
        flow = farneback(img1, img2n, cfg)
        depth1 = kf1[i].depth
        depth2 = reg.warp(kf2[j].depth, shape, nearest=True) if (kf2[j].depth is not None and depth1 is not None) else None
        pairs.append(PairResult(i, j, reg, img1, img2n, valid, sacks1, sacks2, ghue, flow))

        union2 = np.zeros(shape, bool)
        for s2 in sacks2:
            union2 |= s2.mask
        for s_idx, s in enumerate(sacks1):
            m = mdict.get(s_idx)
            m_inst = sacks2[m[0]] if m else None
            if not reliable:
                d = {"status": UNVERIFIED, "action": ACTIONS[UNVERIFIED],
                     "reasons": [f"Day 1 and Day 2 views too different to compare reliably "
                                 f"({matched_in_view}/{len(in_view)} sacks matched in this view)"]}
            elif m_inst is not None and _match_confirmed(s, m_inst, d_local := diagnose_local(
                    img1, img2n, s, m_inst, ghue, cfg, valid, union2, flow), cfg):
                d = d_local
            elif _truncated(s.box, shape) and m_inst is None or s.area < cfg["anomaly"].get("min_missing_area_frac", 0.01) * shape[0] * shape[1]:
                # cut off at the frame edge / tiny partial view: too little evidence for a MISSING call
                d = {"status": UNVERIFIED, "action": ACTIONS[UNVERIFIED],
                     "reasons": ["only partially visible in this view - occupancy not judged"]}
            else:
                d = diagnose(img1, img2n, s.mask, None, 0.0, flow, ghue, cfg, valid2=valid, depth1=depth1,
                             depth2=depth2, day2_sacks=union2).to_dict()
            obs.append(Observation(i, p_idx, s, m_inst, d, project_box(s.box, mosaic1.transforms[i]),
                                   _completeness(s.box, shape)))
        if not reliable:
            continue
        for u in un2:
            s2 = sacks2[u]
            if any(box_iou(s2.box, s1.box) > 0.1 for s1 in sacks1):
                continue
            new_obs.append(Observation(i, p_idx, s2, None,
                                       {"status": NEW, "reasons": ["sack present on Day 2 at a slot empty on Day 1"],
                                        "action": ACTIONS[NEW]},
                                       project_box(s2.box, mosaic1.transforms[i]), _completeness(s2.box, shape)))

    # Day 1 sacks in keyframes without a Day 2 pair
    for i, k in enumerate(kf1):
        if i in pair_of_kf:
            continue
        for s in det1[i]:
            if s.cls == "sack":
                obs.append(Observation(i, None, s, None, None, project_box(s.box, mosaic1.transforms[i]),
                                       _completeness(s.box, k.image.shape)))

    groups = _group(obs)
    reps: list[Observation] = []
    for g in groups:
        verified = [o for o in g if o.diag is not None and "outside Day 2 coverage" not in o.diag.get("reasons", [])]
        verified = verified or [o for o in g if o.diag is not None]
        # majority vote over keyframe pairs that actually judged this sack (UNVERIFIED pairs abstain);
        # ties go to the more severe status. One lucky/unlucky view no longer decides alone.
        judged = [o for o in verified if o.diag.get("status") != UNVERIFIED]
        if judged:
            from collections import Counter

            votes = Counter(o.diag["status"] for o in judged)
            winner = max(votes, key=lambda st: (votes[st], SEVERITY.get(st, 0)))
            rep = next(o for o in judged if o.diag["status"] == winner)  # most complete with that status
            if len(judged) > 1:
                rep.diag = {**rep.diag, "votes": dict(votes)}
        else:
            rep = verified[0] if verified else g[0]
        reps.append(rep)
        for o in g:
            o.global_id = len(reps) - 1

    new_groups = [g for g in _group(new_obs) if not any(box_iomin(g[0].mosaic_box, r.mosaic_box) > 0.3 for r in reps)]
    new_reps = []
    for g in new_groups:
        for o in g:
            o.global_id = len(reps) + len(new_reps)
        new_reps.append(g[0])

    all_reps = reps + new_reps
    bays = index_bays([r.mosaic_box for r in all_reps], beams_mosaic, *mosaic1.image.shape[1::-1],
                      min_aspect=cfg["bays"]["min_beam_aspect"])

    order = sorted(range(len(all_reps)), key=lambda k: (bays[k][0], bays[k][1], all_reps[k].mosaic_box[0]))
    slot_counter: dict[tuple[int, int], int] = {}
    sacks: list[SackRecord] = [None] * len(all_reps)  # type: ignore
    for n, k in enumerate(order):
        r = all_reps[k]
        bay, tier = bays[k]
        slot_counter[(bay, tier)] = slot_counter.get((bay, tier), 0) + 1
        slot = f"B{bay:02d}-T{tier:02d}-S{slot_counter[(bay, tier)]:02d}"
        if r.diag is None:
            d = {"status": UNVERIFIED, "reasons": ["Day 2 walkthrough did not cover this sack"],
                 "action": ACTIONS[UNVERIFIED]}
        else:
            d = r.diag
        rec = SackRecord(f"Sack_{n + 1:03d}", bay, tier, slot, d["status"], d,
                         reasoning_text(f"Bay {bay:02d}, Tier {tier:02d} ({slot})", d), d["action"], r)
        img1 = kf1[r.kf].image
        if r.pair is not None:
            pr = pairs[r.pair]
            rec.crop1 = _crop(pr.img1, r.inst.box)
            rec.crop2 = _crop(pr.img2n, r.inst.box)
        else:
            rec.crop1 = _crop(img1, r.inst.box)
        sacks[k] = rec
    sacks.sort(key=lambda s: s.sack_id)

    # Day 2 mosaic on Day 1 canvas
    d2_frames, d2_T = [], []
    for pr in pairs:
        d2_frames.append(kf2[pr.j2].image)
        d2_T.append(mosaic1.transforms[pr.i1] @ pr.reg.H)
    mosaic2 = composite_onto(mosaic1, d2_frames, d2_T) if pairs else Mosaic(np.zeros_like(mosaic1.image), [], np.zeros(mosaic1.image.shape[:2], bool))

    insp = Inspection(kf1, kf2, pairs, mosaic1, mosaic2, sacks, obs + new_obs, beams_mosaic,
                      getattr(detector, "source", type(detector).__name__), Path(day2_dir))
    insp.kpis = compute_kpis(sacks)
    return insp


def compute_kpis(sacks: list[SackRecord]) -> dict:
    from collections import Counter

    c = Counter(s.status for s in sacks)
    day1_total = sum(v for k, v in c.items() if k != NEW)
    present = c[NORMAL] + c["WARNING"] + c["CRITICAL"]
    health = 100.0 * (c[NORMAL] + 0.5 * c["WARNING"]) / present if present else 100.0
    return {
        "total_monitored": day1_total,
        "status_counts": dict(c),
        "missing": c[MISSING],
        "new": c[NEW],
        "net_inventory_change": c[NEW] - c[MISSING],
        "facility_health_index": round(health, 1),
        "urgent_actions": c["CRITICAL"] + c[MISSING],
        "unverified": c[UNVERIFIED],
    }
