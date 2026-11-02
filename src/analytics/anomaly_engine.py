"""Module 4: per-sack diagnostics — missing inventory, surface spoilage (HSV + GLCM), structural slump.

All colour/texture measurements use the YOLO mask eroded by `mask_erode_px` (spec: 5-10 px) so
parallax at sack boundaries never produces false deltas. Never whole-frame subtraction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from src.analytics.lighting_normalizer import circular_mean_hue_deg, hue_delta_deg
from src.registration.optical_flow import slump_metrics

NORMAL, WARNING, CRITICAL, MISSING, NEW = "NORMAL", "WARNING", "CRITICAL", "MISSING", "NEW"
UNVERIFIED_STATUS = "UNVERIFIED"
SEVERITY = {NORMAL: 0, NEW: 1, WARNING: 2, CRITICAL: 3, MISSING: 4}
ACTIONS = {
    NORMAL: "None",
    NEW: "Verify new stock entry against inward register",
    WARNING: "Re-inspect within 48 h; check bay temperature/humidity",
    CRITICAL: "Pull sack for manual inspection; isolate to stop spread",
    MISSING: "Reconcile against dispatch log; investigate if undocumented",
}


@dataclass
class SackDiagnosis:
    status: str
    diff_area_frac: float = 0.0
    match_iou: float = 0.0
    depth_delta_m: float | None = None
    hue_delta_deg: float = 0.0
    hue_rel_deg: float = 0.0
    necrosis_d1: float = 0.0
    necrosis_d2: float = 0.0
    glcm_contrast_d1: float = 0.0
    glcm_contrast_d2: float = 0.0
    glcm_contrast_delta: float = 0.0
    glcm_homog_d1: float = 0.0
    glcm_homog_d2: float = 0.0
    glcm_homog_delta: float = 0.0
    flow_vy: float = 0.0
    flow_div: float = 0.0
    top_drop_px: float = 0.0
    reasons: list[str] = field(default_factory=list)
    action: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    e = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    return e if e.sum() >= 50 else mask.astype(bool)


def necrosis_fraction(img_bgr: np.ndarray, mask: np.ndarray, p: dict) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    nec = (h >= p["h_min"]) & (h <= p["h_max"]) & (s > p["s_min"]) & (v < p["v_max"])
    n = mask.sum()
    return float((nec & mask).sum() / n) if n else 0.0


def masked_glcm(img_bgr: np.ndarray, mask: np.ndarray, distances, angles_deg, levels: int = 32) -> tuple[float, float]:
    """GLCM contrast & homogeneity using only pixel pairs fully inside the mask.

    Pixels outside the mask get grey level 0 and in-mask pixels 1..levels-1; row/col 0 is dropped
    before normalising, so mask borders contribute nothing.
    """
    from skimage.feature import graycomatrix

    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return 0.0, 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    # light smoothing on both days equalises the interpolation blur of the homography warp
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    m = mask[y0:y1, x0:x1]
    q = (g.astype(np.int32) * (levels - 1) // 256 + 1).astype(np.uint8)
    q[~m] = 0
    P = graycomatrix(q, distances=list(distances), angles=[np.radians(a) for a in angles_deg],
                     levels=levels + 1, symmetric=True, normed=False).astype(np.float64)
    P = P[1:, 1:]
    s = P.sum(axis=(0, 1), keepdims=True)
    P = np.divide(P, s, out=np.zeros_like(P), where=s > 0)
    i, j = np.ogrid[0:levels, 0:levels]
    contrast = float((P * ((i - j) ** 2)[..., None, None]).sum(axis=(0, 1)).mean())
    homog = float((P / (1.0 + ((i - j) ** 2)[..., None, None])).sum(axis=(0, 1)).mean())
    return contrast, homog


def saturated_fraction(img_bgr: np.ndarray, mask: np.ndarray, min_sat: int = 40) -> float:
    n = mask.sum()
    if not n:
        return 0.0
    s = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1]
    return float((s[mask] >= min_sat).mean())


def _rel(a: float, b: float) -> float:
    return float((b - a) / a) if a > 1e-9 else 0.0


def diagnose(img1: np.ndarray, img2n: np.ndarray, mask1: np.ndarray, mask2: np.ndarray | None,
             match_iou: float, flow: np.ndarray | None, global_hue: float, cfg: dict,
             valid2: np.ndarray | None = None, depth1: np.ndarray | None = None,
             depth2: np.ndarray | None = None, day2_sacks: np.ndarray | None = None,
             structure: dict | None = None) -> SackDiagnosis:
    """day2_sacks: union of all Day 2 sack masks (registered). Used for occupancy: a Day 1 sack whose
    area is no longer covered by ANY Day 2 sack is missing even if the gap looks similar (e.g. the
    sack behind it now shows through)."""
    a = cfg["anomaly"]
    em = erode(mask1, a["mask_erode_px"])
    if valid2 is not None:
        em = em & valid2
        if em.sum() < 50:
            return SackDiagnosis(NORMAL, reasons=["outside Day 2 coverage"], action="Re-scan bay")

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.int16)
    g2 = cv2.cvtColor(img2n, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff_frac = float((np.abs(g1 - g2)[em] > a["diff_intensity_threshold"]).mean())

    dz = None
    if depth1 is not None and depth2 is not None:
        d1, d2 = depth1[em], depth2[em]
        ok = (d1 > 0.05) & (d2 > 0.05) & np.isfinite(d1) & np.isfinite(d2)
        if ok.sum() > 30:
            dz = float(np.median(d2[ok]) - np.median(d1[ok]))

    diag = SackDiagnosis(NORMAL, diff_area_frac=diff_frac, match_iou=match_iou, depth_delta_m=dz)

    # ---- occlusion -------------------------------------------------------------------------
    # Rot darkens/browns a sack; it never turns it into a different colour family. If the Day 2 view
    # of this spot has a hue > occlusion_hue_deg away from the Day 1 sack, something else is in view
    # (pillar, person, forklift, another object in front) -> don't call it MISSING or spoiled.
    if min(saturated_fraction(img1, em), saturated_fraction(img2n, em)) >= a.get("hue_min_saturated_frac", 0.3):
        h_now = hue_delta_deg(circular_mean_hue_deg(img1, em), circular_mean_hue_deg(img2n, em)) - global_hue
        h_now = (h_now + 180) % 360 - 180
        if abs(h_now) >= a.get("occlusion_hue_deg", 90.0):
            return SackDiagnosis(UNVERIFIED_STATUS, diff_area_frac=diff_frac, match_iou=match_iou, hue_rel_deg=h_now,
                                 reasons=[f"view blocked: a different-coloured object covers this spot on Day 2 "
                                          f"(hue {h_now:+.0f}°) - e.g. a pillar or person in front"],
                                 action="Re-scan this spot from a clear viewpoint")

    # ---- occupancy ------------------------------------------------------------------------
    coverage = float((em & day2_sacks).sum() / em.sum()) if day2_sacks is not None else None
    gone = (mask2 is None or match_iou < cfg["tracking"]["iou_match_threshold"]) and (coverage is None or coverage < 0.3)
    depth_says_gone = dz is not None and dz > a["missing_depth_delta_m"]
    changed = diff_frac > a["missing_diff_area_frac"] or (
        coverage is not None and coverage < 0.15 and diff_frac > a.get("missing_diff_area_frac_uncovered", 0.2))
    if gone and changed and (dz is None or depth_says_gone):
        diag.status = MISSING
        cov = f"; only {coverage:.0%} still covered by any Day 2 sack" if coverage is not None else ""
        diag.reasons.append(f"{diff_frac:.0%} of the sack area changed and no Day 2 sack matches it{cov}")
        if dz is not None:
            diag.reasons.append(f"LiDAR depth increased by {dz:.2f} m (beam reaches the back wall)")
        diag.action = ACTIONS[MISSING]
        return diag

    # ---- colour / necrosis (on Day 2's own mask where available, eroded) -----------------------
    em2 = erode(mask2, a["mask_erode_px"]) & (valid2 if valid2 is not None else True) if mask2 is not None else em
    if em2.sum() < 50:
        em2 = em
    p = a["necrosis_hsv"]
    diag.necrosis_d1 = necrosis_fraction(img1, em, p)
    diag.necrosis_d2 = necrosis_fraction(img2n, em2, p)
    # hue is only meaningful on saturated pixels: pale jute / white woven sacks give random hue
    if min(saturated_fraction(img1, em), saturated_fraction(img2n, em2)) >= a.get("hue_min_saturated_frac", 0.3):
        diag.hue_delta_deg = hue_delta_deg(circular_mean_hue_deg(img1, em), circular_mean_hue_deg(img2n, em2))
        diag.hue_rel_deg = diag.hue_delta_deg - global_hue

    # ---- texture --------------------------------------------------------------------------
    c1, h1 = masked_glcm(img1, em, a["glcm_distances"], a["glcm_angles_deg"])
    c2, h2 = masked_glcm(img2n, em2, a["glcm_distances"], a["glcm_angles_deg"])
    diag.glcm_contrast_d1, diag.glcm_contrast_d2, diag.glcm_homog_d1, diag.glcm_homog_d2 = c1, c2, h1, h2
    diag.glcm_contrast_delta, diag.glcm_homog_delta = _rel(c1, c2), _rel(h1, h2)

    # ---- structure ------------------------------------------------------------------------
    # (structure may be measured by the caller in the rack-aligned frame: per-sack local alignment
    #  would otherwise warp a slumped sack back into place)
    if structure is not None:
        diag.flow_vy, diag.flow_div, diag.top_drop_px = structure["flow_vy"], structure["flow_div"], structure["top_drop_px"]
    elif flow is not None:
        sm = slump_metrics(flow, em)
        diag.flow_vy, diag.flow_div = sm["flow_vy"], sm["flow_div"]
    if mask2 is not None and structure is None:
        t1 = np.nonzero(mask1.any(axis=1))[0]
        t2 = np.nonzero(mask2.any(axis=1))[0]
        if len(t1) and len(t2):
            diag.top_drop_px = float(t2.min() - t1.min())  # positive = sack top moved down

    # ---- decision ---------------------------------------------------------------------------
    nec_gain = diag.necrosis_d2 - diag.necrosis_d1
    glcm_break = diag.glcm_contrast_delta > a["glcm_contrast_rise"] and diag.glcm_homog_delta < a["glcm_homogeneity_drop"]
    settle = max(diag.flow_vy, diag.top_drop_px)
    compress = diag.flow_div < 0 or diag.top_drop_px > a["slump_warning_px"]
    crit, warn = [], []

    if nec_gain >= a["necrosis_critical_frac"]:
        crit.append(f"necrotic-tone area grew from {diag.necrosis_d1:.1%} to {diag.necrosis_d2:.1%}")
    elif nec_gain >= a["necrosis_warning_frac"]:
        warn.append(f"necrotic-tone area grew by {nec_gain:.1%}")
    if glcm_break:
        crit.append(f"GLCM contrast {diag.glcm_contrast_delta:+.1%} with homogeneity {diag.glcm_homog_delta:+.1%} "
                    "indicating surface breakdown")
    elif diag.glcm_contrast_delta > a["glcm_contrast_rise"] and diag.glcm_homog_delta < a["glcm_homogeneity_drop"] / 2:
        # one indicator alone is not a signal: re-encoding/noise inflates contrast with homogeneity unchanged
        warn.append(f"texture drift (contrast {diag.glcm_contrast_delta:+.1%}, homogeneity {diag.glcm_homog_delta:+.1%})")
    if (abs(diag.hue_rel_deg) < a["hue_warning_deg"] and abs(diag.hue_rel_deg) >= a.get("replaced_hue_deg", 10.0)
            and diag.diff_area_frac >= a.get("replaced_diff_frac", 0.4)):
        warn.append(f"sack appearance changed (hue {diag.hue_rel_deg:+.1f}°, {diag.diff_area_frac:.0%} of area differs) "
                    "- possibly swapped/replaced")
    if abs(diag.hue_rel_deg) >= a["hue_warning_deg"]:
        (crit if crit else warn).append(f"relative hue shifted {diag.hue_rel_deg:+.1f}°"
                                        f" (facility baseline {global_hue:+.1f}°)")
    if compress and settle >= a["slump_warning_px"]:
        how = (f"compressive flow (div {diag.flow_div:+.3f})" if diag.flow_div < 0
               else f"sack top dropped {diag.top_drop_px:.0f} px")
        msg = f"stack settled {settle:.1f} px downward; {how}; mean flow v_y {diag.flow_vy:+.1f} px"
        if a.get("slump_cross_view"):  # days matched across different viewpoints: parallax mimics settling
            warn.append(msg + " (camera viewpoint differs between days - verify on site)")
        else:
            (crit if settle >= a["slump_critical_px"] else warn).append(msg)

    if crit:
        diag.status, diag.reasons = CRITICAL, crit + warn
    elif warn:
        diag.status, diag.reasons = WARNING, warn
    else:
        diag.reasons = ["occupancy verified; colour and texture stable"]
    if diag.status != NORMAL:
        diag.reasons.append("occupancy verified")
    diag.action = ACTIONS[diag.status]
    return diag


def reasoning_text(slot: str, d: dict) -> str:
    return f"{slot}: Flagged {d['status']}. " + "; ".join(d["reasons"]) + "."
