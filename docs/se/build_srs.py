"""Builds docs/SRS_Extended_ColdStore_Vision.pdf.   python docs/se/build_srs.py"""
from __future__ import annotations

from common import (actor, arrow, box, bubble, figure, line, page, poly, store, svg, table, text, to_pdf, uml_class)

# ---- measured size (non-blank, non-comment Python lines, counted 2026-11-02) ------------------------
LOC = {"src/ (inspection pipeline)": 1902, "process.py + main.py (entry points)": 235,
       "tools/dataset (dataset building)": 1261, "tools/remote + models/train_yolo.py (training)": 419,
       "tools/synth (synthetic test data)": 311}
LOC_OTHER = {"tests/": 198, "video/ (explainer, not part of the product)": 878}
KLOC = sum(LOC.values()) / 1000

# ---- COCOMO --------------------------------------------------------------------------------------
BASIC = {"Organic": (2.4, 1.05, 0.38), "Semi-detached": (3.0, 1.12, 0.35), "Embedded": (3.6, 1.20, 0.32)}
basic_rows, basic = [], {}
for mode, (a, b, c) in BASIC.items():
    e = a * KLOC ** b
    t = 2.5 * e ** c
    basic[mode] = (e, t)
    basic_rows.append([mode, f"{a} &times; {KLOC:.3f}<sup>{b}</sup> = {e:.1f}", f"2.5 &times; {e:.1f}<sup>{c}</sup> = {t:.1f}", f"{e / t:.1f}"])
DRIVERS = [("RELY: required reliability", "High", 1.15, "A false alarm costs a manual stack check"),
           ("CPLX: product complexity", "High", 1.15, "Multi-stage computer vision with learned models"),
           ("AEXP: application experience", "Low", 1.13, "First cold-store vision system for the team"),
           ("MODP: modern practices", "High", 0.91, "Automated tests, config-driven thresholds"),
           ("TOOL: software tools", "High", 0.91, "OpenCV, ultralytics, Kaggle GPU, AI coding assistant")]
EAF = 1.0
for d in DRIVERS:
    EAF *= d[2]
E_INT = 3.2 * KLOC ** 1.05 * EAF
T_INT = 2.5 * E_INT ** 0.38

# ---- function points -----------------------------------------------------------------------------
FP_ROWS = [("External inputs", 5, 4, "Day 1 recording, Day 2 recording, Polycam export, CLI options and notes, YAML config"),
           ("External outputs", 7, 5, "Five inspection outputs, Day 1 baseline, run_info.json"),
           ("External inquiries", 3, 4, "Audit table filter, click-a-sack in digital twin, curtain slider"),
           ("Internal logical files", 3, 10, "Prepared day folders, inspection records, label dataset"),
           ("External interface files", 3, 7, "YOLO weights, CLIP and LoFTR models, Kaggle kernel results")]
UFP = sum(r[1] * r[2] for r in FP_ROWS)
GSC = [("Data communications", 1), ("Distributed processing", 1), ("Performance", 4), ("Heavily used configuration", 1),
       ("Transaction rate", 1), ("Online data entry", 1), ("End-user efficiency", 3), ("Online update", 0),
       ("Complex processing", 5), ("Reusability", 4), ("Installation ease", 3), ("Operational ease", 3),
       ("Multiple sites", 2), ("Facilitate change", 4)]
DI = sum(g[1] for g in GSC)
TCF = 0.65 + 0.01 * DI
FP = UFP * TCF

# ---- schedule: (id, name, duration, predecessors, optimistic, pessimistic) ---------------------------
ACT = [("A", "Requirements and spec analysis", 1, [], 0.5, 2),
       ("B", "Scrape and filter images", 1, ["A"], 0.5, 2),
       ("C", "Auto-label and review", 2, ["B"], 1, 4),
       ("D", "Train YOLO on Kaggle", 1, ["C"], 0.5, 3),
       ("E", "Core pipeline modules", 2, ["A"], 1, 3),
       ("F", "Report and video outputs", 1, ["E"], 0.5, 2),
       ("G", "Synthetic test suite", 1, ["E"], 0.5, 2),
       ("H", "Capture Day 1 and Day 2 footage", 2, ["A"], 2, 3),
       ("I", "Integrate, tune on real footage", 4, ["D", "F", "G", "H"], 2, 7),
       ("J", "Explainer video", 2, ["I"], 1, 3),
       ("K", "Guide and documents", 1, ["I"], 0.5, 2)]
ES, EF = {}, {}
for a in ACT:
    ES[a[0]] = max([EF[p] for p in a[3]], default=0)
    EF[a[0]] = ES[a[0]] + a[2]
END = max(EF.values())
LF, LS = {}, {}
for a in reversed(ACT):
    succ = [s[0] for s in ACT if a[0] in s[3]]
    LF[a[0]] = min([LS[s] for s in succ], default=END)
    LS[a[0]] = LF[a[0]] - a[2]
CRIT = [a[0] for a in ACT if LS[a[0]] == ES[a[0]]]
PERT = {a[0]: ((a[4] + 4 * a[2] + a[5]) / 6, ((a[5] - a[4]) / 6) ** 2) for a in ACT}
T_E = sum(PERT[c][0] for c in CRIT)
VAR = sum(PERT[c][1] for c in CRIT)


# ---- diagrams ------------------------------------------------------------------------------------
def dfd0() -> str:
    c = (380, 170)
    return svg(760, 340,
               box(30, 40, 150, 48, "Inspector"), box(30, 250, 150, 48, "Capture device", "(phone, LiDAR)"),
               box(580, 40, 150, 48, "Trained model and", "matcher weights"), box(580, 250, 150, 48, "Store manager"),
               bubble(*c, 105, 62, "0", "ColdStore-Vision", "Inspection System"),
               arrow(180, 70, 293, 135, "day paths, trims, notes", 28, -18),
               arrow(290, 200, 180, 118, "run log, errors", -44, 16),
               arrow(180, 272, 288, 200, "videos, depth, poses", 38, 26),
               arrow(580, 72, 468, 136, "model files", -32, -16),
               arrow(468, 205, 580, 262, "status reports, videos", 40, -22))


def dfd1() -> str:
    p = {"1": (110, 80), "2": (300, 80), "3": (490, 80), "4": (670, 80), "5": (670, 240), "6": (450, 240), "7": (230, 240)}
    r = (72, 38)
    return svg(760, 400,
               bubble(*p["1"], *r, "1 Prepare", "input"), bubble(*p["2"], *r, "2 Extract", "keyframes"),
               bubble(*p["3"], *r, "3 Pair and", "align days"), bubble(*p["4"], 72, 38, "4 Detect", "sacks"),
               bubble(*p["5"], *r, "5 Normalise", "and match"), bubble(*p["6"], *r, "6 Diagnose", "each sack"),
               bubble(*p["7"], *r, "7 Generate", "outputs"),
               arrow(182, 80, 228, 80, "day folder", 0, -8), arrow(372, 80, 418, 80, "frames", 0, -8),
               arrow(562, 80, 598, 80, "warped", 0, -8), arrow(670, 118, 670, 202, "sack masks", 36, 3),
               arrow(598, 240, 522, 240, "pairs", 0, -8), arrow(378, 240, 302, 240, "diagnoses", 0, -8),
               arrow(50, 10, 92, 44, "recordings, day paths", 66, -6), arrow(158, 240, 60, 240, "reports to manager", 2, -8),
               store(30, 345, 150, "D1 Prepared days"), store(205, 345, 150, "D2 Inspection"),
               store(380, 345, 150, "D3 Outputs"), store(555, 345, 170, "D4 Config, weights"),
               poly([(85, 118), (85, 345)]), poly([(445, 278), (445, 310), (280, 310), (280, 345)], "records", 365, 305),
               poly([(250, 278), (450, 345)]), poly([(640, 345), (660, 278)], "thresholds", 700, 315))


def dfd2_align() -> str:
    return svg(760, 215,
               bubble(95, 75, 70, 36, "3.1 Shortlist", "with CLIP"), bubble(285, 75, 70, 36, "3.2 Match", "with LoFTR"),
               bubble(475, 75, 70, 36, "3.3 Fit", "homography"), bubble(665, 75, 70, 36, "3.4 Check", "warp, inliers"),
               bubble(380, 170, 80, 34, "3.5 Build", "panoramas"),
               arrow(165, 75, 215, 75, "top 4", 0, -8), arrow(355, 75, 405, 75, "points", 0, -8),
               arrow(545, 75, 595, 75, "H", 0, -8), arrow(665, 111, 455, 158, "valid H", 22, -12),
               arrow(95, 10, 95, 39, "keyframes", 44, 0), arrow(708, 104, 745, 150, "H, to 4 and 5", -66, 22))


def dfd2_diag() -> str:
    return svg(760, 215,
               bubble(95, 75, 70, 36, "6.1 Align", "sack crop"), bubble(285, 75, 70, 36, "6.2 Measure", "colour, texture"),
               bubble(475, 75, 70, 36, "6.3 Measure", "motion"), bubble(665, 75, 70, 36, "6.4 Apply", "safety rules"),
               bubble(475, 170, 76, 34, "6.5 Classify", "status"), bubble(240, 170, 76, 34, "6.6 Majority", "vote"),
               arrow(165, 75, 215, 75, "crops", 0, -8), arrow(355, 75, 405, 75, "deltas", 0, -8),
               arrow(545, 75, 595, 75, "flow", 0, -8), arrow(665, 111, 545, 156, "verdict inputs", 36, -2),
               arrow(399, 170, 316, 170, "per view", 0, -8), arrow(164, 170, 60, 170, "sack status", 0, -8),
               arrow(95, 10, 95, 39, "matched pair", 48, 0))


def structure() -> str:
    leaves = ["extract_keyframes", "build_mosaic", "pair_keyframes_robust", "SackDetector.detect", "normalize_pair",
              "match_by_centroid", "diagnose_local", "index_bays", "compute_kpis"]
    parts = [box(20, 190, 110, 40, "process.main"), box(190, 60, 140, 34, "prepare_day"),
             box(190, 193, 140, 34, "run_inspection"), box(190, 340, 140, 34, "generate_outputs")]
    for y in (77, 210, 357):
        parts.append(poly([(130, 210), (160, 210), (160, y), (190, y)]))
    for i, n in enumerate(leaves):
        y = 15 + i * 36 + (0 if i < 9 else 0)
        parts.append(box(420, y + 12, 170, 28, n, size=11))
        parts.append(poly([(330, 210), (375, 210), (375, y + 26), (420, y + 26)]))
    # data couples on the right
    parts.append(text(600, 42, "frames, poses", 10, "start"))
    parts.append(text(600, 78, "panorama", 10, "start"))
    parts.append(text(600, 114, "H per pair", 10, "start"))
    parts.append(text(600, 150, "masks", 10, "start"))
    parts.append(text(600, 186, "Day 2 normalised", 10, "start"))
    parts.append(text(600, 222, "sack pairs", 10, "start"))
    parts.append(text(600, 258, "diagnosis", 10, "start"))
    parts.append(text(600, 294, "bay, tier", 10, "start"))
    parts.append(text(600, 330, "KPIs", 10, "start"))
    return svg(760, 385, *parts)


def usecase() -> str:
    u = [(380, 45, "Run inspection"), (380, 105, "Run Day 1 baseline"), (380, 165, "Trim recordings"),
         (380, 225, "View audit report"), (380, 285, "Explore digital twin"), (380, 345, "Compare panoramas"),
         (380, 405, "Train or update model")]
    parts = [f'<rect x="215" y="8" width="330" height="440" fill="none" stroke="#000"/>',
             text(380, 438, "ColdStore-Vision", 12, bold=True), actor(90, 50, "Inspector"),
             actor(90, 245, "Store manager"), actor(670, 340, "ML engineer"), actor(670, 60, "Capture device")]
    for x, y, n in u:
        parts.append(bubble(x, y, 120, 22, n))
    for (x, y) in ((260, 45), (260, 105), (260, 165)):
        parts.append(line(108, 90, x, y))
    for (x, y) in ((260, 225), (260, 285), (260, 345)):
        parts.append(line(108, 285, x, y))
    parts.append(line(652, 385, 500, 405))
    parts.append(line(652, 100, 500, 45))
    parts.append(line(652, 105, 500, 105))
    return svg(760, 460, *parts)


def classes() -> str:
    return svg(760, 590,
               uml_class(20, 15, 175, "FrameSource", ["fps: float", "n: int", "iter_frames()", "get(indices)"]),
               uml_class(20, 125, 175, "Keyframe", ["index: int", "t: float", "image", "pose, depth", "sharpness: float"]),
               uml_class(290, 15, 180, "Inspection", ["kf1, kf2: list", "mosaic1, mosaic2", "detector_source: str", "kpis: dict"]),
               uml_class(565, 15, 175, "Mosaic", ["image", "transforms: list", "valid"]),
               uml_class(290, 165, 180, "PairResult", ["i1, j2: int", "img1, img2n", "valid, flow", "global_hue: float"]),
               uml_class(565, 165, 175, "Registration", ["H", "inliers, matches: int", "ok: bool", "warp()", "valid_mask()"]),
               uml_class(20, 300, 175, "SackDetector", ["conf, iou, imgsz", "device", "detect(img)"]),
               uml_class(20, 440, 175, "Instance", ["cls: str", "conf: float", "box, mask, polygon", "area, centroid"]),
               uml_class(290, 315, 180, "Observation", ["kf, pair: int", "mosaic_box", "completeness: float", "global_id: int"]),
               uml_class(565, 330, 175, "SackRecord", ["sack_id, slot: str", "bay, tier: int", "status: str", "metrics: dict", "reasoning, action"]),
               uml_class(290, 470, 180, "SackDiagnosis", ["status: str", "diff_area_frac", "hue_rel_deg", "necrosis_d1, d2", "glcm contrast, homog.", "flow_vy, flow_div"]),
               arrow(107, 83, 107, 125, "creates", 26, 4, dash=True),
               line(290, 75, 195, 165, "2..*", -34, -14),
               line(470, 50, 565, 50, "2", 0, -6),
               line(380, 100, 380, 165, "1..*", 18, 3),
               line(470, 215, 565, 215, "1", -32, -6),
               arrow(107, 386, 107, 440, "returns", 28, 3, dash=True),
               line(290, 400, 195, 480, "1..2", 24, 14),
               line(380, 235, 380, 315, "0..*", 18, 3),
               line(470, 380, 565, 390, "1", 0, -6),
               line(380, 402, 380, 470, "0..1", 18, 3),
               poly([(470, 80), (520, 80), (520, 345), (565, 345)], "0..*", 540, 338))


def sequence() -> str:
    names = [("Inspector", 60), ("process.py", 170), ("pipeline", 280), ("matcher", 390), ("detector", 490), ("analytics", 590), ("reporting", 695)]
    parts = []
    for n, x in names:
        parts.append(box(x - 46, 8, 92, 28, n, size=11))
        parts.append(line(x, 36, x, 500, dash=True))
    msg = [(60, 170, "day paths, trims"), (170, 280, "run_inspection"), (280, 390, "pair keyframes"), (390, 280, "pairs + H"),
           (280, 490, "detect (each frame)"), (490, 280, "sack masks"), (280, 590, "normalise, match"), (590, 280, "matched sacks"),
           (280, 590, "diagnose_local (each)"), (590, 280, "SackDiagnosis"), (280, 280, "majority vote, bays"),
           (280, 170, "Inspection"), (170, 695, "generate_outputs"), (695, 170, "5 output files"), (170, 60, "run folder path")]
    for i, (a, b, m) in enumerate(msg):
        y = 62 + i * 29
        if a == b:
            parts.append(poly([(a, y - 7), (a + 30, y - 7), (a + 30, y + 7), (a + 2, y + 7)]))
            parts.append(text(a + 36, y + 3, m, 10, "start"))
        else:
            parts.append(arrow(a, y, b, y, m, 0, -4, size=10, dash=b < a))
    return svg(760, 510, *parts)


def network() -> str:
    pos = {"A": (60, 150), "B": (180, 55), "C": (295, 55), "D": (410, 55), "E": (180, 150), "F": (295, 125),
           "G": (295, 195), "H": (180, 265), "I": (530, 150), "J": (665, 95), "K": (665, 215)}
    w, h = 78, 44
    parts = []
    for a in ACT:
        for p in a[3]:
            x1, y1 = pos[p]
            x2, y2 = pos[a[0]]
            parts.append(arrow(x1 + w / 2, y1, x2 - w / 2, y2))
    for a in ACT:
        x, y = pos[a[0]]
        cr = a[0] in CRIT
        fill = "#d9d9d9" if cr else "#fff"
        parts.append(box(x - w / 2, y - h / 2, w, h, f"{a[0]}  ({a[2]}d)", f"{ES[a[0]]}-{EF[a[0]]} / {LS[a[0]]}-{LF[a[0]]}", size=11, fill=fill))
    return svg(760, 310, *parts)


def gantt() -> str:
    x0, sc, y0, rh = 215, 44, 30, 23
    parts = []
    for d in range(END + 1):
        x = x0 + d * sc
        parts.append(line(x, y0 - 8, x, y0 + rh * len(ACT)))
        parts.append(text(x, y0 - 12, str(d), 10))
    for i, a in enumerate(ACT):
        y = y0 + i * rh
        parts.append(text(8, y + 15, f"{a[0]}  {a[1]}", 11, "start"))
        fill = "#555" if a[0] in CRIT else "#bbb"
        parts.append(f'<rect x="{x0 + ES[a[0]] * sc}" y="{y + 4}" width="{a[2] * sc}" height="14" fill="{fill}" stroke="#000"/>')
        if LS[a[0]] > ES[a[0]]:
            parts.append(f'<rect x="{x0 + EF[a[0]] * sc}" y="{y + 8}" width="{(LS[a[0]] - ES[a[0]]) * sc}" height="6" fill="none" stroke="#000" stroke-dasharray="3 2"/>')
    return svg(760, y0 + rh * len(ACT) + 10, *parts)


# ---- document ------------------------------------------------------------------------------------
PH = '<mark>[to fill]</mark>'
FR = [
    ("FR-01 Input handling", [
        ("FR-01.1", "The system shall accept a video file (.mp4, .mov, .m4v, .avi), a Polycam Raw Data export (.zip or folder), or a folder of at least 5 images as the recording for each day.", "Implemented", "test_polycam_roundtrip; onion run"),
        ("FR-01.2", "When a folder holds both a video and a Polycam export, the system shall use the video unless the user passes --prefer-lidar.", "Implemented", "Code inspection"),
        ("FR-01.3", "The system shall accept start and end times in seconds to process part of a recording.", "Implemented", "Onion run (trim 9.8 s)"),
        ("FR-01.4", "If Day 2 is omitted, the system shall produce a Day 1 baseline report with the sack count and a recording quality check.", "Implemented", "outputs/day1_baseline"),
        ("FR-01.5", "If an input path does not exist, the system shall stop and print the path with the message \"input not found\".", "Implemented", "TC-B2"),
    ]),
    ("FR-02 Keyframe selection", [
        ("FR-02.1", "The system shall resize all frames to a width of 1280 pixels before analysis.", "Implemented", "test_keyframes_and_registration"),
        ("FR-02.2", "The system shall select a keyframe every 0.5 m of camera travel when poses exist, and every 0.5 s otherwise.", "Implemented", "test_keyframes_and_registration"),
        ("FR-02.3", "The system shall replace a keyframe whose Laplacian variance is below 100 with the sharpest frame within 5 frames either side.", "Implemented", "Code inspection, path testing (B.4)"),
    ]),
    ("FR-03 Alignment between days", [
        ("FR-03.1", "The system shall pair each Day 1 keyframe with the Day 2 keyframe showing the same scene, using a pair only when it has at least 40 inlier matches (LoFTR mode).", "Implemented", "test_keyframes_and_registration; onion run (10 pairs)"),
        ("FR-03.2", "The system shall compute a mapping that warps the Day 2 frame onto the Day 1 frame, and shall reject an implausible mapping.", "Implemented", "test_homography_recovers_known_warp"),
        ("FR-03.3", "The system shall build one panorama per day, aligned to each other pixel for pixel.", "Implemented", "test_mosaics_aligned"),
    ]),
    ("FR-04 Sack detection", [
        ("FR-04.1", "The system shall output an outline, box and confidence for each sack, rack beam and empty slot at a confidence of at least 0.25.", "Implemented", "test_sack_count_matches_day1"),
        ("FR-04.2", "If the trained model file is missing, the system shall log a warning and use the fallback detector.", "Implemented", "Code inspection"),
    ]),
    ("FR-05 Lighting compensation", [
        ("FR-05.1", "The system shall remap Day 2 colours so its histogram matches Day 1, excluding sacks that are missing or new.", "Implemented", "test_histogram_matching_undoes_global_lighting"),
        ("FR-05.2", "The system shall report each sack's hue change relative to the hue drift of the background.", "Implemented", "test_global_hue_shift_on_bimodal_background"),
    ]),
    ("FR-06 Sack matching", [
        ("FR-06.1", "The system shall assign Day 1 sacks to Day 2 sacks one to one, and reject a pair whose cost exceeds 0.6.", "Implemented", "test_hungarian_iou_matching"),
        ("FR-06.2", "The system shall re-align each matched pair locally before comparing them.", "Implemented", "Onion run"),
        ("FR-06.3", "When a sack appears in several keyframe pairs, the system shall report the majority status.", "Implemented", "Onion run"),
    ]),
    ("FR-07 Diagnosis", [
        ("FR-07.1", "The system shall mark a sack MISSING when at least 45% of its eroded area changes (20% if no Day 2 sack covers the spot) and no Day 2 sack matches it, or when LiDAR depth grows by more than 0.30 m.", "Implemented, with limit L1", "test_every_anomaly_detected...; test_missing_uses_depth"),
        ("FR-07.2", "The system shall mark a sack NEW when a Day 2 sack has no Day 1 match.", "Implemented", "test_every_anomaly_detected..."),
        ("FR-07.3", "The system shall mark a sack WARNING when its rot-coloured area grows by 3 percentage points, and CRITICAL at 8.", "Implemented", "test_necrosis_and_glcm_respond_to_rot"),
        ("FR-07.4", "The system shall mark a sack WARNING on relative hue change above 15 degrees, GLCM contrast rise above 35%, or homogeneity drop above 20%.", "Implemented", "test_necrosis_and_glcm_respond_to_rot"),
        ("FR-07.5", "The system shall mark a sack WARNING on settling above 4 px and CRITICAL above 10 px, capped at WARNING when the views differ.", "Implemented", "test_flow_detects_downward_shift"),
        ("FR-07.6", "The system shall flag a sack as possibly replaced when hue shifts by at least 10 degrees and at least 40% of its area differs.", "Implemented", "Onion run (Sack_011)"),
        ("FR-07.7", "The system shall mark a sack UNVERIFIED when fewer than 50% of the sacks in its view match, or its hue jumps by more than 90 degrees (object in front).", "Implemented", "Onion run (13 sacks)"),
        ("FR-07.8", "The system shall not mark a sack MISSING if it is cut by the left or right frame edge, or smaller than 1% of the frame.", "Implemented", "Code inspection"),
    ]),
    ("FR-08 Location indexing", [
        ("FR-08.1", "The system shall assign each sack a bay, tier and slot without markers or a bay configuration file.", "Implemented", "test_bay_indexer_uses_uprights_and_rows; test_bays_and_tiers"),
    ]),
    ("FR-09 Outputs", [
        ("FR-09.1", "The system shall write a Day 2 video with each sack outlined in its status colour.", "Implemented", "test_outputs_written"),
        ("FR-09.2", "The system shall write an audit report (HTML and JSON) listing, per sack: slot, status, measurements, reason and action.", "Implemented", "test_outputs_written"),
        ("FR-09.3", "The system shall write a side-by-side video of Day 1 and aligned Day 2 with the reason for each flag.", "Implemented", "test_outputs_written"),
        ("FR-09.4", "The system shall write a bay-by-tier digital twin page showing before and after crops for a clicked sack.", "Implemented", "test_outputs_written"),
        ("FR-09.5", "The system shall write a panorama page with a Day 1 / Day 2 curtain slider.", "Implemented", "test_outputs_written"),
        ("FR-09.6", "The system shall write run_info.json with inputs, trims, notes, keyframe counts, KPIs and run time.", "Implemented", "Onion run"),
    ]),
    ("FR-10 Model training support", [
        ("FR-10.1", "The system shall provide commands to scrape, filter, auto-label, split and validate a sack image dataset.", "Implemented", "Dataset v1.1 built"),
        ("FR-10.2", "The system shall provide a command that trains the detector on a cloud GPU and fetches the weights and metrics.", "Implemented", "Kaggle T4 run, 105 epochs"),
    ]),
]
FR_HTML = ""
for title, rows in FR:
    FR_HTML += f"<h3>{title}</h3>" + table(["ID", "Requirement", "Verified by"], [[r[0], r[1], r[3]] for r in rows], [9, 63, 28])

NFR = [
    ["NFR-P1", "Performance", "A 15-second recording pair shall complete within 20 minutes on a laptop CPU with no GPU.", "791 s (13.2 min) for the onion pair. About 20 min for full-length 4K walks.", "Met"],
    ["NFR-A1", "Accuracy", "Detector mask mAP@50 shall be at least 0.85 on the test set.", "0.797", "Not met"],
    ["NFR-A2", "Accuracy", "Detector mask precision shall be at least 0.80 on the test set.", "0.798", "Not met (marginal)"],
    ["NFR-A3", "Accuracy", "On the synthetic aisle, every planted anomaly shall be found with no false alarm.", "All found, no false alarm", "Met"],
    ["NFR-R1", "Reliability", "All automated tests shall pass before a release.", "16 of 16 passed on 2 Nov 2026", "Met"],
    ["NFR-R2", "Reliability", "A view that cannot be compared shall produce UNVERIFIED, never MISSING or CRITICAL.", "Enforced by the pair gate (FR-07.7)", "Met"],
    ["NFR-U1", "Usability", "An inspection shall need one command with at most two required paths.", "process.py --day1 X --day2 Y", "Met"],
    ["NFR-U2", "Usability", "Every flagged sack shall state its reason in a plain sentence.", "Reason column in the audit report", "Met"],
    ["NFR-M1", "Maintainability", "All thresholds shall be changeable in one YAML file without code changes.", "configs/pipeline_config.yaml", "Met"],
    ["NFR-M2", "Maintainability", "A new detector shall be installable by replacing one weights file.", "models/weights/best.pt", "Met"],
    ["NFR-O1", "Portability", "The system shall run on Windows 11 with Python 3.12 on CPU.", "All runs on this setup. Linux and macOS not tested.", "Met (Windows only)"],
    ["NFR-C1", "Compatibility", "HTML outputs shall open offline in a modern browser, as single files.", "Images embedded as base64 JPEG", "Met"],
    ["NFR-C2", "Compatibility", "Output videos shall play in a desktop media player.", "mp4v by default. Browser playback needs avc1 (openh264).", "Partly met"],
    ["NFR-S1", "Security", "Inspection footage and reports shall remain on the local machine.", "No inspection step uploads data. Only training images go to Kaggle.", "Met"],
    ["NFR-S2", "Security", "API tokens shall not be stored in the project folder.", "Kaggle token is in the user profile folder", "Met"],
    ["NFR-B1", "Backup", "Each run shall write to a new folder, so earlier results are not overwritten.", "outputs/run_&lt;timestamp&gt;/ by default", "Met"],
    ["NFR-L1", "Scalability", "The system shall handle at least 40 sacks in one frame.", "44 sacks separated in one Day 1 frame. Whole-aisle runs not tested.", "Met for one stack"],
]

TESTS = [
    ["test_homography_recovers_known_warp", "Unit", "Known warp", "Registration recovers the warp", "Pass"],
    ["test_histogram_matching_undoes_global_lighting", "Unit", "Globally darkened image", "Lighting difference removed", "Pass"],
    ["test_global_hue_shift_on_bimodal_background", "Unit", "Two-colour background, shifted hue", "Drift measured correctly", "Pass"],
    ["test_necrosis_and_glcm_respond_to_rot", "Unit", "Sack with rot patch", "Rot fraction and GLCM both change", "Pass"],
    ["test_erosion_shrinks_mask", "Unit", "Binary mask", "Eroded mask is smaller", "Pass"],
    ["test_flow_detects_downward_shift", "Unit", "Image shifted down", "Positive vertical flow", "Pass"],
    ["test_hungarian_iou_matching", "Unit", "Two sets of boxes", "Correct one-to-one pairs", "Pass"],
    ["test_bay_indexer_uses_uprights_and_rows", "Unit", "Beam and sack boxes", "Correct bay and tier", "Pass"],
    ["test_keyframes_and_registration", "Integration", "Synthetic aisle", "Keyframes paired", "Pass"],
    ["test_sack_count_matches_day1", "Integration", "Synthetic aisle", "Count equals known sacks", "Pass"],
    ["test_every_anomaly_detected_without_false_alarms", "System", "Planted anomalies", "All found, no false alarm", "Pass"],
    ["test_missing_uses_depth", "Integration", "Synthetic depth", "Depth drives MISSING", "Pass"],
    ["test_bays_and_tiers", "Integration", "Synthetic aisle", "Correct layout", "Pass"],
    ["test_mosaics_aligned", "Integration", "Two days", "Panoramas line up", "Pass"],
    ["test_outputs_written", "System", "Inspection result", "All five outputs exist", "Pass"],
    ["test_polycam_roundtrip", "Integration", "Synthetic Polycam export", "Frames, poses, depth preserved", "Pass"],
]

SYLLABUS = [
    ["1", "Software and its characteristics; components; role of software engineer", "B.1"],
    ["1", "Software crisis; emergence of SE; comparison with conventional engineering", "B.1"],
    ["1", "Quality attributes", "B.1, A.3.2"],
    ["1", "Waterfall, iterative waterfall, prototyping, evolutionary, spiral, RAD", "B.1"],
    ["1", "Agile: essential idea, Agile versus others, XP, Scrum", "B.1"],
    ["2", "Requirements gathering and analysis (anomaly, inconsistency, incompleteness)", "A.1.6, A.1.7"],
    ["2", "SRS document and IEEE 830", "Part A"],
    ["2", "Design process; good design; cohesion and coupling", "B.2"],
    ["2", "Function-oriented design: DFD levels, structural analysis, structure chart", "B.2"],
    ["2", "Object-oriented design: concepts; use case, class, sequence diagrams", "B.2, A.2.6"],
    ["3", "Project management complexity; project planning", "B.3"],
    ["3", "Size metrics: LOC, function points", "B.3"],
    ["3", "Empirical estimation: expert judgement, Delphi", "B.3"],
    ["3", "Heuristic estimation: COCOMO", "B.3"],
    ["3", "WBS, activity network, CPM, PERT, Gantt", "B.3"],
    ["3", "Organisation and team structure; risk management", "B.3"],
    ["4", "Code review", "B.4"],
    ["4", "Testing basics, objectives, verification and validation, testing activity", "B.4"],
    ["4", "Unit, functional and performance testing; top-down and bottom-up; drivers and stubs", "B.4"],
    ["4", "Black box, white box, integration, system, acceptance, regression testing", "B.4"],
    ["5", "SQA plan; ISO 9000; SEI-CMM", "B.5"],
    ["5", "Software reliability", "B.5"],
    ["5", "Maintenance; reverse engineering; re-engineering; reuse", "B.5"],
    ["Emerging", "Software as a Service; AI-based software engineering", "B.6"],
]

body = f"""
<div class="cover">
<p><b>KALINGA INSTITUTE OF INDUSTRIAL TECHNOLOGY</b><br>School of Computer Engineering</p>
<p style="margin-top:22mm">Software Engineering</p>
<h1>Software Requirements Specification (Extended)</h1>
<h1>ColdStore-Vision</h1>
<p>Automated Cold-Storage Sack Inspection from Video</p>
<p style="margin-top:22mm"><b>Submitted by</b><br><mark>[Name] - [Roll number]</mark><br><mark>[Name] - [Roll number]</mark><br><mark>[Name] - [Roll number]</mark></p>
<p><b>Group:</b> <mark>[Group name]</mark> &nbsp; <b>Section:</b> <mark>[Section]</mark></p>
<p><b>Submitted to:</b> <mark>[Faculty name]</mark></p>
<p>Version 1.0 &middot; 2 November 2026</p>
</div>

<h1>How this document is organised</h1>
<p>Part A is the SRS, in IEEE 830 order. It states what the system must do and treats the system as a black box. Part B applies the rest of the syllabus to this project; designs and plans are kept there because they do not belong in an SRS. Part C maps each syllabus topic to its section.</p>
<ul class="toc">
<li><b>Part A: Software Requirements Specification</b>
<ul><li>A.1 Introduction</li><li>A.2 Overall description</li><li>A.3 Specific requirements</li><li>A.4 Constraints, assumptions, dependencies</li><li>A.5 Validation of this SRS</li></ul></li>
<li><b>Part B: Syllabus extension</b>
<ul><li>B.1 Software engineering basics and process model</li><li>B.2 Software design</li><li>B.3 Project management</li><li>B.4 Coding and testing</li><li>B.5 Quality, reliability and maintenance</li><li>B.6 Emerging topics</li></ul></li>
<li><b>Part C: Syllabus traceability</b></li>
</ul>

<div class="part"><h1>Part A: Software Requirements Specification</h1></div>
<h2>A.1 Introduction</h2>
<h3>A.1.1 Purpose</h3>
<p>This document specifies the requirements of ColdStore-Vision, a software system that compares two video walkthroughs of the same cold-storage stack and reports the condition of each sack. It is written for the project team, the course evaluators, testers and any future customer.</p>
<h3>A.1.2 Scope</h3>
<p>Cold stores hold thousands of onion, potato and garlic sacks. Checking them by hand for rot, settling or missing stock is slow and irregular. ColdStore-Vision takes a Day 1 recording and a Day 2 recording, and gives every sack one status with a written reason.</p>
<p><b>In scope:</b> offline processing of recordings; detection, matching and diagnosis of sacks; reports and videos; tools to train the detector.</p>
<p><b>Out of scope in this version:</b> the inspection drone (concept only), live video streams, integration with stock or dispatch software, automatic stack-level stock counts, and a web service.</p>
<h3>A.1.3 Definitions</h3>
{table(["Term", "Meaning"], [
    ["Sack", "One jute or mesh bag of produce."],
    ["Bay, tier, slot", "Horizontal section of a rack, its row counted from the bottom, and a position within the row."],
    ["Keyframe", "A sharp frame chosen from the video for analysis."],
    ["Homography", "An 8-parameter mapping between two views of a flat surface."],
    ["Parallax", "Near objects shifting more than far ones when the camera moves."],
    ["GLCM", "Grey-level co-occurrence matrix, a measure of texture."],
    ["mAP@50", "Mean average precision of the detector at 50% overlap."],
    ["UNVERIFIED", "Status meaning the system could not compare the sack reliably."],
], [22, 78])}
<h3>A.1.4 References</h3>
<ul><li>IEEE Std 830-1998, Recommended Practice for Software Requirements Specifications.</li>
<li>R. Mall, <i>Fundamentals of Software Engineering</i>, PHI Learning.</li>
<li>ColdStore-Vision project guide (GUIDE.md) and master specification.</li></ul>
<h3>A.1.5 Stakeholders</h3>
{table(["Stakeholder", "Need from the system"], [
    ["Cold-store manager", "A quick view of which sacks need attention, and where they are"],
    ["Inspector or operator", "A simple way to record and run an inspection"],
    ["Dispatch and inward staff", "Missing and new sacks to reconcile against their registers"],
    ["ML engineer or developer", "A way to retrain the detector and change thresholds"],
    ["Course evaluators", "Evidence of the method and honest results"],
], [30, 70])}
<h3>A.1.6 Requirements gathering</h3>
{table(["Activity", "How it was applied"], [
    ["Study of existing documents", "Read the project's master specification, then the documentation of Polycam exports and YOLO dataset formats."],
    ["Interview", "Discussions with the project owner fixed the key decisions: no site markers, train on Kaggle, compare onion stack only."],
    ["Observation", "Two visits to a real godown on 21 and 22 October 2026 to record it. This showed dim light, pillars and touching sacks."],
    ["Task analysis", "Inspection was split into: record, align, find sacks, match, diagnose, report."],
    ["Scenario analysis", "For one sack: unchanged, removed, swapped, rotting, settled, hidden by a pillar, cut by the frame, not filmed on Day 2."],
    ["Form analysis", "Output fields were derived from what staff need: the inward register, the dispatch log and an action per sack."],
], [26, 74])}
<h3>A.1.7 Requirements analysis: problems found and resolved</h3>
{table(["Type", "Problem", "Resolution"], [
    ["Anomaly", "\"Report a sack as missing\" has several meanings. A sack hidden by a pillar also looks missing.", "Measurable rule in FR-07.1, and a separate UNVERIFIED status in FR-07.7."],
    ["Anomaly", "\"Sack has changed colour\" had no threshold.", "Relative hue above 15 degrees, measured after lighting compensation (FR-07.4, FR-05)."],
    ["Inconsistency", "The master spec required training to finish first. The owner wanted all modules built in parallel.", "Owner's decision adopted. A fallback detector and a synthetic test detector allowed parallel work."],
    ["Inconsistency", "The spec preferred LiDAR capture, but the actual LiDAR recording was a room scan, not a rack pass.", "Video is the default. LiDAR is optional through --prefer-lidar (FR-01.2)."],
    ["Incompleteness", "No status existed for sacks the system could not compare.", "UNVERIFIED added (FR-07.7)."],
    ["Incompleteness", "A swapped sack was not covered by the original statuses.", "\"Possibly replaced\" rule added (FR-07.6)."],
    ["Incompleteness", "Nothing said how to treat sacks cut off at the frame edge.", "Truncation rule added (FR-07.8)."],
], [15, 45, 40])}

<h2>A.2 Overall description</h2>
<h3>A.2.1 Product perspective</h3>
<p>ColdStore-Vision is a self-contained command-line tool. It depends on a phone camera for input and on Kaggle only for training. It does not form part of a larger system today.</p>
<h3>A.2.2 Product functions</h3>
<ol><li>Read recordings in several formats.</li><li>Choose sharp keyframes.</li><li>Align the two days and build panoramas.</li><li>Outline every sack.</li><li>Compensate for lighting.</li><li>Match each sack across days.</li><li>Diagnose each sack and give a reason.</li><li>Locate it by bay, tier and slot.</li><li>Write five outputs.</li></ol>
<h3>A.2.3 User characteristics</h3>
{table(["User", "Technical skill", "Frequency of use"], [
    ["Inspector", "Can use a phone and copy a command; no vision knowledge", "Each inspection"],
    ["Store manager", "Can open a web page; no command-line use", "After each inspection"],
    ["ML engineer", "Python and machine learning", "When retraining or tuning"],
], [22, 53, 25])}
<h3>A.2.4 Operating environment</h3>
<p>Windows 11, Python 3.12, CPU only, at least 8 GB of RAM and about 1 GB of disk for models. A modern browser opens the reports. Training uses a Kaggle NVIDIA T4 GPU.</p>
<h3>A.2.5 External interfaces</h3>
{table(["Interface", "Description"], [
    ["User interface", "Command line for running; offline HTML pages and MP4 videos for results."],
    ["Input files", "Video (.mp4, .mov, .m4v, .avi); Polycam zip with frames, camera JSON and depth; JPEG or PNG images."],
    ["Config file", "YAML: configs/pipeline_config.yaml."],
    ["Output files", "MP4, HTML, JSON, JPEG."],
    ["Hardware", "Any phone recording at 1080p or 4K, 30 fps. LiDAR is optional."],
    ["Software", "OpenCV, PyTorch, ultralytics, kornia, OpenCLIP, SciPy, scikit-image; Kaggle CLI for training."],
], [20, 80])}
<h3>A.2.6 Use cases</h3>
{figure(usecase(), "Figure 1: Use case diagram")}
{table(["Field", "UC-1: Run inspection"], [
    ["Actor", "Inspector"],
    ["Precondition", "Two recordings of the same stack exist; the model file is installed."],
    ["Main flow", "1. Inspector runs process.py with the two paths. 2. System processes both. 3. System writes the outputs and prints the folder."],
    ["Alternative 1", "Day 2 omitted: system writes a Day 1 baseline."],
    ["Alternative 2", "Views too different: affected sacks are UNVERIFIED and the inspection still finishes."],
    ["Exception", "Path does not exist: system prints \"input not found\" and stops."],
    ["Postcondition", "A new run folder holds all outputs."],
], [20, 80])}

<h2>A.3 Specific requirements</h2>
<h3>A.3.1 Functional requirements</h3>
<p>All requirements below are implemented. <b>Known limit L1 (FR-07.1):</b> when a removed sack is overlapped by a neighbour, the synthetic real-photo test gives WARNING instead of MISSING.</p>
{FR_HTML}
<h3>A.3.2 Non-functional requirements</h3>
{table(["ID", "Attribute", "Requirement", "Measured result", "Status"], NFR, [8, 12, 38, 28, 14])}
<p><b>Availability:</b> a figure such as 99.9% uptime does not apply, because this version is a batch tool, not a running service. It would apply to the SaaS form in B.6.</p>

<h2>A.4 Constraints, assumptions and dependencies</h2>
<h3>Constraints</h3>
<ul><li>No GPU on the inspection machine, so processing is CPU based.</li>
<li>Free cloud GPU quota limits training runs.</li>
<li>No site markers or bay configuration may be required.</li>
<li>Training images could only come from the web at the start, so the first model was trained without cold-store frames.</li></ul>
<h3>Assumptions</h3>
<ul><li>Both recordings cover the same stack, walking at about 0.5 m/s.</li>
<li>Sacks are visible from the aisle; sacks behind the front layer cannot be assessed.</li>
<li>Most of the frame does not change between days, so lighting can be matched across the frame.</li>
<li>Visual change is a useful sign of spoilage. Rot inside a sack with no surface change cannot be seen.</li></ul>
<h3>Dependencies</h3>
<ul><li>Python packages listed in requirements.txt.</li><li>Model downloads on first run (OpenCLIP, LoFTR).</li><li>Kaggle service for training.</li></ul>

<h2>A.5 Validation of this SRS</h2>
{table(["Property of a good SRS", "How this document meets it"], [
    ["Concise and unambiguous", "\"Shall\" statements with numeric thresholds, such as 15 degrees or 0.30 m."],
    ["States what, not how", "Part A describes behaviour. Algorithms and designs are in Part B. Thresholds are observable behaviour."],
    ["Easy to change", "Numbered groups; thresholds also sit in one config file."],
    ["Consistent", "The conflicts in A.1.7 were resolved with the project owner."],
    ["Complete", "Scenario analysis covered each sack condition. Known limit L1 is stated."],
    ["Traceable", "Every FR names a test or a run. Part C links the syllabus to sections."],
    ["Verifiable", "\"User-friendly\" was replaced by NFR-U1 and NFR-U2, which can be checked."],
], [28, 72])}

<div class="part"><h1>Part B: Syllabus Extension</h1></div>
<h2>B.1 Software engineering basics and process model</h2>
<h3>Software, its characteristics and components</h3>
<p>Software differs from a manufactured product. It is developed rather than built on a line, it does not wear out, and it can be changed after delivery. It also has more parts than a program does:</p>
{table(["Component", "In this project"], [
    ["Programs", "src/, process.py, tools/"],
    ["Data", "Dataset v1.1 (152 images, 2,782 sack outlines), trained weights"],
    ["Documents", "README, GUIDE.md, this SRS, project report"],
    ["Operating procedures", "Recording guidelines, retraining steps"],
], [28, 72])}
<p><b>Role of a software engineer:</b> understanding requirements, choosing techniques, testing, estimating, managing risk and documenting, not only coding.</p>
<h3>Software crisis and emergence of software engineering</h3>
<p>The software crisis described projects that ran late, over budget and unreliable as size grew past what one person could hold in mind. Engineering discipline arose in response, through methods, abstraction and measured quality.</p>
<h3>Similarities to and differences from conventional engineering</h3>
{table(["Aspect", "Conventional engineering", "This project"], [
    ["Shared", "Requirements, design, testing, quality targets", "The same phases apply"],
    ["Main cost", "Manufacturing", "Development; copies are free"],
    ["Failure", "Parts wear out", "Faults are present from the start, or appear when the input changes, such as a new viewpoint"],
    ["Change", "Costly after build", "Cheap to edit, but side effects are hard to see"],
    ["Standard parts", "Catalogue components", "Libraries such as OpenCV and ultralytics"],
], [18, 36, 46])}
<h3>Quality attributes</h3>
<p>Those that matter most here are correctness (NFR-A), reliability (NFR-R), usability (NFR-U), maintainability (NFR-M), portability (NFR-O) and efficiency (NFR-P). See A.3.2 for measured values.</p>
<h3>Process models compared</h3>
{table(["Model", "Essential idea", "Fit for this project"], [
    ["Classical waterfall", "Sequential phases with no going back", "Poor. Requirements changed after seeing real footage."],
    ["Iterative waterfall", "Waterfall with feedback to earlier phases", "Partial. Feedback was needed, but also early working versions."],
    ["Prototyping", "Build a throwaway model to clarify requirements", "Useful early. The synthetic aisle and fallback detector acted as a prototype."],
    ["Evolutionary", "Deliver a core, then grow it in increments", "<b>Best fit, and the model followed.</b>"],
    ["Spiral", "Risk-driven loops of plan, evaluate, build", "Its risk focus was used, but the full model is heavy for a small team."],
    ["RAD", "Fast construction with reusable parts and user feedback", "Partly matched through library reuse, but needs a customer on hand and a time-boxed team."],
    ["Agile", "Short iterations, working software, changing requirements welcome", "Close in spirit, but no formal sprints were held."],
], [20, 45, 35])}
<h3>Evolutionary increments followed</h3>
<ol><li><b>Core:</b> all modules, tested on synthetic data with a test detector.</li>
<li><b>Detector:</b> dataset v0, then v1.1, trained model installed.</li>
<li><b>Real footage:</b> Day 1 baseline, then Day 2. SIFT pairing failed across days, so LoFTR replaced it.</li>
<li><b>Safety rules:</b> occlusion, truncation, majority vote and replaced-sack rules, each driven by a false result seen in real footage.</li>
<li><b>Planned:</b> fine-tune model v2 on cold-store frames.</li></ol>
<h3>Agile, XP and Scrum</h3>
<p><b>Agile versus other models:</b> plan-driven models fix requirements early and measure progress by documents. Agile expects change and measures progress by working software. <b>Extreme Programming</b> pushes good practices to the limit: pair programming, test-driven work, continuous integration, simple design. Here, automated tests and simple design were used; working with an AI assistant resembled pairing. <b>Scrum</b> organises work into sprints with a backlog, daily stand-up and review, using a product owner and Scrum master. It was not used formally, but it would suit the next phase, since a drone-based product will keep changing requirements.</p>

<h2>B.2 Software design</h2>
<h3>Design process and what makes a good design</h3>
<p>Design turns the SRS into modules, their interfaces, data structures and algorithms. A good design is correct, understandable, efficient and easy to maintain. The design rests on layering (ingestion, registration, segmentation, analytics, reporting) and one configuration file.</p>
<h3>Cohesion and coupling</h3>
<p>Good modules have high cohesion and low coupling.</p>
{table(["Module", "Cohesion", "Reason"], [
    ["optical_flow.py", "Functional", "Every function serves one task: measure motion"],
    ["lighting_normalizer.py", "Functional", "All of it serves lighting compensation"],
    ["anomaly_engine.py", "Functional / sequential", "Measurements flow into one diagnosis"],
    ["pipeline.py", "Sequential / communicational", "Steps pass results along and share the Inspection data"],
    ["dashboard_builder.py", "Communicational", "Different reports built from the same inspection data"],
    ["tools/dataset/common.py", "Logical", "Shared paths and CSV helpers; acceptable for utilities"],
], [30, 25, 45])}
{table(["Module pair", "Coupling", "Reason"], [
    ["pipeline to SackDetector", "Data", "Passes an image, gets a list of Instance. Replacing the model touches nothing else."],
    ["tracker to anomaly_engine", "None direct", "pipeline.py passes results between them"],
    ["All modules to config", "Stamped", "They receive the config dictionary but read only their own section; a small weakness accepted for simplicity"],
    ["reporting to pipeline", "Stamped", "Reporting reads the Inspection structure, so changes to it may affect reports"],
], [30, 15, 55])}
<h3>Function-oriented design: DFD</h3>
{figure(dfd0(), "Figure 2: Level 0 DFD (context diagram)")}
{figure(dfd1(), "Figure 3: Level 1 DFD")}
{figure(dfd2_align(), "Figure 4: Level 2 DFD, process 3 (pair and align days)")}
{figure(dfd2_diag(), "Figure 5: Level 2 DFD, process 6 (diagnose each sack)")}
<h3>Data dictionary</h3>
{table(["Data item", "Definition"], [
    ["recording", "video file | Polycam export | image folder"],
    ["keyframe", "index + time + image + [pose] + [depth] + sharpness"],
    ["H", "3 &times; 3 homography matrix + inlier count + ok flag"],
    ["sack mask", "class + confidence + box + binary mask + polygon"],
    ["diagnosis", "status + hue + rot fraction + GLCM + flow + reason + action"],
    ["status", "NORMAL | WARNING | CRITICAL | MISSING | NEW | UNVERIFIED"],
    ["reports", "overlay video + audit report + side-by-side video + digital twin + panorama slider"],
], [22, 78])}
<h3>Structural design</h3>
<p>Transform analysis applied to the Level 1 DFD gives this structure chart: input handling, the central transform in run_inspection, and output generation. It reads left to right, with the data returned to run_inspection written beside each module.</p>
{figure(structure(), "Figure 6: Structure chart")}
<h3>Object-oriented design</h3>
<p><b>Concepts used:</b> <i>abstraction and encapsulation</i> in SackDetector, which hides the choice of YOLO or YOLOE; <i>composition</i> in Inspection, which holds keyframes, pairs and sack records; <i>polymorphism</i> by duck typing, since SyntheticColorDetector and SackDetector offer the same detect() method. Inheritance is not used, as data classes and composition were sufficient.</p>
{figure(classes(), "Figure 7: Class diagram (main fields shown)")}
{figure(sequence(), "Figure 8: Sequence diagram for Run inspection")}

<h2>B.3 Project management</h2>
<h3>Why software projects are complex to manage</h3>
<p>Software is invisible, requirements change, and each project differs. Here the main uncertainties were real footage quality, cloud GPU quota and whether vision methods would survive the change between days.</p>
<h3>Project planning</h3>
<p>Planning covers size, effort, schedule, staffing, risks and quality. The estimates below use the measured code size.</p>
<h3>Size: lines of code</h3>
{table(["Part", "LOC"], [[k, v] for k, v in LOC.items()] + [[f"<b>Delivered product</b>", f"<b>{sum(LOC.values())}</b>"]] + [[f"Not counted: {k}", v] for k, v in LOC_OTHER.items()], [80, 20])}
<p>Counted as non-blank, non-comment Python lines.</p>
<h3>Size: function points</h3>
{table(["Type", "Count", "Weight (average)", "Points", "Items"], [[r[0], r[1], r[2], r[1] * r[2], r[3]] for r in FP_ROWS] + [["<b>UFP</b>", "", "", f"<b>{UFP}</b>", ""]], [20, 7, 10, 8, 55])}
<p>The 14 general system characteristics, rated 0 to 5: {", ".join(f"{g[0]} {g[1]}" for g in GSC)}. Their sum is DI = {DI}.</p>
<p class="formula">TCF = 0.65 + 0.01 &times; {DI} = {TCF:.2f} &nbsp;&nbsp; FP = {UFP} &times; {TCF:.2f} = {FP:.1f}</p>
<h3>Empirical estimation: expert judgement and Delphi</h3>
<p>In <b>expert judgement</b>, one expert estimates from experience; the risk is personal bias. In <b>Delphi</b>, several experts estimate anonymously through a coordinator, see a summary and revise, repeating until the numbers settle. The table shows the method using <mark>illustrative figures, not an actual survey; replace them if your group runs the exercise.</mark></p>
{table(["Estimator", "Round 1 (KLOC)", "Round 2 (KLOC)", "Round 3 (KLOC)"], [["A", 2.5, 3.5, 4.0], ["B", 6.0, 5.0, 4.5], ["C", 4.0, 4.0, 4.0], ["Mean", 4.2, 4.2, 4.2]], [25, 25, 25, 25])}
<h3>Heuristic estimation: COCOMO</h3>
<p><b>Basic COCOMO</b>, with size = {KLOC:.3f} KLOC. Effort E = a &times; KLOC<sup>b</sup> person-months, time T = 2.5 &times; E<sup>c</sup> months.</p>
{table(["Mode", "Effort (PM)", "Time (months)", "Avg. staff"], basic_rows, [20, 35, 33, 12])}
<p>Organic mode applies: small size, small team, and limited outside constraints. <b>Intermediate COCOMO</b> adjusts it with cost drivers (others nominal at 1.00):</p>
{table(["Driver", "Rating", "Multiplier", "Reason"], [[d[0], d[1], d[2], d[3]] for d in DRIVERS], [33, 9, 11, 47])}
<p class="formula">EAF = {EAF:.3f} &nbsp;&nbsp; E = 3.2 &times; {KLOC:.3f}<sup>1.05</sup> &times; {EAF:.3f} = {E_INT:.1f} PM &nbsp;&nbsp; T = 2.5 &times; {E_INT:.1f}<sup>0.38</sup> = {T_INT:.1f} months</p>
<p><b>Comparison with actual:</b> the working system took about one week of elapsed time (19 to 26 October 2026), far below {T_INT:.1f} months. COCOMO does not account for an AI coding assistant or heavy library reuse, so it reflects a conventional team.</p>
<h3>Work breakdown structure</h3>
<ol><li><b>Requirements:</b> study spec; decide scope; write requirements.</li>
<li><b>Dataset:</b> scrape; filter and deduplicate; auto-label; review and fix; split and validate.</li>
<li><b>Model:</b> set up Kaggle; train; evaluate; install weights.</li>
<li><b>Pipeline:</b> ingestion; registration; segmentation wrapper; lighting; matching; diagnosis; bay index.</li>
<li><b>Outputs:</b> videos; audit report; digital twin; panorama slider.</li>
<li><b>Testing:</b> synthetic data; unit tests; integration and system tests.</li>
<li><b>Field work:</b> record Day 1; record Day 2; run, tune and re-run.</li>
<li><b>Documentation:</b> guide; explainer video; SRS; report.</li></ol>
<h3>Activity network and critical path (CPM)</h3>
<p>Durations are ideal working days, in a re-baselined plan. Each node shows activity (duration) and ES-EF / LS-LF. Shaded nodes are critical.</p>
{figure(network(), "Figure 9: Activity network")}
{table(["Activity", "Name", "Dur.", "Pred.", "ES", "EF", "LS", "LF", "Slack"], [[a[0], a[1], a[2], ", ".join(a[3]) or "-", ES[a[0]], EF[a[0]], LS[a[0]], LF[a[0]], LS[a[0]] - ES[a[0]]] for a in ACT], [8, 42, 6, 11, 6, 6, 7, 7, 7])}
<p><b>Critical path: {" - ".join(CRIT)}, length {END} days.</b></p>
<h3>PERT</h3>
<p>Expected time t = (o + 4m + p) / 6 and variance = ((p - o) / 6)<sup>2</sup>. Values for o and p are the team's judgement.</p>
{table(["Activity", "o", "m", "p", "t", "Variance"], [[a[0], a[4], a[2], a[5], f"{PERT[a[0]][0]:.2f}", f"{PERT[a[0]][1]:.3f}"] for a in ACT], [20, 16, 16, 16, 16, 16])}
<p>On the critical path: expected length = {T_E:.2f} days, variance = {VAR:.3f}, standard deviation = {VAR ** 0.5:.2f} days. There is about an 84% chance of finishing within {T_E + VAR ** 0.5:.1f} days (mean plus one standard deviation).</p>
<h3>Gantt chart</h3>
{figure(gantt(), "Figure 10: Gantt chart in project days. Dark bars are critical; dashed bars show slack.")}
<h3>Actual milestones</h3>
{table(["Date (2026)", "Milestone"], [
    ["19 Oct", "Scope decisions; dataset work started"],
    ["20 Oct", "Model v1 trained on Kaggle T4 (105 epochs)"],
    ["21 Oct", "Day 1 recorded; baseline run"],
    ["22 Oct", "Day 2 recorded"],
    ["26 Oct", "Onion comparison delivered; explainer video and guide completed"],
    ["2 Nov", "SRS and project report completed; all 16 tests re-run"],
], [20, 80])}
<h3>Organisation and team structure</h3>
<p>Common structures are <b>chief programmer</b> (one senior leads, suits small and simple work), <b>democratic</b> (decisions shared, suits hard problems but small groups only) and <b>mixed</b> (for large projects). A democratic team suits this project because the hard problems benefit from discussion. <mark>Fill in who did what:</mark></p>
{table(["Role", "Responsibility", "Member"], [
    ["Project lead and requirements", "Scope, SRS, site visits", PH], ["ML engineer", "Dataset and model training", PH],
    ["Vision developer", "Pipeline modules", PH], ["Test and documentation", "Tests, report, video", PH],
], [32, 48, 20])}
<h3>Risk management</h3>
<p>Risks are identified, assessed (probability &times; impact on a 1 to 5 scale), then avoided, transferred or reduced. The risks below actually arose, except where the last column says otherwise.</p>
{table(["Risk", "P", "I", "Exposure", "Action taken", "Outcome"], [
    ["Footage from different angles on each day", 5, 5, 25, "Stronger matcher; UNVERIFIED status; recording guide", "Occurred. 13 of 22 sacks unverified"],
    ["Web images differ from godown frames", 4, 4, 16, "Grey and blur augmentation; v2 fine-tune planned", "Occurred. Slightly under accuracy target"],
    ["SIFT cannot pair days", 4, 5, 20, "Fallback: CLIP + LoFTR", "Occurred and resolved"],
    ["False alarms from lighting or a pillar", 4, 4, 16, "Histogram matching; occlusion rule", "Occurred and reduced"],
    ["Cloud training ran on CPU unnoticed", 3, 4, 12, "Force T4 accelerator in the script", "Occurred and resolved"],
    ["Model training blocks other work", 3, 3, 9, "Build in parallel with a fallback and test detector", "Avoided"],
    ["Manual labelling takes too long", 4, 3, 12, "Agent review with SAM 2.1 fixes", "Avoided"],
    ["New threshold breaks earlier behaviour", 3, 4, 12, "Regression tests", "Occurred and caught"],
    ["Data loss", 2, 4, 8, "New folder per run; no version control backup yet", "Open: add a git repository"],
], [28, 4, 4, 8, 30, 26])}

<h2>B.4 Coding and testing</h2>
<h3>Coding standards and code review</h3>
<p>The code follows Python conventions: modules per task, typed function signatures, docstrings explaining why, and thresholds in config. <b>Code review</b> comes in two forms: a <i>walkthrough</i>, where members trace code by hand, and an <i>inspection</i>, which checks against common errors. Reading the code against its imports while writing the guide found five wrong library entries in the draft documentation. A formal, recorded inspection has not been held.</p>
<h3>Basic concepts</h3>
{table(["Term", "Meaning", "Example here"], [
    ["Error / fault", "A mistake made in code", "No rule for an object blocking the view of a sack"],
    ["Failure", "Observable wrong behaviour", "Sacks behind a pillar reported MISSING"],
    ["Test case", "Input, conditions and expected result", "See the tables below"],
    ["Test suite", "A set of test cases", "tests/ (16 cases)"],
], [18, 37, 45])}
<p><b>Objective of testing:</b> to find faults, not to prove there are none. <b>Verification</b> asks \"are we building the product right?\" and checks each phase's output: reviews, unit tests. <b>Validation</b> asks \"are we building the right product?\" and checks the final product against needs: the real godown run. <b>Testing activities:</b> plan, design cases, execute, check results, debug, then re-test.</p>
<h3>Unit, integration and system test results</h3>
<p>Run with pytest on 2 November 2026: <b>16 passed in 93.56 s.</b></p>
{table(["Test case", "Level", "Input", "Expected", "Result"], TESTS, [42, 11, 19, 21, 7])}
<h3>Test drivers and stubs</h3>
<p><b>Driver:</b> pytest with the fixtures in tests/conftest.py, which call modules and check results. <b>Stub:</b> SyntheticColorDetector stands in for the YOLO model. It offers the same detect() method, finding sacks by colour in synthetic images, which let the pipeline be tested before any model existed.</p>
<h3>Top-down and bottom-up testing</h3>
<p><b>Bottom-up</b> tests small modules first with drivers; <b>top-down</b> tests the main control flow first with stubs. This project used both, which is <b>mixed (sandwich)</b> integration: unit tests for low-level modules, and run_inspection early with the stub detector. Integration then proceeded incrementally rather than all at once.</p>
<h3>Black box (functional) testing</h3>
<p>Cases come from the requirements, through equivalence classes and boundary values. Status shows what was actually run; the rest form the plan.</p>
{table(["ID", "Class or boundary", "Input", "Expected", "Status"], [
    ["TC-B1", "Valid: video", "IMG_3866.MOV + IMG_3939.MOV", "Five outputs", "Pass (onion run)"],
    ["TC-B2", "Invalid: no such path", "--day1 no_such_path_for_test", "\"input not found\", stops", "Pass (2 Nov)"],
    ["TC-B3", "Valid: Day 2 omitted", "--day1 only", "Baseline report", "Pass (21 Oct)"],
    ["TC-B4", "Valid: Polycam export", "Synthetic export", "Frames, poses, depth converted", "Pass (automated)"],
    ["TC-B5", "Boundary: 4 images", "Folder of 4 images", "Rejected", "Planned"],
    ["TC-B6", "Boundary: 5 images", "Folder of 5 images", "Accepted", "Planned"],
    ["TC-B7", "Boundary: inliers 39 and 40", "Pair at each count", "Rejected, then accepted", "Planned"],
    ["TC-B8", "Boundary: sharpness 99.9 and 100", "Frame at each score", "Replaced, then kept", "Planned"],
    ["TC-B9", "Boundary: rot gain 2.9, 3, 8 points", "Sack at each gain", "NORMAL, WARNING, CRITICAL", "Planned"],
], [8, 26, 26, 24, 16])}
<h3>White box (structural) testing</h3>
<p>Example: the blur rejection loop in keyframe_extractor.py.</p>
<pre style="font-size:9.5pt;margin:4pt 0 6pt 8pt">for c in cands:                                   # D1
    if sharp[c] &lt; thr:                            # D2
        lo, hi = max(0, c - win), min(n, c + win + 1)
        c = lo + int(np.argmax(sharp[lo:hi]))
    if not chosen or c != chosen[-1]:             # D3, D4 (two conditions)
        chosen.append(c)</pre>
<p><b>Cyclomatic complexity</b> V(G) = decisions + 1 = 4 + 1 = 5, so five independent paths need testing:</p>
{table(["Path", "Condition", "Test idea"], [
    ["1", "cands is empty", "Skip loop; chosen stays empty"],
    ["2", "Sharp frame, first in list", "Added directly"],
    ["3", "Sharp frame, differs from the last chosen", "Added"],
    ["4", "Blurred frame, replacement differs", "Replaced, then added"],
    ["5", "Blurred frame, replacement equals the last chosen", "Duplicate dropped"],
], [10, 50, 40])}
<h3>Performance testing</h3>
<p>Measured on a laptop CPU: 791 s for the onion pair (11 Day 1 and 15 Day 2 keyframes, 10 pairs). Full-length 4K walks take about 20 minutes.</p>
<h3>System testing</h3>
<p><b>Alpha testing</b> was the team's run on real godown footage. <b>Beta testing</b> at a cold store with real staff has not been done. Results from the real run: 22 sacks, 4 NORMAL, 4 WARNING, 1 MISSING and 13 UNVERIFIED, with a swapped sack found.</p>
<h3>Acceptance testing</h3>
{table(["Criterion", "Result"], [
    ["A pair of phone videos gives all five outputs with one command", "Accepted"],
    ["A changed sack is flagged with a reason", "Accepted (Sack_011)"],
    ["Blocked sacks are not called missing", "Accepted (pillar case)"],
    ["Most sacks get a definite verdict from handheld footage", "<b>Not accepted:</b> 13 of 22 unverified; needs consistent capture"],
    ["Detector reaches mask mAP@50 of 0.85", "<b>Not accepted:</b> 0.797"],
], [70, 30])}
<h3>Regression testing</h3>
<p>The test suite was re-run after each rule change. A real example: adding the occupancy rules made removed-sack tests fail, which led to the truncation rule and a more realistic gap fill in tools/synth/make_real_pair.py.</p>

<h2>B.5 Quality, reliability and maintenance</h2>
<h3>Software quality assurance plan</h3>
{table(["Item", "Plan"], [
    ["Purpose", "Ensure that releases meet Part A"],
    ["Standards", "Python conventions; IEEE 830 for requirements"],
    ["Reviews", "SRS review by team and owner; code walkthrough before each release"],
    ["Testing", "All automated tests pass; one real-footage run per release"],
    ["Metrics", "mAP, precision, share of UNVERIFIED sacks, false alarms per run, run time"],
    ["Problem reporting", "Record each false result with sack ID, reason text and run folder (see GUIDE.md checklist)"],
    ["Configuration management", "To add: git repository, tagged versions of code, dataset and weights. Dataset zips are already versioned (v0, v1, v1.1)."],
], [24, 76])}
<h3>ISO 9000</h3>
<p>ISO 9000 is a family of standards for quality management systems. ISO 9001 applies to organisations that design, develop and maintain software. It certifies the process rather than the product, and requires documented procedures, reviews and corrective action. This project has parts of it: documented requirements, test records and corrective action.</p>
<h3>SEI Capability Maturity Model</h3>
{table(["Level", "Name", "Key trait", "This project"], [
    ["1", "Initial", "Ad hoc; depends on individuals", "Started here"],
    ["2", "Repeatable", "Basic project management; size, cost, schedule tracked", "<b>Partly reached:</b> estimates, milestones and a risk table exist, with repeatable commands"],
    ["3", "Defined", "Standard documented process", "Partly: pipeline and dataset steps are documented; no organisation-wide process"],
    ["4", "Managed", "Process and product measured", "Product measured (mAP, tests), process not"],
    ["5", "Optimising", "Continuous improvement from the data", "Not reached"],
], [8, 15, 35, 42])}
<p>ISO 9000 asks whether a documented process exists, while CMM gives steps for improving one.</p>
<h3>Software reliability</h3>
<p>Reliability is the probability of failure-free operation over a period. Common metrics are ROCOF, MTTF, MTTR, MTBF and POFOD. Reliability here depends strongly on the input: the same code does well on consistent footage and poorly on changed viewpoints, a known feature of software reliability, which depends on the operational profile. Too few real runs exist to estimate MTTF. Available measures:</p>
<ul><li><b>Per verdict:</b> detector precision of 0.798 and a synthetic false-alarm count of 0.</li>
<li><b>Design for reliability:</b> the UNVERIFIED status turns unreliable conditions into a declared limit rather than a wrong answer.</li></ul>
<h3>Software maintenance</h3>
{table(["Type", "Meaning", "Example"], [
    ["Corrective", "Fix faults", "Kaggle multi-GPU AutoBatch crash; pillar false MISSING"],
    ["Adaptive", "Fit a new environment", "Support 4K and Polycam input; future drone camera"],
    ["Perfective", "Improve or add", "Trim options; replaced-sack rule; model v2"],
    ["Preventive", "Ease future change", "Config-driven thresholds; test suite"],
], [15, 30, 55])}
<p>Thresholds need retuning for each new site, and the model needs retraining when conditions change.</p>
<h3>Reverse engineering</h3>
<p>Reverse engineering recovers design from existing code. GUIDE.md section 5 was produced this way: file roles and libraries were recovered by reading the code and its imports.</p>
<h3>Re-engineering</h3>
<p>Re-engineering improves a working system without changing what it does for the user. Here the pairing step was re-engineered: SIFT was replaced by CLIP and LoFTR behind the same interface, returning the same Registration structure. A config switch keeps both (registration.matcher).</p>
<h3>Software reuse</h3>
{table(["Reused item", "Source", "Benefit"], [
    ["Computer vision primitives", "OpenCV", "SIFT, Farneback, warping, without writing them"],
    ["Detector architecture and training", "ultralytics YOLO11s-seg", "Transfer learning from pretrained weights"],
    ["Dense matching", "kornia LoFTR", "Solved cross-day pairing"],
    ["Image embeddings", "OpenCLIP", "Retrieval and dataset filtering"],
    ["Assignment and texture", "SciPy, scikit-image", "Hungarian algorithm, GLCM"],
    ["Within the project", "tools/dataset, detector seam", "Autolabel filters shared with fix tools; pipeline reused by process.py and main.py"],
], [34, 28, 38])}

<h2>B.6 Emerging topics</h2>
<h3>Software as a Service</h3>
<p>In SaaS, the provider hosts one application and customers use it over the internet, usually by subscription. The current version is a local tool, but SaaS fits its future well: cold stores have no GPUs or ML staff.</p>
{table(["Aspect", "Today (local tool)", "As SaaS"], [
    ["Compute", "Customer's laptop", "Provider's GPU servers"],
    ["Updates", "Manual copy of weights", "Central; all customers improve at once"],
    ["Interface", "Command line", "Web upload and dashboard"],
    ["New requirements", "None", "Login, roles, tenant data separation, encryption, billing, 99.9% availability"],
    ["Risk", "Local data only", "Footage leaves the site, so privacy terms are needed"],
], [20, 35, 45])}
<h3>AI-based software engineering</h3>
<p>AI now supports the software process itself. This project used it in three ways, recorded here so the method is transparent:</p>
<ul><li><b>AI coding assistant:</b> Claude Code was used to write and revise code, scripts and documents under human direction. The team set scope, gave footage and checked results.</li>
<li><b>AI agents in the data pipeline:</b> reviewer agents judged images and label quality, and gave correction points for SAM 2.1. This reduced an estimated 8 to 15 hours of manual labelling.</li>
<li><b>AI components in the product:</b> YOLO, LoFTR and CLIP. Their behaviour is learned, not written, so they are tested statistically (mAP) rather than only with fixed cases.</li></ul>
<p>AI output must be verified: tests caught regressions, and checks against the code found documentation errors.</p>

<div class="part"><h1>Part C: Syllabus Traceability</h1></div>
{table(["Module", "Topic", "Section"], SYLLABUS, [11, 74, 15])}
"""

if __name__ == "__main__":
    to_pdf(page("SRS Extended: ColdStore-Vision", body), "SRS_Extended_ColdStore_Vision.pdf")
    print(f"KLOC={KLOC:.3f} E_org={basic['Organic'][0]:.2f} EAF={EAF:.3f} E_int={E_INT:.2f} T_int={T_INT:.2f} UFP={UFP} FP={FP:.1f} crit={CRIT} END={END} Te={T_E:.2f}")
