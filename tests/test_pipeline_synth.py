"""End-to-end test on the synthetic aisle: every injected anomaly found, no false alarms, 5 outputs."""
import json

import numpy as np


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


EXPECTED_STATUS = {"missing": {"MISSING"}, "rot": {"CRITICAL"}, "slump": {"WARNING", "CRITICAL"}, "new": {"NEW"}}


def _gt_boxes(gt):
    boxes = {g["id"]: g["box"] for g in gt["day1"] if g["box"]}
    boxes.update({g["id"]: g["box"] for g in gt["day2"] if g["kind"] == "new"})
    return boxes


def test_keyframes_and_registration(inspection):
    assert 8 <= len(inspection.kf1) <= 20
    assert len(inspection.pairs) >= len(inspection.kf1) - 1
    assert all(p.reg.inliers >= 25 for p in inspection.pairs)


def test_sack_count_matches_day1(inspection, synth_pair):
    _, gt = synth_pair
    n_day1 = sum(1 for g in gt["day1"] if g["box"])
    assert inspection.kpis["total_monitored"] == n_day1


def test_every_anomaly_detected_without_false_alarms(inspection, synth_pair):
    _, gt = synth_pair
    scale = 1280 / 1600  # synthetic canvas -> frame/mosaic scale
    false_alarms, misses = [], []
    for gid, box in _gt_boxes(gt).items():
        b = [v * scale for v in box]
        best = max(inspection.sacks, key=lambda s: _iou(s.obs.mosaic_box, b))
        assert _iou(best.obs.mosaic_box, b) > 0.4, f"{gid} not localised"
        kind = gt["expected"].get(gid, "normal")
        if kind == "normal" and best.status != "NORMAL":
            false_alarms.append((gid, best.status, best.reasoning))
        elif kind != "normal" and best.status not in EXPECTED_STATUS[kind]:
            misses.append((gid, kind, best.status, best.reasoning))
    assert not misses, misses
    assert len(false_alarms) <= 1, false_alarms


def test_missing_uses_depth(inspection):
    m = [s for s in inspection.sacks if s.status == "MISSING"]
    assert m and m[0].metrics["depth_delta_m"] > 0.3


def test_bays_and_tiers(inspection):
    assert {s.bay for s in inspection.sacks} == {1, 2, 3, 4}
    assert {s.tier for s in inspection.sacks} == {1, 2, 3}


def test_mosaics_aligned(inspection):
    m1, m2 = inspection.mosaic1, inspection.mosaic2
    assert m1.image.shape == m2.image.shape
    both = m1.valid & m2.valid
    assert both.mean() > 0.6
    g1 = m1.image[both].astype(float).mean()
    assert g1 > 0


def test_outputs_written(inspection, cfg, tmp_path):
    from src.reporting.output_generator import generate_outputs

    paths = generate_outputs(inspection, cfg, tmp_path)
    for key in ("anomaly_video", "side_by_side", "manifest", "audit_report", "digital_twin", "panoramic_slider"):
        assert paths[key].exists() and paths[key].stat().st_size > 1000, key
    man = json.loads(paths["manifest"].read_text())
    assert man["kpis"]["missing"] == 1 and len(man["sacks"]) == len(inspection.sacks)
    import cv2

    cap = cv2.VideoCapture(str(paths["anomaly_video"]))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 10
    ok, frame = cap.read()
    assert ok and np.asarray(frame).std() > 5
