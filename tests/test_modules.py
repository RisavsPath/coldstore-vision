"""Unit tests for individual modules on controlled inputs."""
import cv2
import numpy as np

from src.analytics.anomaly_engine import erode, masked_glcm, necrosis_fraction
from src.analytics.bay_indexer import index_bays
from src.analytics.lighting_normalizer import global_hue_shift, match_histograms
from src.analytics.tracker import match_instances
from src.registration.homography_warper import register
from src.registration.optical_flow import farneback, slump_metrics
from src.segmentation.sack_detector import instance_from_mask


def textured(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 1.5)
    for _ in range(60):
        cv2.circle(img, tuple(int(v) for v in rng.integers(0, [w, h])), int(rng.integers(4, 20)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def test_homography_recovers_known_warp(cfg):
    ref = textured()
    H_true = np.array([[1.02, 0.01, 12], [-0.01, 0.99, -8], [1e-5, 0, 1]])
    mov = cv2.warpPerspective(ref, np.linalg.inv(H_true), (640, 480))
    r = register(ref, mov, cfg)
    assert r.ok and r.inliers > 50
    pts = np.float32([[100, 100], [500, 400], [320, 240]]).reshape(-1, 1, 2)
    err = np.abs(cv2.perspectiveTransform(pts, r.H) - cv2.perspectiveTransform(pts, H_true)).max()
    assert err < 1.5


def test_histogram_matching_undoes_global_lighting():
    ref = textured()
    dark = np.clip(ref.astype(np.float32) * 0.7 * np.array([0.9, 1.0, 1.1]), 0, 255).astype(np.uint8)
    fixed = match_histograms(dark, ref)
    assert np.abs(fixed.astype(int) - ref.astype(int)).mean() < 0.35 * np.abs(dark.astype(int) - ref.astype(int)).mean()


def test_global_hue_shift_on_bimodal_background():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = (200, 100, 30)    # blue-ish
    img[:, 100:] = (0, 140, 255)     # orange
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + 5) % 180  # +10 degrees everywhere
    shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert abs(global_hue_shift(img, shifted, np.ones((100, 200), bool)) - 10) < 2.5


def test_necrosis_and_glcm_respond_to_rot():
    rng = np.random.default_rng(1)
    healthy = np.full((120, 120, 3), (70, 120, 160), np.uint8)
    healthy = np.clip(healthy + rng.normal(0, 4, healthy.shape), 0, 255).astype(np.uint8)
    rotten = healthy.copy()
    # luminance speckle (correlated across channels) keeps the hue of the rot patch in the necrotic band
    patch = np.clip(np.array((35, 60, 95)) + rng.normal(0, 25, (50, 60, 1)), 0, 255).astype(np.uint8)
    rotten[30:80, 30:90] = patch
    mask = np.ones((120, 120), bool)
    p = {"h_min": 10, "h_max": 30, "s_min": 60, "v_max": 110}
    assert necrosis_fraction(rotten, mask, p) - necrosis_fraction(healthy, mask, p) > 0.1
    c1, h1 = masked_glcm(healthy, mask, [1], [0, 45, 90, 135])
    c2, h2 = masked_glcm(rotten, mask, [1], [0, 45, 90, 135])
    assert c2 > 1.35 * c1 and h2 < h1


def test_erosion_shrinks_mask():
    m = np.zeros((100, 100), bool)
    m[20:80, 20:80] = True
    assert erode(m, 7).sum() < m.sum() * 0.75


def test_flow_detects_downward_shift(cfg):
    a = textured(seed=3)
    b = np.roll(a, 6, axis=0)
    flow = farneback(a, b, cfg)
    m = np.zeros(a.shape[:2], bool)
    m[100:380, 100:540] = True
    assert slump_metrics(flow, m)["flow_vy"] > 4.0


def test_hungarian_iou_matching():
    def inst(x):
        m = np.zeros((100, 300), bool)
        m[20:80, x:x + 50] = True
        return instance_from_mask("sack", m)

    a = [inst(10), inst(100), inst(200)]
    b = [inst(205), inst(12)]  # sack at x=100 removed, order shuffled
    pairs, un_a, un_b = match_instances(a, b, 0.3)
    assert {(i, j) for i, j, _ in pairs} == {(0, 1), (2, 0)} and un_a == [1] and un_b == []


def test_bay_indexer_uses_uprights_and_rows():
    sacks = [np.array([x + 5, y, x + 95, y + 80]) for x in (0, 100, 300, 400) for y in (20, 120)]
    beams = [np.array([195, 0, 205, 220]), np.array([-5, 0, 5, 220]), np.array([495, 0, 505, 220])]
    bt = index_bays(sacks, beams, 500, 220)
    assert {b for b, _ in bt} == {1, 2}
    assert bt[0] == (1, 2) and bt[1] == (1, 1)  # top row is tier 2
