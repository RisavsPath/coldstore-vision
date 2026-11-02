import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg():
    from src.config import load_config

    return load_config()


@pytest.fixture(scope="session")
def synth_pair(tmp_path_factory):
    """Synthetic Day 1 / Day 2 walkthroughs with ground truth (generated once per session)."""
    import json

    from tools.synth import make_pair

    out = tmp_path_factory.mktemp("synth")
    c1, d1, gt1 = make_pair.render(1)
    c2, d2, gt2 = make_pair.render(2)
    c2 = make_pair.relight(c2)
    make_pair.walk(c1, d1, out / "day_01", 1)
    make_pair.walk(c2, d2, out / "day_02", 2)
    expected = {f"b{b}t{t}s{s}": k for k, lst in make_pair.DAY2_CHANGES.items() for (b, t, s) in lst}
    gt = {"expected": expected, "day1": gt1, "day2": gt2}
    (out / "ground_truth.json").write_text(json.dumps(gt))
    return out, gt


@pytest.fixture(scope="session")
def inspection(synth_pair, cfg):
    from src.pipeline import run_inspection
    from tools.synth.synthetic_detector import SyntheticColorDetector

    out, _ = synth_pair
    return run_inspection(out / "day_01", out / "day_02", cfg, SyntheticColorDetector())
