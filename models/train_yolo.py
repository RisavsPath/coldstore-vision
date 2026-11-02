"""STEP 0: train the YOLO11-seg sack / rack_beam / empty_slot model.

Runs unchanged locally, on Kaggle (pushed as a script kernel) and on Colab:
  local : python models/train_yolo.py --data configs/dataset.yaml
  kaggle: auto-detects /kaggle/input, finds dataset.yaml (or extracts the zip), writes to /kaggle/working
  colab : python train_yolo.py --data /content/data/dataset.yaml --out /content/drive/MyDrive/coldstore/weights

Outputs (in --out): best.pt, best.onnx, best.mlpackage (if coremltools works), metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

GATE_MAP50 = 0.85
GATE_PRECISION = 0.80
ON_KAGGLE = Path("/kaggle/input").exists()


def _deps_dir() -> Path | None:
    """Kaggle 'coldstore-train-deps' dataset: ultralytics wheels, base weights, plot font (works offline)."""
    if not ON_KAGGLE:
        return None
    for whl in Path("/kaggle/input").rglob("ultralytics-*.whl"):
        return whl.parent
    return None


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass
    deps = _deps_dir()
    if deps is not None:  # offline install from the deps dataset (Kaggle kernels may have no internet)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                               *map(str, sorted(deps.glob("*.whl")))])
        font_dir = Path.home() / ".config" / "Ultralytics"
        font_dir.mkdir(parents=True, exist_ok=True)
        for f in deps.glob("*.ttf"):
            shutil.copy2(f, font_dir / f.name)
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "albumentations"])


def resolve_weights(name: str) -> str:
    """Prefer a local copy of the base weights (deps dataset) so no download is needed."""
    deps = _deps_dir()
    if deps is not None and (deps / name).exists():
        for extra in deps.glob("*.pt"):  # e.g. yolo11n.pt for the AMP check
            shutil.copy2(extra, Path.cwd() / extra.name)
        return str(Path.cwd() / name)
    return name


def find_kaggle_dataset() -> Path:
    work = Path("/kaggle/working/data")
    for y in Path("/kaggle/input").rglob("dataset.yaml"):
        return y
    for z in Path("/kaggle/input").rglob("*.zip"):
        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(work)
        for y in work.rglob("dataset.yaml"):
            return y
    raise FileNotFoundError("no dataset.yaml or zip under /kaggle/input")


def localise_yaml(yaml_path: Path, workdir: Path) -> Path:
    """Rewrite `path:` to the yaml's own directory so the dataset is location-independent."""
    import yaml

    cfg = yaml.safe_load(yaml_path.read_text())
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    if not (root / cfg["train"]).exists():
        root = yaml_path.parent
    cfg["path"] = str(root)
    out = workdir / "dataset_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        # single GPU: AutoBatch (batch=-1) is not supported with multi-GPU DDP, and DDP from a
        # Kaggle script kernel adds failure modes for little gain on a ~150-image dataset
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extra_augmentations():
    """Spec 2.2 augmentations that YOLO's built-ins don't cover (grayscale 15%, blur <=1.5px)."""
    try:
        import albumentations as A
    except ImportError:
        print("albumentations not installed: spec grayscale/blur augmentations skipped")
        return None
    return [
        A.ToGray(p=0.15),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.3, 1.5), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.2, p=0.3),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto batch")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a fraction (smoke tests)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workdir", type=Path, default=None, help="where training runs are written")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    ensure_ultralytics()
    from ultralytics import YOLO

    if ON_KAGGLE:
        workdir = Path("/kaggle/working")
        data = args.data or find_kaggle_dataset()
    else:
        repo = Path(__file__).resolve().parents[1]
        workdir = repo / "runs"
        data = args.data or repo / "configs" / "dataset.yaml"
    workdir = args.workdir or workdir
    workdir.mkdir(parents=True, exist_ok=True)
    out = args.out or (workdir / "weights_out" if ON_KAGGLE else Path(__file__).resolve().parent / "weights")
    out.mkdir(parents=True, exist_ok=True)
    data_yaml = localise_yaml(Path(data), workdir)
    device = pick_device(args.device)
    batch = args.batch if not (device == "cpu" and args.batch == -1) else 4
    import torch

    print(f"data={data_yaml} device={device} model={args.model} cuda={torch.cuda.is_available()} "
          f"gpus={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}", flush=True)
    if ON_KAGGLE and device == "cpu":
        print("WARNING: no GPU on this Kaggle session (phone-verify the account to get GPU)", flush=True)

    os.chdir(workdir)
    model = YOLO(resolve_weights(args.model))
    train_kw = dict(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=batch, patience=args.patience,
        device=device, project=str(workdir / "coldstore_seg"), name="yolo11s_sack", exist_ok=True,
        fraction=args.fraction, cos_lr=True, close_mosaic=15, plots=True, workers=2 if ON_KAGGLE else 0,
        # spec 2.2: brightness/exposure ±25/20 %, hue ±15° (0.04 of the 360° wheel), mosaic + mixup
        hsv_h=0.04, hsv_s=0.5, hsv_v=0.25, degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.1, copy_paste=0.3, overlap_mask=False,
    )
    aug = extra_augmentations()
    if aug:
        train_kw["augmentations"] = aug  # custom Albumentations (ultralytics >= 8.3.1xx)
    try:
        model.train(**train_kw)
    except SyntaxError as e:  # older ultralytics rejects unknown keys
        if "augmentations" not in str(e):
            raise
        print("ultralytics too old for custom augmentations; training without them")
        train_kw.pop("augmentations")
        model.train(**train_kw)

    best = Path(model.trainer.best)
    best_model = YOLO(str(best))
    m = best_model.val(data=str(data_yaml), split="test" if _has_test(data_yaml) else "val", device=device)
    metrics = {
        "mask_map50": float(m.seg.map50), "mask_map50_95": float(m.seg.map),
        "mask_precision": float(m.seg.mp), "mask_recall": float(m.seg.mr),
        "box_map50": float(m.box.map50),
        "per_class_mask_map50_95": {best_model.names[i]: float(v) for i, v in enumerate(m.seg.maps)},
        "gate": {"map50_required": GATE_MAP50, "precision_required": GATE_PRECISION},
    }
    metrics["gate"]["passed"] = metrics["mask_map50"] >= GATE_MAP50 and metrics["mask_precision"] >= GATE_PRECISION
    print(json.dumps(metrics, indent=2))

    shutil.copy2(best, out / "best.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    results_csv = best.parent.parent / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, out / "results.csv")
    if not args.no_export:
        try:
            onnx = best_model.export(format="onnx", dynamic=True, imgsz=args.imgsz)
            shutil.copy2(onnx, out / "best.onnx")
        except Exception as e:  # export problems must not lose the trained weights
            print("onnx export failed:", e)
        if sys.platform == "win32":
            print("coreml export skipped: not supported on Windows (runs on Kaggle/Colab Linux)")
        else:
            try:
                ml = Path(best_model.export(format="coreml", nms=True, imgsz=args.imgsz))
                shutil.copytree(ml, out / ml.name, dirs_exist_ok=True) if ml.is_dir() else shutil.copy2(ml, out)
            except Exception as e:
                print("coreml export skipped:", e)
    print(f"artifacts -> {out}  gate passed: {metrics['gate']['passed']}")
    if ON_KAGGLE:
        shutil.make_archive("/kaggle/working/weights_out", "zip", out)


def _has_test(yaml_path: Path) -> bool:
    import yaml

    cfg = yaml.safe_load(yaml_path.read_text())
    return bool(cfg.get("test")) and (Path(cfg["path"]) / cfg["test"]).exists() and any(
        (Path(cfg["path"]) / cfg["test"]).iterdir())


if __name__ == "__main__":
    os.environ.setdefault("YOLO_VERBOSE", "True")
    main()
