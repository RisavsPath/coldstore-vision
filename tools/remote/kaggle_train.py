"""Drive YOLO training on a Kaggle GPU from this machine (official `kaggle` CLI).

One-time setup: kaggle.com -> Settings -> Phone verification (needed for GPU) and
"Create New Token" -> save as  %USERPROFILE%\\.kaggle\\kaggle.json

    python -m tools.remote.kaggle_train upload   # create/version the dataset from data/<name>.zip
    python -m tools.remote.kaggle_train push     # push models/train_yolo.py as a GPU script kernel
    python -m tools.remote.kaggle_train status
    python -m tools.remote.kaggle_train wait     # poll until done, then fetch
    python -m tools.remote.kaggle_train fetch    # download outputs -> models/weights/, reports/training/
    python -m tools.remote.kaggle_train all      # upload + push + wait
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "kaggle"
DATASET_SLUG = "coldstore-sacks-v1"
KERNEL_SLUG = "coldstore-yolo11s-seg-train"
DEPS_SLUG = "coldstore-train-deps"
ACCELERATOR = "NvidiaTeslaT4"  # explicit: enable_gpu alone left runs queued / on CPU


def kaggle_bin() -> list[str]:
    exe = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    return [str(exe)] if exe.exists() else ["kaggle"]


def run(*args: str, check: bool = True) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}  # Kaggle logs contain emoji (cp1252 crash)
    p = subprocess.run(kaggle_bin() + list(args), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    if check and p.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout + p.stderr


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    # new-style KGAT access token (~/.kaggle/access_token): ask the CLI who we are
    for line in run("config", "view", check=False).splitlines():
        if line.strip().startswith("- username:"):
            name = line.split(":", 1)[1].strip()
            if name and name != "None":
                return name
    raise SystemExit("no Kaggle credentials - create an API token on kaggle.com (Settings -> API)")


def upload(zip_name: str) -> None:
    src = ROOT / "data" / f"{zip_name}.zip"
    if not src.exists():
        raise SystemExit(f"{src} not found - run tools.dataset.split_and_package first")
    d = BUILD / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    # upload the single zip; Kaggle extracts archives on its side, and train_yolo.py handles both an
    # extracted tree (finds dataset.yaml) and a still-zipped mount (extracts it to /kaggle/working)
    shutil.copy2(src, d / src.name)
    meta = {"title": DATASET_SLUG.replace("-", " ").title(), "id": f"{username()}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}
    (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    # new version if the dataset exists, otherwise create it
    out = run("datasets", "version", "-p", str(d), "-m", f"update {time.strftime('%Y-%m-%d %H:%M')}",
              check=False)
    if "error" in out.lower() or "not found" in out.lower() or "404" in out:
        out = run("datasets", "create", "-p", str(d))
    print(out)
    print("waiting for dataset to be ready...")
    for _ in range(60):
        s = run("datasets", "status", meta["id"], check=False).strip().lower()
        if "ready" in s:
            print("dataset ready")
            return
        time.sleep(10)
    print("dataset still processing; push will wait on Kaggle's side")


def upload_deps() -> None:
    """Private dataset with ultralytics wheels + base weights + font, so the kernel works without internet."""
    src = ROOT / "build" / "kaggle" / "deps_src"
    if not any(src.glob("ultralytics-*.whl")):
        raise SystemExit(f"{src} has no ultralytics wheel - run: pip download ultralytics ultralytics-thop "
                         f"--no-deps -d {src} and copy yolo11s-seg.pt, yolo11n.pt, Arial.ttf there")
    meta = {"title": "ColdStore train deps", "id": f"{username()}/{DEPS_SLUG}", "licenses": [{"name": "other"}]}
    (src / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    out = run("datasets", "version", "-p", str(src), "-m", "update", check=False)
    if "error" in out.lower() or "not found" in out.lower() or "404" in out:
        out = run("datasets", "create", "-p", str(src))
    print(out)
    for _ in range(60):
        if "ready" in run("datasets", "status", meta["id"], check=False).lower():
            print("deps dataset ready")
            return
        time.sleep(10)


def push(epochs: int, imgsz: int, model: str) -> None:
    d = BUILD / "kernel"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    script = (ROOT / "models" / "train_yolo.py").read_text()
    # bake CLI defaults for the kernel run
    script = script.replace('default="yolo11s-seg.pt"', f'default="{model}"')
    script = script.replace("default=150", f"default={epochs}")
    script = script.replace("default=640", f"default={imgsz}")
    (d / "train_yolo.py").write_text(script)
    user = username()
    meta = {
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "train_yolo.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": ACCELERATOR,
        "dataset_sources": [f"{user}/{DATASET_SLUG}", f"{user}/{DEPS_SLUG}"], "competition_sources": [], "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(run("kernels", "push", "-p", str(d), "--accelerator", ACCELERATOR))


def status() -> str:
    s = run("kernels", "status", f"{username()}/{KERNEL_SLUG}", check=False)
    print(s.strip())
    return s.lower()


def fetch() -> None:
    out = BUILD / "output" / KERNEL_SLUG
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    print(run("kernels", "output", f"{username()}/{KERNEL_SLUG}", "-p", str(out)))
    weights_dst = ROOT / "models" / "weights"
    report_dst = ROOT / "reports" / "training"
    weights_dst.mkdir(parents=True, exist_ok=True)
    report_dst.mkdir(parents=True, exist_ok=True)
    found = False
    for p in out.rglob("*"):
        if p.is_file() and "weights_out" in p.parts:
            if p.name in ("best.pt", "best.onnx"):
                shutil.copy2(p, weights_dst / p.name)
                found = True
            elif p.suffix in (".json", ".csv"):
                shutil.copy2(p, report_dst / p.name)
        elif p.is_file() and p.suffix == ".png" and "coldstore_seg" in p.parts:
            shutil.copy2(p, report_dst / p.name)
        elif p.is_file() and p.suffix == ".log":
            shutil.copy2(p, report_dst / p.name)
    if not found:
        raise SystemExit(f"best.pt not found in kernel output ({out}); check the kernel log")
    m = report_dst / "metrics.json"
    if m.exists():
        print(m.read_text())
    print(f"weights -> {weights_dst}")


def wait(poll_s: int) -> None:
    while True:
        s = status()
        if "complete" in s:
            fetch()
            return
        if "error" in s or "cancel" in s:
            raise SystemExit("kernel failed - see https://www.kaggle.com/code (log) or run fetch for the log")
        time.sleep(poll_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "deps", "push", "status", "fetch", "wait", "all"])
    ap.add_argument("--zip-name", default="coldstore_sacks_v1")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--dataset-slug", default=None, help="Kaggle dataset name (default coldstore-sacks-v1)")
    ap.add_argument("--kernel-slug", default=None, help="Kaggle kernel name, lets several runs coexist")
    a = ap.parse_args()
    global DATASET_SLUG, KERNEL_SLUG
    DATASET_SLUG = a.dataset_slug or DATASET_SLUG
    KERNEL_SLUG = a.kernel_slug or KERNEL_SLUG
    if a.cmd in ("upload", "all"):
        upload(a.zip_name)
    if a.cmd in ("deps", "all"):
        upload_deps()
    if a.cmd in ("push", "all"):
        push(a.epochs, a.imgsz, a.model)
    if a.cmd == "status":
        status()
    if a.cmd == "fetch":
        fetch()
    if a.cmd in ("wait", "all"):
        wait(a.poll)


if __name__ == "__main__":
    main()
