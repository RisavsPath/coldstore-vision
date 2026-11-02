"""Shared paths and manifest helpers for the dataset tooling."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRAPED = ROOT / "data" / "scraped"
RAW_DIR = SCRAPED / "raw"
CANDIDATES_DIR = SCRAPED / "candidates"
REVIEW_DIR = SCRAPED / "reviews"
SHEETS_DIR = SCRAPED / "review_sheets"
ACCEPTED_DIR = SCRAPED / "accepted"
LABELS_DIR = SCRAPED / "autolabels"
OVERLAYS_DIR = SCRAPED / "overlays"
DATASET_DIR = ROOT / "data" / "dataset"
REPORTS_DIR = ROOT / "reports"

MANIFEST = SCRAPED / "manifest.csv"
CANDIDATES_CSV = SCRAPED / "candidates.csv"
ACCEPTED_CSV = SCRAPED / "accepted.csv"
LABEL_REVIEW_CSV = SCRAPED / "label_review.csv"

CLASS_NAMES = ["sack", "rack_beam", "empty_slot"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
