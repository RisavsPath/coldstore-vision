"""Step 6: leak-free train/val/test split + zip for Kaggle/Colab.

Inclusion per image (accepted + has a label file):
  - human_verified labels                 -> always included
  - label review verdict `ok`             -> included
  - verdict `fix` but not yet corrected   -> excluded unless --include-unfixed
  - verdict `drop`                        -> excluded
  - no label review run at all            -> included (warning printed)

Near-duplicates are grouped (pHash Hamming <= --group-dist, union-find) and each group goes to a
single split. Groups are assigned per produce type so each split keeps the produce balance.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict

import imagehash
import yaml

from tools.dataset.common import (ACCEPTED_CSV, ACCEPTED_DIR, CLASS_NAMES, DATASET_DIR, LABEL_REVIEW_CSV,
                                  LABELS_DIR, REPORTS_DIR, ROOT, read_csv)

SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-unfixed", action="store_true")
    ap.add_argument("--group-dist", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="coldstore_sacks_v1")
    args = ap.parse_args()

    accepted = read_csv(ACCEPTED_CSV)
    review = {r["id"]: r["verdict"] for r in read_csv(LABEL_REVIEW_CSV)}
    if not review:
        print("warning: no label review found; including all auto-labels")

    items = []
    for r in accepted:
        lbl = LABELS_DIR / f"{r['id']}.txt"
        if not lbl.exists():
            continue
        meta_p = LABELS_DIR / f"{r['id']}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        human = meta.get("human_verified")
        if meta.get("agent_dropped") and not human:
            continue
        v = review.get(r["id"])
        if not human and review:
            if v == "drop" or v is None:
                continue
            if v == "fix" and not args.include_unfixed:
                continue
        if not lbl.read_text().strip():
            continue  # no instances -> skip (background images are rare in this domain)
        items.append(r)

    hashes = [imagehash.hex_to_hash(r["phash"]) for r in items]
    dsu = DSU(len(items))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if hashes[i] - hashes[j] <= args.group_dist:
                dsu.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(items)):
        groups[dsu.find(i)].append(i)

    by_produce: dict[str, list[list[int]]] = defaultdict(list)
    for g in groups.values():
        produce = Counter(items[i]["produce"] for i in g).most_common(1)[0][0]
        by_produce[produce].append(g)

    rng = random.Random(args.seed)
    assign: dict[int, str] = {}
    for produce, glist in by_produce.items():
        rng.shuffle(glist)
        total = sum(len(g) for g in glist)
        counts = Counter()
        for g in glist:
            # put the group in the split furthest below its target share
            split = min(SPLITS, key=lambda s: counts[s] / max(total, 1) - SPLITS[s])
            for i in g:
                assign[i] = split
            counts[split] += len(g)

    if DATASET_DIR.exists():
        for sub in ("images", "labels"):
            shutil.rmtree(DATASET_DIR / sub, ignore_errors=True)
    stats = {s: Counter() for s in SPLITS}
    for i, r in enumerate(items):
        s = assign[i]
        (DATASET_DIR / "images" / s).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / s).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ACCEPTED_DIR / f"{r['id']}.jpg", DATASET_DIR / "images" / s / f"{r['id']}.jpg")
        txt = (LABELS_DIR / f"{r['id']}.txt").read_text()
        (DATASET_DIR / "labels" / s / f"{r['id']}.txt").write_text(txt)
        stats[s]["images"] += 1
        stats[s][f"produce_{r['produce']}"] += 1
        for line in txt.splitlines():
            if line.strip():
                stats[s][CLASS_NAMES[int(line.split()[0])]] += 1

    ds_yaml = {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
               "names": dict(enumerate(CLASS_NAMES))}
    (DATASET_DIR / "dataset.yaml").write_text(yaml.safe_dump(ds_yaml, sort_keys=False))
    cfg = ROOT / "configs" / "dataset.yaml"
    cfg.write_text(yaml.safe_dump({**ds_yaml, "path": str(DATASET_DIR).replace("\\", "/")}, sort_keys=False))

    zpath = ROOT / "data" / f"{args.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p in DATASET_DIR.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(DATASET_DIR))

    lines = ["## Dataset split", f"- groups: {len(groups)}, images: {len(items)}", f"- zip: {zpath.name}"]
    for s in SPLITS:
        lines.append(f"- {s}: {dict(stats[s])}")
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "dataset_split.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
