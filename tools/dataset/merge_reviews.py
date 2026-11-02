"""Step 3b / 5b: merge reviewer-agent verdict files.

mode images -> data/scraped/accepted.csv, data/scraped/accepted/<id>.jpg, reports/dataset_stats.md section
mode labels -> data/scraped/label_review.csv (ok / fix / drop per image)
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter

from tools.dataset.common import (ACCEPTED_CSV, ACCEPTED_DIR, CANDIDATES_CSV, LABEL_REVIEW_CSV, REPORTS_DIR,
                                  REVIEW_DIR, ROOT, read_csv, write_csv)


def load_results(mode: str) -> tuple[dict, list[str]]:
    results, missing_batches = {}, []
    spec_dir = ROOT / "data" / "scraped" / "review_sheets" / mode
    for spec_path in sorted(spec_dir.glob("*.json")):
        spec = json.loads(spec_path.read_text())
        out = ROOT / spec["output"]
        if not out.exists():
            missing_batches.append(spec["batch"])
            continue
        res = json.loads(out.read_text(encoding="utf-8")).get("results", {})
        lost = set(spec["images"]) - set(res)
        if lost:
            print(f"warning: {spec['batch']} missing verdicts for {len(lost)} ids")
        results.update(res)
    return results, missing_batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["images", "labels"], default="images")
    args = ap.parse_args()

    results, missing = load_results(args.mode)
    if missing:
        print(f"batches without verdicts yet: {missing}")

    if args.mode == "images":
        cands = {r["id"]: r for r in read_csv(CANDIDATES_CSV)}
        accepted = []
        ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
        for img_id, v in results.items():
            if not v.get("accept") or img_id not in cands:
                continue
            r = cands[img_id]
            dst = ACCEPTED_DIR / f"{img_id}.jpg"
            if not dst.exists():
                shutil.copy2(ROOT / r["candidate"], dst)
            accepted.append({**r, "produce": v.get("produce", "unknown"), "material": v.get("material", ""),
                             "scene": v.get("scene", ""), "sack_count_est": v.get("sack_count_est", ""),
                             "has_rack_beam": v.get("has_rack_beam", False),
                             "has_empty_slot": v.get("has_empty_slot", False),
                             "accepted": str(dst.relative_to(ROOT)).replace("\\", "/")})
        write_csv(ACCEPTED_CSV, accepted)
        rej = Counter(v.get("reject_reason", "?") for v in results.values() if not v.get("accept"))
        prod = Counter(r["produce"] for r in accepted)
        mat = Counter(r["material"] for r in accepted)
        scene = Counter(r["scene"] for r in accepted)
        lines = ["## Image review", f"- reviewed: {len(results)}", f"- accepted: {len(accepted)}",
                 f"- produce: {dict(prod)}", f"- material: {dict(mat)}", f"- scene: {dict(scene)}",
                 f"- rack beams present: {sum(str(r['has_rack_beam']).lower() == 'true' for r in accepted)}",
                 "- top reject reasons: " + ", ".join(f"{k} ({n})" for k, n in rej.most_common(8))]
        REPORTS_DIR.mkdir(exist_ok=True)
        (REPORTS_DIR / "review_images.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
    else:
        rows = [{"id": k, "verdict": v.get("verdict", "fix"), "issues": "|".join(v.get("issues", [])),
                 "note": v.get("note", "")} for k, v in results.items()]
        write_csv(LABEL_REVIEW_CSV, rows, ["id", "verdict", "issues", "note"])
        c = Counter(r["verdict"] for r in rows)
        issues = Counter(i for r in rows for i in r["issues"].split("|") if i)
        lines = ["## Label review", f"- reviewed: {len(rows)}", f"- verdicts: {dict(c)}",
                 f"- issues: {dict(issues)}"]
        (REPORTS_DIR / "review_labels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
