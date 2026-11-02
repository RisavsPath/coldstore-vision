"""Step 1: scrape sack images from Openverse, Wikimedia Commons and Bing.

Openverse / Wikimedia images carry CC licenses (recorded in the manifest).
Bing results are supplementary and for internal research/training use only.

Usage:
    python -m tools.dataset.scrape_images [--sources openverse wikimedia bing] [--limit-queries N]
Re-running is safe: files already present (by URL hash) are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import time
from pathlib import Path

import requests
import yaml
from PIL import Image
from tqdm import tqdm

from tools.dataset.common import MANIFEST, RAW_DIR, ROOT, read_csv, write_csv

UA = "ColdStoreVision-dataset/0.1 (research; python-requests)"
FIELDS = ["file", "source", "url", "landing_url", "query", "produce_hint", "license", "creator", "width", "height"]
MIN_SIDE = 320
session = requests.Session()
session.headers["User-Agent"] = UA


def url_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def download(url: str, dest_dir: Path) -> tuple[Path, int, int] | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{url_id(url)}.jpg"
    if dest.exists():
        with Image.open(dest) as im:
            return dest, im.width, im.height
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    if min(im.size) < MIN_SIDE:
        return None
    if max(im.size) > 1600:
        im.thumbnail((1600, 1600))
    im.save(dest, quality=92)
    return dest, im.width, im.height


# --------------------------------------------------------------------------- sources
def search_openverse(q: str, n: int) -> list[dict]:
    out, page = [], 1
    while len(out) < n and page <= 5:
        try:
            r = session.get(
                "https://api.openverse.org/v1/images/",
                params={"q": q, "page_size": 20, "page": page, "mature": "false"},
                timeout=20,
            )
            if r.status_code != 200:
                break
            res = r.json().get("results", [])
        except Exception:
            break
        if not res:
            break
        for it in res:
            out.append({
                "url": it.get("url"),
                "landing_url": it.get("foreign_landing_url", ""),
                "license": f"{it.get('license', '')} {it.get('license_version', '')}".strip(),
                "creator": it.get("creator") or "",
            })
        page += 1
        time.sleep(1.0)  # anonymous rate limit
    return out[:n]


def search_wikimedia(q: str, n: int) -> list[dict]:
    try:
        r = session.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": f"filetype:bitmap {q}", "gsrnamespace": 6, "gsrlimit": min(n, 50),
                "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1280,
            },
            timeout=20,
        )
        pages = r.json().get("query", {}).get("pages", {})
    except Exception:
        return []
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        out.append({
            "url": ii.get("thumburl") or ii.get("url"),
            "landing_url": ii.get("descriptionurl", ""),
            "license": meta.get("LicenseShortName", {}).get("value", ""),
            "creator": meta.get("Artist", {}).get("value", "")[:120],
        })
    return out[:n]


def scrape_crawler(q: str, n: int, dest: Path, engine: str = "bing") -> list[dict]:
    """icrawler (Bing or Baidu) downloads directly; we read back what it saved."""
    from icrawler.builtin import BaiduImageCrawler, BingImageCrawler

    tmp = dest / "_tmp" / url_id(q)
    tmp.mkdir(parents=True, exist_ok=True)
    cls = BaiduImageCrawler if engine == "baidu" else BingImageCrawler
    crawler = cls(storage={"root_dir": str(tmp)}, log_level=40, downloader_threads=4)
    crawler.crawl(keyword=q, max_num=n, min_size=(MIN_SIDE, MIN_SIDE))
    rows = []
    for f in sorted(tmp.iterdir()):
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            f.unlink(missing_ok=True)
            continue
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600))
        key = f"{engine}:{q}:{f.name}"
        out = dest / f"{url_id(key)}.jpg"
        im.save(out, quality=92)
        f.unlink(missing_ok=True)
        rows.append({"file": out, "url": key, "landing_url": "", "license": "unknown (web)",
                     "creator": "", "width": im.width, "height": im.height})
    return rows


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["openverse", "wikimedia", "bing"])
    ap.add_argument("--limit-queries", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "scrape_queries.yaml", encoding="utf-8"))
    caps = cfg["per_query_max"]
    lim = args.limit_queries or None

    rows = read_csv(MANIFEST)
    seen = {r["url"] for r in rows}

    for src in args.sources:
        dest = RAW_DIR / src
        key = {"bing": "queries", "baidu": "baidu_queries"}.get(src, "cc_queries")
        queries = cfg[key][:lim]
        if src == "bing":
            queries = queries + cfg.get("extra_queries", [])[:lim]
        for qd in tqdm(queries, desc=src):
            q, produce = qd["q"], qd["produce"]
            if src in ("bing", "baidu"):
                got = scrape_crawler(q, caps[src], dest, src)
                for g in got:
                    if g["url"] in seen:
                        continue
                    g.update(source=src, query=q, produce_hint=produce,
                             file=str(Path(g["file"]).relative_to(ROOT)).replace("\\", "/"))
                    rows.append(g)
                    seen.add(g["url"])
            else:
                hits = search_openverse(q, caps[src]) if src == "openverse" else search_wikimedia(q, caps[src])
                for h in hits:
                    if not h["url"] or h["url"] in seen:
                        continue
                    res = download(h["url"], dest)
                    if res is None:
                        continue
                    path, w, hgt = res
                    rows.append({**h, "file": str(path.relative_to(ROOT)).replace("\\", "/"), "source": src,
                                 "query": q, "produce_hint": produce, "width": w, "height": hgt})
                    seen.add(h["url"])
            write_csv(MANIFEST, rows, FIELDS)  # checkpoint after every query

    print(f"manifest: {len(rows)} images -> {MANIFEST}")
    for src in args.sources:
        print(f"  {src}: {sum(r['source'] == src for r in rows)}")


if __name__ == "__main__":
    main()
