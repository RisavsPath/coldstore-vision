"""Render every figure in the built HTML to PNG (figs/) for the online docs versions."""
from pathlib import Path

from playwright.sync_api import sync_playwright

out = Path("figs")
out.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(device_scale_factor=2)
    for stem, tag in (("SRS_Extended_ColdStore_Vision", "srs"), ("Project_Report_ColdStore_Vision", "rep")):
        pg.goto(Path(f"{stem}.html").resolve().as_uri())
        for i, el in enumerate(pg.query_selector_all("figure")):
            f = out / f"{tag}_{i + 1:02d}.png"
            el.query_selector("svg, img").screenshot(path=str(f))
            print(f, "|", el.query_selector("figcaption").inner_text())
    b.close()
