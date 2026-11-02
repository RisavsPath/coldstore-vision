"""Shared helpers for the SE documents: plain SVG diagram primitives, HTML page shell, PDF export."""
from __future__ import annotations

import base64
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs"

CSS = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
body { font-family: 'Times New Roman', serif; font-size: 11.5pt; line-height: 1.32; color: #000; }
h1 { font-size: 16pt; margin: 18pt 0 8pt; break-after: avoid; }
h2 { font-size: 13pt; margin: 13pt 0 5pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 9pt 0 3pt; break-after: avoid; }
p { margin: 4pt 0 6pt; text-align: justify; }
ul, ol { margin: 3pt 0 6pt; padding-left: 20pt; }
li { margin: 1.5pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 10pt; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 1px solid #000; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #e6e6e6; }
figure { margin: 8pt 0 10pt; text-align: center; break-inside: avoid; }
figcaption { font-size: 10pt; margin-top: 3pt; }
figure img { max-width: 100%; }
svg { max-width: 100%; height: auto; }
code { font-family: Consolas, monospace; font-size: 9.5pt; }
mark { background: #ffef8a; }
.cover { text-align: center; padding-top: 28mm; break-after: page; }
.cover h1 { font-size: 21pt; margin: 8pt 0; }
.cover p { text-align: center; font-size: 12.5pt; }
.part { break-before: page; }
.note { border: 1px solid #000; padding: 4pt 7pt; margin: 6pt 0 10pt; font-size: 10.5pt; }
.toc { list-style: none; padding-left: 0; }
.toc ul { list-style: none; padding-left: 16pt; }
.formula { text-align: center; margin: 5pt 0; font-style: italic; }
"""


def table(head: list[str], rows: list[list], widths: list[int] | None = None) -> str:
    cols = ""
    if widths:
        cols = "<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>"
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{cols}<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def figure(svg_or_html: str, caption: str) -> str:
    return f"<figure>{svg_or_html}<figcaption>{caption}</figcaption></figure>"


def image(path: Path, width_pct: int = 100) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:{mime};base64,{data}" style="width:{width_pct}%">'


# ---- SVG primitives ---------------------------------------------------------------------------

def svg(w: int, h: int, *parts: str) -> str:
    defs = ('<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#000"/></marker></defs>')
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="Arial, sans-serif" font-size="12">{defs}{"".join(parts)}</svg>')


def text(x: float, y: float, s: str, size: int = 12, anchor: str = "middle", bold: bool = False) -> str:
    w = ' font-weight="700"' if bold else ""
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{w} '
            f'stroke="#fff" stroke-width="3" paint-order="stroke">{s}</text>')


def _lines(cx: float, cy: float, lines: tuple, size: int, bold_first: bool = False) -> str:
    lh = size + 3
    y0 = cy - (len(lines) - 1) * lh / 2 + size / 3
    return "".join(text(cx, y0 + i * lh, ln, size, bold=bold_first and i == 0) for i, ln in enumerate(lines))


def box(x: float, y: float, w: float, h: float, *lines: str, size: int = 12, fill: str = "#fff",
        bold_first: bool = False) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#000"/>'
            + _lines(x + w / 2, y + h / 2, lines, size, bold_first))


def bubble(cx: float, cy: float, rx: float, ry: float, *lines: str, size: int = 12) -> str:
    return (f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#fff" stroke="#000"/>'
            + _lines(cx, cy, lines, size))


def store(x: float, y: float, w: float, label: str, size: int = 12) -> str:
    return (f'<line x1="{x}" y1="{y}" x2="{x + w}" y2="{y}" stroke="#000"/>'
            f'<line x1="{x}" y1="{y + 24}" x2="{x + w}" y2="{y + 24}" stroke="#000"/>'
            + text(x + w / 2, y + 16, label, size))


def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "", dx: float = 0, dy: float = -4,
          size: int = 11, dash: bool = False, both: bool = False) -> str:
    d = ' stroke-dasharray="5 4"' if dash else ""
    s = ' marker-start="url(#ar)"' if both else ""
    out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#000" marker-end="url(#ar)"{d}{s}/>'
    if label:
        out += text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, size)
    return out


def line(x1: float, y1: float, x2: float, y2: float, label: str = "", dx: float = 0, dy: float = -4,
         size: int = 11, dash: bool = False) -> str:
    d = ' stroke-dasharray="5 4"' if dash else ""
    out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#000"{d}/>'
    if label:
        out += text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, size)
    return out


def poly(points: list[tuple[float, float]], label: str = "", lx: float = 0, ly: float = 0, size: int = 11) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    out = f'<polyline points="{pts}" fill="none" stroke="#000" marker-end="url(#ar)"/>'
    if label:
        out += text(lx, ly, label, size)
    return out


def actor(x: float, y: float, *name: str) -> str:
    """Stick figure with its head top at y."""
    body = (f'<circle cx="{x}" cy="{y + 9}" r="9" fill="#fff" stroke="#000"/>'
            f'<line x1="{x}" y1="{y + 18}" x2="{x}" y2="{y + 44}" stroke="#000"/>'
            f'<line x1="{x - 15}" y1="{y + 28}" x2="{x + 15}" y2="{y + 28}" stroke="#000"/>'
            f'<line x1="{x}" y1="{y + 44}" x2="{x - 12}" y2="{y + 62}" stroke="#000"/>'
            f'<line x1="{x}" y1="{y + 44}" x2="{x + 12}" y2="{y + 62}" stroke="#000"/>')
    return body + "".join(text(x, y + 77 + 14 * i, n, 12) for i, n in enumerate(name))


def uml_class(x: float, y: float, w: float, name: str, attrs: list[str], size: int = 11) -> str:
    """Two-compartment class box; returns the SVG. Height = 22 + 14 per attribute + 6."""
    h = 22 + 14 * len(attrs) + 6
    out = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff" stroke="#000"/>'
           f'<line x1="{x}" y1="{y + 22}" x2="{x + w}" y2="{y + 22}" stroke="#000"/>'
           + text(x + w / 2, y + 15, name, 12, bold=True))
    for i, a in enumerate(attrs):
        out += text(x + 5, y + 36 + 14 * i, a, size, anchor="start")
    return out


# ---- output -----------------------------------------------------------------------------------

def page(title: str, body: str) -> str:
    return f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>'


def to_pdf(html: str, name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    src = OUT / "se" / f"{Path(name).stem}.html"
    src.write_text(html, encoding="utf-8")
    pdf = OUT / name
    footer = ('<div style="width:100%;font-size:9px;text-align:center;font-family:serif;">'
              '<span class="pageNumber"></span></div>')
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(src.as_uri())
        pg.pdf(path=str(pdf), format="A4", print_background=True, display_header_footer=True,
               header_template="<span></span>", footer_template=footer, prefer_css_page_size=True)
        b.close()
    print(f"-> {pdf}")
    return pdf
