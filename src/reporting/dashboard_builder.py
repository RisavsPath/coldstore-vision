"""Outputs 2, 4, 5: audit report (HTML + manifest.json), rack digital twin, orthomosaic curtain slider.

Every HTML file is self-contained (inline CSS/JS, images as base64), so it opens offline from disk.
"""
from __future__ import annotations

import base64
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np

STATUS_HEX = {"NORMAL": "#2e9e4f", "WARNING": "#e0a800", "CRITICAL": "#d63a3a", "MISSING": "#14b8c8",
              "NEW": "#c14fd1", "UNVERIFIED": "#8a8f98"}
SEV = {"NORMAL": 0, "UNVERIFIED": 1, "NEW": 1, "WARNING": 2, "CRITICAL": 3, "MISSING": 4}

BASE_CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1b1f24;--muted:#5f6b7a;--line:#e3e6ea;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#1a1f25;--ink:#e8ebef;--muted:#9aa5b1;--line:#2a3139;--accent:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 10px}.muted{color:var(--muted)}
main{padding:16px 24px;max-width:1500px;margin:auto}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;color:#fff}
"""


def b64img(img: np.ndarray | None, quality: int = 82) -> str:
    if img is None:
        return ""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def pill(status: str) -> str:
    return f'<span class="pill" style="background:{STATUS_HEX.get(status, "#888")}">{status}</span>'


def _fmt(v, spec="{:+.1f}", suffix=""):
    return "–" if v is None else spec.format(v) + suffix


def sack_rows(insp) -> list[dict]:
    rows = []
    for s in insp.sacks:
        m = s.metrics
        rows.append({
            "sack_id": s.sack_id, "slot": s.slot, "bay": s.bay, "tier": s.tier, "status": s.status,
            "delta_hue_rel_deg": m.get("hue_rel_deg"), "delta_glcm_contrast": m.get("glcm_contrast_delta"),
            "delta_glcm_homogeneity": m.get("glcm_homog_delta"), "delta_z_m": m.get("depth_delta_m"),
            "flow_y_px": m.get("flow_vy"), "top_drop_px": m.get("top_drop_px"),
            "necrosis_d1": m.get("necrosis_d1"), "necrosis_d2": m.get("necrosis_d2"),
            "diff_area_frac": m.get("diff_area_frac"), "match_iou": m.get("match_iou"),
            "reasoning": s.reasoning, "action": s.action,
            "keyframe_day1": int(s.obs.kf), "mosaic_box": [round(float(v), 1) for v in s.obs.mosaic_box],
        })
    return rows


def write_manifest(insp, out: Path) -> Path:
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector": insp.detector_source,
        "keyframes": {"day1": len(insp.kf1), "day2": len(insp.kf2), "registered_pairs": len(insp.pairs)},
        "kpis": insp.kpis,
        "sacks": sack_rows(insp),
    }
    p = out / "manifest.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def write_audit_report(insp, out: Path) -> Path:
    k = insp.kpis
    rows = sorted(insp.sacks, key=lambda s: (-SEV.get(s.status, 0), s.sack_id))
    cards = [("Total monitored", k["total_monitored"], "Day 1 sacks tracked"),
             ("Net inventory change", f"{k['net_inventory_change']:+d}", f"{k['missing']} missing · {k['new']} new"),
             ("Facility health index", f"{k['facility_health_index']:.1f}%", "present sacks, warning = ½"),
             ("Urgent action items", k["urgent_actions"], "critical + missing")]
    counts = k["status_counts"]
    total = max(sum(counts.values()), 1)
    bar = "".join(f'<div title="{s}: {n}" style="flex:{n};background:{STATUS_HEX[s]}"></div>'
                  for s, n in sorted(counts.items(), key=lambda t: -SEV.get(t[0], 0)) if n)
    legend = " ".join(f"{pill(s)} {n}" for s, n in sorted(counts.items(), key=lambda t: -SEV.get(t[0], 0)))
    trs = []
    for s in rows:
        m = s.metrics
        nec = None if m.get("necrosis_d2") is None else (m.get("necrosis_d2", 0) - m.get("necrosis_d1", 0)) * 100
        trs.append(
            f'<tr data-status="{s.status}"><td><b>{s.sack_id}</b><br><span class="muted">{s.slot}</span></td>'
            f"<td>{pill(s.status)}</td>"
            f'<td class="thumbs"><img src="{b64img(s.crop1)}" alt="Day 1"><img src="{b64img(s.crop2)}" alt="Day 2"></td>'
            f"<td>{_fmt(m.get('hue_rel_deg'), suffix='°')}</td>"
            f"<td>{_fmt(None if m.get('glcm_contrast_delta') is None else m['glcm_contrast_delta'] * 100, suffix='%')}</td>"
            f"<td>{_fmt(None if m.get('glcm_homog_delta') is None else m['glcm_homog_delta'] * 100, suffix='%')}</td>"
            f"<td>{_fmt(nec, suffix=' pp')}</td>"
            f"<td>{_fmt(m.get('depth_delta_m'), '{:+.2f}', ' m')}</td>"
            f"<td>{_fmt(m.get('flow_vy'), suffix=' px')}</td>"
            f"<td class='why'>{html.escape('; '.join(m.get('reasons', [])))}</td><td>{html.escape(s.action)}</td></tr>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cold Store Audit</title><style>{BASE_CSS}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.card .v{{font-size:28px;font-weight:700;margin:4px 0}}.bar{{display:flex;height:12px;border-radius:6px;overflow:hidden;margin:14px 0 6px}}
.filters button{{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:5px 10px;border-radius:6px;cursor:pointer;margin:0 4px 6px 0}}
.filters button.on{{border-color:var(--accent);color:var(--accent)}}
.tablewrap{{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;min-width:1100px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-size:12px;color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--panel)}}
.thumbs img{{width:84px;height:84px;object-fit:cover;border-radius:4px;margin-right:4px}}.why{{max-width:340px}}
</style></head><body><header><h1>Cold Storage Comparison Audit</h1>
<div class="muted">Generated {time.strftime('%Y-%m-%d %H:%M')} · detector {html.escape(insp.detector_source)} ·
{len(insp.kf1)} / {len(insp.kf2)} keyframes · {len(insp.pairs)} registered pairs</div></header><main>
<div class="cards">{''.join(f'<div class="card"><div class="muted">{t}</div><div class="v">{v}</div><div class="muted">{d}</div></div>' for t, v, d in cards)}</div>
<div class="bar">{bar}</div><div>{legend}</div>
<h2>Bay anomaly table</h2><div class="filters" id="f"><button class="on" data-s="ALL">All</button>
{''.join(f'<button data-s="{s}">{s.title()}</button>' for s in counts)}</div>
<div class="tablewrap"><table><thead><tr><th>Sack / slot</th><th>Status</th><th>Day 1 · Day 2</th><th>ΔHue (rel)</th>
<th>ΔGLCM contrast</th><th>ΔGLCM homog.</th><th>ΔNecrosis</th><th>ΔZ</th><th>Flow<sub>y</sub></th><th>Root-cause reasoning</th><th>Action required</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div></main>
<script>
document.getElementById('f').addEventListener('click',e=>{{const b=e.target.closest('button');if(!b)return;
document.querySelectorAll('#f button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('tbody tr').forEach(r=>r.style.display=(b.dataset.s==='ALL'||r.dataset.status===b.dataset.s)?'':'none');}});
</script></body></html>"""
    p = out / "audit_report.html"
    p.write_text(page, encoding="utf-8")
    return p


def write_digital_twin(insp, out: Path) -> Path:
    bays = max((s.bay for s in insp.sacks), default=0)
    tiers = max((s.tier for s in insp.sacks), default=0)
    data = []
    for s in insp.sacks:
        r = {k: v for k, v in s.metrics.items() if not isinstance(v, list)}
        data.append({"id": s.sack_id, "slot": s.slot, "bay": s.bay, "tier": s.tier, "status": s.status,
                     "reasoning": s.reasoning, "action": s.action, "metrics": r,
                     "img1": b64img(s.crop1), "img2": b64img(s.crop2)})
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rack Digital Twin</title><style>{BASE_CSS}
.wrap{{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px}}@media (max-width:900px){{.wrap{{grid-template-columns:1fr}}}}
.stage{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;overflow:auto;perspective:1400px}}
.grid{{display:grid;gap:6px;transition:transform .5s;transform-origin:50% 100%}}.grid.three{{transform:rotateX(38deg) rotateZ(-8deg)}}
.cell{{border:2px solid var(--line);border-radius:6px;min-height:70px;padding:5px;display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start;position:relative}}
.cell .lbl{{position:absolute;right:5px;bottom:3px;font-size:10px;color:var(--muted)}}
.sack{{width:26px;height:20px;border-radius:5px;cursor:pointer;border:2px solid transparent}}.sack.sel{{border-color:var(--ink)}}
.axis{{font-size:12px;color:var(--muted);text-align:center;align-self:center}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}
.panel img{{width:48%;border-radius:6px}}table{{width:100%;font-size:12px;border-collapse:collapse}}td{{padding:3px 0;border-bottom:1px solid var(--line)}}
button{{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:5px 10px;border-radius:6px;cursor:pointer}}
</style></head><body><header><h1>Rack Elevation Digital Twin</h1><div class="muted">Bays 1–{bays} along the aisle · Tiers 1–{tiers} (bottom = 1) · click a sack</div></header>
<main><div style="margin-bottom:10px"><button id="v">Toggle 3D view</button> {' '.join(pill(s) for s in STATUS_HEX)}</div>
<div class="wrap"><div class="stage"><div class="grid" id="g"></div></div><div class="panel" id="p"><span class="muted">Select a sack to see Day 1 vs Day 2 imagery and telemetry.</span></div></div></main>
<script>
const D={json.dumps(data)},C={json.dumps(STATUS_HEX)},S={json.dumps(SEV)},B={bays},T={tiers};
const g=document.getElementById('g');g.style.gridTemplateColumns=`40px repeat(${{B}},minmax(110px,1fr))`;
for(let t=T;t>=1;t--){{g.insertAdjacentHTML('beforeend',`<div class="axis">T${{t}}</div>`);
 for(let b=1;b<=B;b++){{const ss=D.filter(d=>d.bay===b&&d.tier===t);const w=ss.reduce((a,d)=>S[d.status]>S[a]?d.status:a,'NORMAL');
  const c=document.createElement('div');c.className='cell';if(ss.length)c.style.borderColor=C[w];c.style.background=ss.length?C[w]+'1f':'transparent';
  c.innerHTML=ss.map(d=>`<div class="sack" title="${{d.id}} ${{d.status}}" data-id="${{d.id}}" style="background:${{C[d.status]}}"></div>`).join('')+`<span class="lbl">B${{b}}·T${{t}}</span>`;g.appendChild(c);}}}}
g.insertAdjacentHTML('beforeend','<div></div>'+Array.from({{length:B}},(_, i)=>`<div class="axis">Bay ${{i+1}}</div>`).join(''));
const f=v=>v===null||v===undefined?'–':(typeof v==='number'?v.toFixed(3):v);
g.addEventListener('click',e=>{{const el=e.target.closest('.sack');if(!el)return;document.querySelectorAll('.sack').forEach(x=>x.classList.toggle('sel',x===el));
 const d=D.find(x=>x.id===el.dataset.id);const m=d.metrics;
 const keys=['hue_rel_deg','glcm_contrast_delta','glcm_homog_delta','necrosis_d1','necrosis_d2','depth_delta_m','flow_vy','flow_div','top_drop_px','diff_area_frac','match_iou'];
 document.getElementById('p').innerHTML=`<h2 style="margin-top:0">${{d.id}} <span class="pill" style="background:${{C[d.status]}}">${{d.status}}</span></h2>
 <div class="muted">${{d.slot}}</div><p>${{d.reasoning}}</p><p><b>Action:</b> ${{d.action}}</p>
 <div>${{d.img1?`<img src="${{d.img1}}" alt="Day 1">`:''}} ${{d.img2?`<img src="${{d.img2}}" alt="Day 2">`:''}}</div>
 <div class="muted" style="font-size:12px">Day 1 · Day 2 (registered, normalised)</div>
 <table>${{keys.filter(k=>k in m).map(k=>`<tr><td>${{k}}</td><td style="text-align:right">${{f(m[k])}}</td></tr>`).join('')}}</table>`;}});
document.getElementById('v').onclick=()=>g.classList.toggle('three');
</script></body></html>"""
    p = out / "rack_digital_twin.html"
    p.write_text(page, encoding="utf-8")
    return p


def write_panoramic_slider(insp, out: Path, display_h: int = 700) -> Path:
    m1, m2 = insp.mosaic1.image, insp.mosaic2.image
    s = min(1.0, display_h / m1.shape[0])
    a = cv2.resize(m1, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.resize(m2, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    w, h = a.shape[1], a.shape[0]
    rects = "".join(
        f'<rect x="{x1 * s:.1f}" y="{y1 * s:.1f}" width="{(x2 - x1) * s:.1f}" height="{(y2 - y1) * s:.1f}" '
        f'stroke="{STATUS_HEX.get(sk.status)}" ><title>{sk.sack_id} {sk.slot} {sk.status}</title></rect>'
        for sk in insp.sacks for (x1, y1, x2, y2) in [sk.obs.mosaic_box])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aisle Curtain Slider</title><style>{BASE_CSS}
.scroller{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:#000}}
.pano{{position:relative;width:{w}px;height:{h}px;user-select:none;touch-action:none}}
.pano img{{position:absolute;inset:0;width:{w}px;height:{h}px;display:block}}#after{{clip-path:inset(0 0 0 50%)}}
.handle{{position:absolute;top:0;bottom:0;width:4px;margin-left:-2px;background:#fff;cursor:ew-resize;box-shadow:0 0 6px #000}}
.handle::after{{content:'⟷';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;color:#000;border-radius:50%;width:34px;height:34px;display:grid;place-items:center;font-weight:700}}
.tag{{position:absolute;top:8px;padding:2px 8px;background:#000a;color:#fff;border-radius:4px;font-size:12px}}
svg{{position:absolute;inset:0;pointer-events:none}}svg rect{{fill:none;stroke-width:2.5;pointer-events:all}}svg.off{{display:none}}
label{{cursor:pointer}}</style></head><body><header><h1>Orthomosaic Curtain Slider</h1>
<div class="muted">Drag the handle across the aisle (scroll sideways for long aisles). Left = Day 1 baseline, right = Day 2 registered.</div></header>
<main><p><label><input type="checkbox" id="ov" checked> Show sack status boxes</label> {' '.join(pill(s) for s in STATUS_HEX)}</p>
<div class="scroller"><div class="pano" id="pano"><img src="{b64img(a, 80)}" alt="Day 1 mosaic"><img id="after" src="{b64img(b, 80)}" alt="Day 2 mosaic">
<svg id="svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{rects}</svg>
<span class="tag" style="left:8px">DAY 1</span><span class="tag" id="t2" style="right:8px">DAY 2</span><div class="handle" id="hd" style="left:50%"></div></div></div></main>
<script>
const pano=document.getElementById('pano'),hd=document.getElementById('hd'),af=document.getElementById('after');let drag=false;
function set(x){{const r=pano.getBoundingClientRect();const p=Math.max(0,Math.min(1,(x-r.left)/r.width));hd.style.left=(p*100)+'%';af.style.clipPath=`inset(0 0 0 ${{p*100}}%)`;}}
hd.addEventListener('pointerdown',e=>{{drag=true;hd.setPointerCapture(e.pointerId);}});
hd.addEventListener('pointermove',e=>{{if(drag)set(e.clientX);}});hd.addEventListener('pointerup',()=>drag=false);
pano.addEventListener('click',e=>{{if(e.target===hd)return;set(e.clientX);}});
document.getElementById('ov').onchange=e=>document.getElementById('svg').classList.toggle('off',!e.target.checked);
</script></body></html>"""
    p = out / "panoramic_slider.html"
    p.write_text(page, encoding="utf-8")
    return p
