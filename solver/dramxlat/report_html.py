"""Generate a self-contained HTML report from a real solver run.

Running the toolkit on your own RAM produces a timing dataset; this turns the
solver's output into a single offline HTML file you can open in a browser —
no build step, no network, no dependencies at view time. It shows:

  * run metadata + headline numbers (KPIs),
  * the measured bimodal latency distribution (the signal),
  * how your physical address decodes (which bits select a bank vs. row/column),
  * the recovered XOR functions and their bit structure,
  * whether the three independent solver routes agree (a confidence check),
  * optionally, a single-bit sweep (from `dram_probe sweep`) showing row bits.

Unlike the synthetic demo in docs/, this carries no ground-truth labels: on real
hardware you recover the bank-selecting XOR functions, but which one is "rank" vs
"bank group" needs correlating with the DIMM topology, so functions are shown
generically.
"""
from __future__ import annotations
import datetime
import json
import math
import os
from typing import Optional
import numpy as np

from . import gf2, report as _report


# ----------------------------------------------------------------- data build

def _histogram(latency: np.ndarray, cls) -> dict:
    mu_h, mu_c, fc = cls.mu_hit, cls.mu_conflict, cls.fault_cutoff
    lo = float(max(float(latency.min()), mu_h - 0.6 * max(mu_c - mu_h, 1.0)))
    if math.isfinite(fc):
        hi = float(min(fc * 1.03, float(np.percentile(latency, 99.9))))
    else:
        hi = float(np.percentile(latency, 99.5))
    if not (hi > lo):
        lo, hi = float(latency.min()), float(latency.max()) or (lo + 1)
    nb = 80
    main = latency[(latency >= lo) & (latency <= hi)]
    counts, edges = np.histogram(main, bins=nb, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = [{"x": round(float(c), 1), "n": int(k)} for c, k in zip(centers, counts)]
    return {"hist": hist, "hist_range": [round(lo, 1), round(hi, 1)],
            "y_max": int(max(int(counts.max()) if counts.size else 1, 1))}


def _sweep(sweep_timing, func_bits: set, threshold: float) -> Optional[list]:
    """Pull single-bit toggles (popcount(delta)==1) out of a `sweep` dataset."""
    if sweep_timing is None:
        return None
    delta = np.asarray(sweep_timing.delta, dtype=np.uint64)
    lat = np.asarray(sweep_timing.latency, dtype=float)
    per_bit: dict[int, list] = {}
    for d, l in zip(delta.tolist(), lat.tolist()):
        if d and (d & (d - 1)) == 0:          # exactly one bit set
            per_bit.setdefault(d.bit_length() - 1, []).append(l)
    if not per_bit:
        return None
    out = []
    for b in sorted(per_bit):
        med = float(np.median(per_bit[b]))
        if med >= threshold:
            role = "row"
        elif b in func_bits:
            role = "bank"
        else:
            role = "column"
        out.append({"bit": b, "latency": round(med, 1), "role": role})
    return out


def report_data(timing, result, *, sweep_timing=None,
                label: Optional[str] = None, dataset: Optional[str] = None) -> dict:
    cls = result.classification
    lat = np.asarray(timing.latency, dtype=float)

    funcs = []
    for i, m in enumerate(gf2.rref(result.functions)):
        funcs.append({"index": i, "bits": gf2.bits_of(m),
                      "expr": _report.mask_to_str(m), "hex": "0x%X" % m})
    func_bits = set()
    for f in funcs:
        func_bits.update(f["bits"])
    cols = set(result.cols)

    hi_bit = min(max([*(cols or {0}), *(func_bits or {0}), 33]), 47)
    bit_roles = []
    for b in range(0, hi_bit + 1):
        if b < 6:
            r = "offset"
        elif b in func_bits:
            r = "bank"
        elif b in cols:
            r = "rowcol"
        else:
            r = "fixed"
        bit_roles.append({"bit": b, "role": r})

    candidates = {}
    for name, bs in result.candidates.items():
        candidates[name] = {"dim": bs.dim,
                            "consistency": round(bs.consistency, 4),
                            "separation": round(bs.separation, 4),
                            "score": round(bs.score, 4),
                            "selected": name == result.method}

    # agreement: how many distinct routes independently span the chosen map
    agree = 0
    if result.functions:
        for name, bs in result.candidates.items():
            if name == "combined":
                continue
            if bs.masks and gf2.same_span(bs.masks, result.functions):
                agree += 1

    data = {
        "meta": {
            "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "label": label or "",
            "dataset": os.path.basename(dataset) if dataset else "",
            "selected": result.method,
            "routes_agree": agree,
        },
        "classifier": {
            "mu_hit": round(float(cls.mu_hit), 1),
            "mu_conflict": round(float(cls.mu_conflict), 1),
            "threshold": round(float(cls.threshold), 1),
            "fault_cutoff": (None if not math.isfinite(cls.fault_cutoff)
                             else round(float(cls.fault_cutoff), 1)),
            "separation": round(float(cls.separation), 2),
            "n_fault": int(cls.n_fault),
            "n_conflict": int(result.n_conflict),
            "n_total": int(result.n_total),
            "method": cls.method,
        },
        "funcs": funcs,
        "num_banks": (2 ** len(funcs)) if funcs else 0,
        "bit_roles": bit_roles,
        "candidates": candidates,
        "weak": float(cls.separation) < 2.0,
        "sweep": _sweep(sweep_timing, func_bits, float(cls.threshold)),
    }
    data.update(_histogram(lat, cls))
    return data


def write_report(path: str, timing, result, *, sweep_timing=None,
                 label: Optional[str] = None, dataset: Optional[str] = None) -> dict:
    data = report_data(timing, result, sweep_timing=sweep_timing,
                        label=label, dataset=dataset)
    html = _PAGE.replace("/*__DATA__*/", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return data


# ------------------------------------------------------------------- template

_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>DRAM Map Report</title>
<style>
  /* Layout: a column of instrument panels; validated data-viz palette tokens. */
  :root{
    color-scheme: light;
    --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10); --border2:rgba(11,11,11,.06);
    --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; --violet:#4a3aa7;
    --good:#0ca30c; --warn:#b0820f; --critical:#d03b3b;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
    --r:12px;
  }
  @media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#8f8d86;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.12); --border2:rgba(255,255,255,.06);
    --blue:#3987e5; --orange:#e2693a; --aqua:#22b589; --violet:#9085e9;
    --good:#0ca30c; --warn:#e0aa2a; --critical:#e05a5a;
  }}
  :root[data-theme="dark"]{
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#8f8d86;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.12); --border2:rgba(255,255,255,.06);
    --blue:#3987e5; --orange:#e2693a; --aqua:#22b589; --violet:#9085e9;
    --good:#0ca30c; --warn:#e0aa2a; --critical:#e05a5a;
  }
  *{box-sizing:border-box} html,body{margin:0}
  body{background:var(--plane); color:var(--ink); font-family:var(--sans); line-height:1.5;
       -webkit-font-smoothing:antialiased}
  img{max-width:100%} [hidden]{display:none!important}
  .wrap{max-width:1000px; margin:0 auto; padding-inline:16px; padding-block:28px 56px;
        display:flex; flex-direction:column; gap:18px}
  .eyebrow{font-family:var(--mono); font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted)}
  h1{font-size:clamp(24px,5vw,34px); line-height:1.1; margin:0; letter-spacing:-.02em; text-wrap:balance}
  .meta{color:var(--ink2); font-size:13px; font-family:var(--mono)}
  .toolbar{display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:2px}
  .btn{font:inherit; font-size:13px; color:var(--ink2); background:var(--surface);
       border:1px solid var(--border); border-radius:999px; padding:6px 12px; cursor:pointer}
  .btn:focus-visible{outline:2px solid var(--blue); outline-offset:2px}

  .banner{border:1px solid var(--warn); background:color-mix(in srgb,var(--warn) 12%, transparent);
          border-radius:var(--r); padding:12px 14px; font-size:13.5px; color:var(--ink)}
  .banner b{color:var(--warn)}

  .kpis{display:grid; grid-template-columns:repeat(6,1fr); gap:10px}
  @media (max-width:820px){ .kpis{grid-template-columns:repeat(3,1fr)} }
  @media (max-width:480px){ .kpis{grid-template-columns:repeat(2,1fr)} }
  .kpi{background:var(--surface); border:1px solid var(--border); border-radius:var(--r);
       padding:12px 12px 10px; display:flex; flex-direction:column; gap:2px; min-width:0}
  .kpi .v{font-size:23px; font-weight:650; letter-spacing:-.02em; font-variant-numeric:tabular-nums}
  .kpi .l{font-size:11.5px; color:var(--muted)}
  .kpi.accent{border-top:3px solid var(--blue)}

  .card{background:var(--surface); border:1px solid var(--border); border-radius:var(--r);
        padding:18px 18px 16px; display:flex; flex-direction:column; gap:6px; min-width:0}
  .card h2{font-size:17px; margin:0; letter-spacing:-.01em}
  .card .sub{font-size:13.5px; color:var(--ink2); margin:0 0 6px; max-width:72ch}
  .plot{width:100%; overflow-x:auto}
  svg{display:block; width:100%; height:auto} svg text{font-family:var(--sans)}
  .ax{fill:var(--muted); font-size:11px} .axname{fill:var(--ink2); font-size:11.5px}
  .gl{stroke:var(--grid); stroke-width:1} .baseline{stroke:var(--axis); stroke-width:1}
  .mk{stroke-dasharray:3 3; stroke-width:1.3}
  .mklab{font-size:10.5px; font-family:var(--mono); paint-order:stroke;
         stroke:var(--surface); stroke-width:3px; stroke-linejoin:round}
  .legend{display:flex; gap:14px; flex-wrap:wrap; font-size:12.5px; color:var(--ink2); margin-top:2px}
  .legend span{display:inline-flex; align-items:center; gap:6px}
  .sw{width:11px; height:11px; border-radius:3px; flex:none}
  .cap{font-size:12px; color:var(--muted); margin:6px 0 0; max-width:74ch}

  .strip{display:flex; gap:2px; min-width:760px}
  .cell{flex:1; min-width:18px; aspect-ratio:1/1.35; border-radius:4px; display:flex;
        align-items:center; justify-content:center; font-family:var(--mono); font-size:11px; color:#fff}
  .cell.offset{background:var(--muted)} .cell.rowcol{background:var(--orange)}
  .cell.bank{background:var(--blue)} .cell.fixed{background:var(--border); color:var(--muted)}
  .grouprow{display:flex; gap:2px; min-width:760px; margin-top:4px}
  .grp{display:flex; align-items:center; justify-content:center; font-size:11px; color:var(--ink2);
       border-top:2px solid var(--border); padding-top:4px; text-align:center}

  .funcs{display:flex; flex-direction:column; gap:8px; margin-top:4px}
  .fn{display:flex; align-items:center; gap:12px; flex-wrap:wrap; padding:9px 12px;
      border:1px solid var(--border2); border-radius:9px; background:var(--plane)}
  .fn .dot{width:10px; height:10px; border-radius:3px; flex:none; background:var(--blue)}
  .fn .nm{font-size:13px; font-weight:600; min-width:92px; color:var(--ink2)}
  .fn .ex{font-family:var(--mono); font-size:13px; color:var(--ink)}
  .fn .hx{font-family:var(--mono); font-size:12px; color:var(--muted); margin-left:auto}
  .note{font-size:12.5px; color:var(--ink2)}

  .mx{display:grid; gap:3px; font-family:var(--mono); font-size:11px; min-width:380px}
  .mx .h{color:var(--muted); text-align:center; padding-bottom:2px}
  .mx .rl{color:var(--ink2); text-align:right; padding-right:8px; white-space:nowrap}
  .mcell{aspect-ratio:1; border-radius:4px; border:1px solid var(--border2); background:var(--plane)}
  .mcell.on{background:var(--blue); border-color:transparent}

  table.routes{border-collapse:collapse; width:100%; font-size:13px; min-width:420px}
  table.routes th,table.routes td{text-align:left; padding:7px 10px; border-bottom:1px solid var(--border2)}
  table.routes th{color:var(--muted); font-weight:600; font-size:11.5px; text-transform:uppercase; letter-spacing:.04em}
  table.routes td{font-variant-numeric:tabular-nums}
  table.routes tr.sel td{background:color-mix(in srgb,var(--blue) 10%, transparent)}
  .pill{display:inline-block; font-size:11px; font-family:var(--mono); padding:1px 7px; border-radius:999px;
        background:color-mix(in srgb,var(--good) 18%, transparent); color:var(--good)}

  #tt{position:fixed; z-index:50; pointer-events:none; opacity:0; transition:opacity .08s;
      background:var(--ink); color:var(--surface); font-size:12px; font-family:var(--mono);
      padding:6px 9px; border-radius:7px; max-width:240px; box-shadow:0 4px 16px rgba(0,0,0,.25)}
  footer{color:var(--muted); font-size:12.5px; border-top:1px solid var(--border); padding-top:14px}
  a{color:var(--blue)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">dram_xlat · measurement report</div>
    <h1>Your DRAM address map</h1>
    <div class="meta" id="meta"></div>
    <div class="toolbar"><button class="btn" id="theme">Toggle theme</button></div>
  </header>

  <div class="banner" id="weak" hidden></div>

  <section class="kpis" id="kpis" aria-label="headline results"></section>

  <section class="card">
    <h2>Measured latency distribution</h2>
    <p class="sub">Same-bank/different-row accesses collide in the row buffer and run slower than
      accesses served in parallel from different banks. A clean split means the measurement is trustworthy.</p>
    <div class="plot"><svg id="hist" viewBox="0 0 760 300" role="img" aria-label="latency histogram"></svg></div>
    <div class="legend">
      <span><i class="sw" style="background:var(--blue)"></i>fast (hit / different bank)</span>
      <span><i class="sw" style="background:var(--orange)"></i>slow (row-buffer conflict)</span>
    </div>
    <p class="cap" id="hist-cap"></p>
  </section>

  <section class="card" id="sec-bits">
    <h2>How your physical address decodes</h2>
    <p class="sub">Bits the solver found to select a bank/rank/group are confident. The remaining varying
      bits are row or column (this method recovers the bank functions, not the row/column split); low bits
      are the byte offset within a cache line; "fixed" bits never varied in this dataset.</p>
    <div class="plot"><div class="strip" id="strip"></div><div class="grouprow" id="groups"></div></div>
    <div class="legend" style="margin-top:10px">
      <span><i class="sw" style="background:var(--muted)"></i>byte offset</span>
      <span><i class="sw" style="background:var(--blue)"></i>bank / rank / group</span>
      <span><i class="sw" style="background:var(--orange)"></i>row / column</span>
      <span><i class="sw" style="background:var(--border)"></i>fixed (unobserved)</span>
    </div>
  </section>

  <section class="card" id="sec-funcs">
    <h2>Recovered XOR functions</h2>
    <p class="sub">Each is a GF(2) function the memory controller XORs to pick a bank. Labels (rank vs bank
      group) need correlating with your DIMM topology, so they are shown generically.</p>
    <div class="funcs" id="funcs"></div>
    <p class="note" id="banks-note" style="margin-top:4px"></p>
  </section>

  <section class="card" id="sec-matrix">
    <h2>XOR structure</h2>
    <p class="sub">Which address bits (columns) each recovered function (rows) combines.</p>
    <div class="plot"><div class="mx" id="matrix"></div></div>
  </section>

  <section class="card" id="sec-routes">
    <h2>Solver agreement</h2>
    <p class="sub">Three independent methods try to recover the map. When they agree, confidence is high.</p>
    <div class="plot"><table class="routes" id="routes"></table></div>
    <p class="cap" id="routes-cap"></p>
  </section>

  <section class="card" id="sec-sweep" hidden>
    <h2>Single-bit sweep</h2>
    <p class="sub">Toggling one address bit at a time: a tall (slow) bar is a pure row bit (same bank,
      different row); bank and column bits stay fast.</p>
    <div class="plot"><svg id="sweep" viewBox="0 0 760 260" role="img" aria-label="single-bit sweep"></svg></div>
    <div class="legend">
      <span><i class="sw" style="background:var(--blue)"></i>bank bit</span>
      <span><i class="sw" style="background:var(--aqua)"></i>column bit</span>
      <span><i class="sw" style="background:var(--orange)"></i>row bit</span>
    </div>
  </section>

  <footer>
    Generated by <a href="https://github.com/willroll/dram_xlat" target="_blank" rel="noopener">dram_xlat</a>.
    Method: row-buffer-conflict timing (Pessl et al., “DRAMA”, USENIX 2016). All figures are from your run.
  </footer>
</div>
<div id="tt" role="status" aria-live="polite"></div>

<script id="data" type="application/json">/*__DATA__*/</script>
<script>
(function(){
  "use strict";
  var D; try{ D = JSON.parse(document.getElementById("data").textContent); }catch(e){ D=null; }
  if(!D) return;
  var SVGNS="http://www.w3.org/2000/svg";
  var fmt=function(n){ return (n==null)?"—":Number(n).toLocaleString("en-US"); };
  function css(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
  function mk(t,a,p){ var e=document.createElementNS(SVGNS,t); for(var k in a)e.setAttribute(k,a[k]); if(p)p.appendChild(e); return e; }
  function clear(el){ while(el.firstChild) el.removeChild(el.firstChild); }
  var tt=document.getElementById("tt");
  function tip(h,ev){ tt.innerHTML=h; tt.style.opacity=1; var x=ev.clientX+14,y=ev.clientY+14; if(x>innerWidth-180)x=ev.clientX-170; tt.style.left=x+"px"; tt.style.top=y+"px"; }
  function untip(){ tt.style.opacity=0; }
  function hov(el,html){ el.addEventListener("mouseenter",function(e){tip(html,e);}); el.addEventListener("mousemove",function(e){tip(html,e);}); el.addEventListener("mouseleave",untip); }

  try{ var s=localStorage.getItem("dramrep-theme"); if(s)document.documentElement.setAttribute("data-theme",s);}catch(e){}
  document.getElementById("theme").addEventListener("click",function(){
    var cur=document.documentElement.getAttribute("data-theme");
    var m=window.matchMedia("(prefers-color-scheme: dark)");
    var next=(cur?cur:(m.matches?"dark":"light"))==="dark"?"light":"dark";
    document.documentElement.setAttribute("data-theme",next);
    try{localStorage.setItem("dramrep-theme",next);}catch(e){}
    render();
  });

  var C=D.classifier, M=D.meta;
  var metaBits=[];
  if(M.label) metaBits.push(M.label);
  if(M.dataset) metaBits.push(M.dataset);
  metaBits.push(fmt(C.n_total)+" pairs"); metaBits.push(M.generated);
  document.getElementById("meta").textContent=metaBits.join("  ·  ");

  if(D.weak){
    var w=document.getElementById("weak"); w.hidden=false;
    w.innerHTML="<b>Weak separation ("+C.separation.toFixed(1)+"σ).</b> The two latency modes barely "+
      "split, so recovery may be unreliable. Increase --reps, pin an isolated core, and disable prefetchers, "+
      "then re-measure.";
  }

  // KPIs
  var confPct = C.n_total? (100*C.n_conflict/C.n_total):0;
  var kpis=[
    {v:C.separation.toFixed(1)+"σ", l:"mode separation", accent:true},
    {v:fmt(C.n_conflict), l:"conflict pairs"},
    {v:confPct.toFixed(1)+"%", l:"of all pairs"},
    {v:D.funcs.length, l:"functions recovered"},
    {v:D.num_banks, l:"banks resolved"},
    {v:fmt(C.n_fault), l:"faults trimmed"}
  ];
  var kw=document.getElementById("kpis"); clear(kw);
  kpis.forEach(function(k){ var d=document.createElement("div"); d.className="kpi"+(k.accent?" accent":"");
    d.innerHTML='<div class="v"></div><div class="l"></div>';
    d.querySelector(".v").textContent=k.v; d.querySelector(".l").textContent=k.l; kw.appendChild(d); });

  function histogram(){
    var svg=document.getElementById("hist"); clear(svg);
    var W=760,H=300,L=48,R=14,T=16,B=36,pw=W-L-R,ph=H-T-B,by=T+ph;
    var x0=D.hist_range[0],x1=D.hist_range[1]; if(x1<=x0)x1=x0+1;
    var xs=function(v){return L+(v-x0)/(x1-x0)*pw;};
    var ymax=Math.max(D.y_max,10), lym=Math.log10(ymax);
    var ys=function(n){ if(n<1)return by; return by-(Math.log10(n)/lym)*ph; };
    var ticks=[1,10,100,1000,10000,100000].filter(function(t){return t<=ymax*1.2;});
    ticks.forEach(function(t){ var y=ys(t); mk("line",{x1:L,y1:y,x2:L+pw,y2:y,class:"gl"},svg);
      mk("text",{x:L-6,y:y+3,"text-anchor":"end",class:"ax"},svg).textContent=t>=1000?(t/1000)+"k":t; });
    var xt=4; for(var i=0;i<=xt;i++){ var v=x0+(x1-x0)*i/xt; mk("text",{x:xs(v),y:by+16,"text-anchor":"middle",class:"ax"},svg).textContent=Math.round(v); }
    mk("text",{x:L+pw/2,y:H-4,"text-anchor":"middle",class:"axname"},svg).textContent="access latency (CPU cycles)";
    mk("line",{x1:L,y1:by,x2:L+pw,y2:by,class:"baseline"},svg);
    var thr=C.threshold, bw=Math.max(1,(pw*(D.hist[1]?(D.hist[1].x-D.hist[0].x):5)/(x1-x0))-1.2);
    D.hist.forEach(function(d){ if(d.n<=0)return; var col=d.x<thr?css("--blue"):css("--orange"); var y=ys(d.n);
      var r=mk("rect",{x:xs(d.x)-bw/2,y:y,width:bw,height:by-y,fill:col,rx:1.3},svg);
      hov(r,"≈"+d.x+" cyc<br>"+fmt(d.n)+" accesses"); });
    function marker(v,label,color,al){ if(v==null)return; var x=xs(v); if(x<L||x>L+pw)return;
      mk("line",{x1:x,y1:T,x2:x,y2:by,class:"mk",stroke:color},svg);
      mk("text",{x:al==="end"?x-4:x+4,y:T+10,"text-anchor":al,class:"mklab",fill:color},svg).textContent=label; }
    marker(C.mu_hit,"hit "+Math.round(C.mu_hit),css("--blue"),"start");
    marker(C.threshold,"split "+Math.round(C.threshold),css("--ink2"),"start");
    marker(C.mu_conflict,"conflict "+Math.round(C.mu_conflict),css("--orange"),"end");
    marker(C.fault_cutoff,"faults →",css("--critical"),"end");
    var cap="Hit ≈ "+Math.round(C.mu_hit)+" cyc, conflict ≈ "+Math.round(C.mu_conflict)+" cyc ("+C.separation.toFixed(1)+"σ apart).";
    if(C.fault_cutoff!=null) cap+=" "+fmt(C.n_fault)+" fault-scale accesses above "+Math.round(C.fault_cutoff)+" cyc were trimmed.";
    cap+=" Log scale.";
    document.getElementById("hist-cap").textContent=cap;
  }

  function strip(){
    var s=document.getElementById("strip"); clear(s);
    var order=D.bit_roles.slice().reverse();
    order.forEach(function(d){ var c=document.createElement("div"); c.className="cell "+d.role; c.textContent=d.bit;
      hov(c,"bit "+d.bit+" · "+({offset:"byte offset",bank:"bank/rank/group",rowcol:"row/column",fixed:"fixed (unobserved)"}[d.role])); s.appendChild(c); });
    var g=document.getElementById("groups"); clear(g);
    var groups=[],cur=null;
    order.forEach(function(d){ if(!cur||cur.role!==d.role){cur={role:d.role,bits:[d.bit]};groups.push(cur);} else cur.bits.push(d.bit); });
    var lab={offset:"byte offset",bank:"bank/rank/group",rowcol:"row/column",fixed:"fixed"};
    groups.forEach(function(gr){ var el=document.createElement("div"); el.className="grp"; el.style.flex=gr.bits.length+" "+gr.bits.length+" 0";
      var hi=Math.max.apply(null,gr.bits),lo=Math.min.apply(null,gr.bits);
      el.textContent=lab[gr.role]+(hi===lo?" ("+hi+")":" ("+hi+"–"+lo+")"); g.appendChild(el); });
  }

  function funcs(){
    var f=document.getElementById("funcs"); clear(f);
    if(!D.funcs.length){ document.getElementById("sec-funcs").hidden=false;
      f.innerHTML='<div class="note">No functions were recovered — the latency modes likely did not separate cleanly. See the warning above.</div>';
      document.getElementById("sec-matrix").hidden=true; return; }
    D.funcs.forEach(function(fn){ var row=document.createElement("div"); row.className="fn";
      row.innerHTML='<span class="dot"></span><span class="nm">function '+fn.index+'</span><span class="ex"></span><span class="hx"></span>';
      row.querySelector(".ex").textContent=fn.expr; row.querySelector(".hx").textContent=fn.hex; f.appendChild(row); });
    document.getElementById("banks-note").innerHTML="<b>"+D.funcs.length+"</b> independent functions → up to <b>2<sup>"+D.funcs.length+"</sup> = "+D.num_banks+"</b> distinct banks addressable.";
  }

  function matrix(){
    var host=document.getElementById("matrix"); clear(host);
    if(!D.funcs.length){ document.getElementById("sec-matrix").hidden=true; return; }
    var fns=D.funcs.slice().reverse();
    var minb=99,maxb=0; fns.forEach(function(fn){ fn.bits.forEach(function(b){ if(b<minb)minb=b; if(b>maxb)maxb=b; }); });
    var cols=[]; for(var b=minb;b<=maxb;b++) cols.push(b);
    host.style.gridTemplateColumns="minmax(84px,auto) repeat("+cols.length+",1fr)";
    host.appendChild(Object.assign(document.createElement("div"),{className:"h"}));
    cols.forEach(function(b){ var h=document.createElement("div"); h.className="h"; h.textContent="b"+b; host.appendChild(h); });
    fns.forEach(function(fn){ var rl=document.createElement("div"); rl.className="rl"; rl.textContent="f"+fn.index; host.appendChild(rl);
      cols.forEach(function(b){ var c=document.createElement("div"); var on=fn.bits.indexOf(b)>=0; c.className="mcell"+(on?" on":"");
        if(on) hov(c,"function "+fn.index+" uses bit "+b); host.appendChild(c); }); });
  }

  function routes(){
    var t=document.getElementById("routes"); clear(t);
    var head=document.createElement("tr");
    ["route","functions","consistency","separation","selected"].forEach(function(h){ var th=document.createElement("th"); th.textContent=h; head.appendChild(th); });
    t.appendChild(head);
    var labels={"exact-nullspace":"exact null-space","robust-nullspace":"RANSAC null-space","genetic":"genetic search","combined":"combined"};
    Object.keys(D.candidates).forEach(function(name){ var c=D.candidates[name]; var tr=document.createElement("tr"); if(c.selected)tr.className="sel";
      var cells=[labels[name]||name, c.dim, c.consistency.toFixed(3), c.separation.toFixed(3), c.selected?"✓":""];
      cells.forEach(function(v,i){ var td=document.createElement("td"); if(i===4&&c.selected){ td.innerHTML='<span class="pill">selected</span>'; } else td.textContent=v; tr.appendChild(td); });
      t.appendChild(tr); });
    var a=D.meta.routes_agree;
    document.getElementById("routes-cap").textContent = D.funcs.length ?
      (a>=2 ? a+" independent methods recovered the same map — high confidence." :
              "Only the "+(labels[D.meta.selected]||D.meta.selected)+" route recovered this map; treat with some caution.")
      : "No route recovered a map.";
  }

  function sweep(){
    if(!D.sweep||!D.sweep.length){ document.getElementById("sec-sweep").hidden=true; return; }
    document.getElementById("sec-sweep").hidden=false;
    var svg=document.getElementById("sweep"); clear(svg);
    var W=760,H=260,L=48,R=14,T=16,B=40,pw=W-L-R,ph=H-T-B,by=T+ph;
    var lats=D.sweep.map(function(d){return d.latency;});
    var y0=Math.min.apply(null,lats)*0.95, y1=Math.max.apply(null,lats)*1.05; if(y1<=y0)y1=y0+1;
    var ys=function(v){return by-(v-y0)/(y1-y0)*ph;};
    [y0,(y0+y1)/2,y1].forEach(function(t){ var y=ys(t); mk("line",{x1:L,y1:y,x2:L+pw,y2:y,class:"gl"},svg);
      mk("text",{x:L-6,y:y+3,"text-anchor":"end",class:"ax"},svg).textContent=Math.round(t); });
    var n=D.sweep.length, step=pw/n, bw=Math.min(step-3,20);
    var rc={bank:"--blue",column:"--aqua",row:"--orange"};
    D.sweep.forEach(function(d,i){ var cx=L+step*(i+0.5),y=ys(d.latency),col=css(rc[d.role]);
      var r=mk("rect",{x:cx-bw/2,y:y,width:bw,height:by-y,fill:col,rx:2},svg); hov(r,"bit "+d.bit+" ("+d.role+")<br>≈"+Math.round(d.latency)+" cyc");
      if(i%Math.ceil(n/14)===0) mk("text",{x:cx,y:by+14,"text-anchor":"middle",class:"ax"},svg).textContent=d.bit; });
    mk("line",{x1:L,y1:by,x2:L+pw,y2:by,class:"baseline"},svg);
    mk("text",{x:L+pw/2,y:H-4,"text-anchor":"middle",class:"axname"},svg).textContent="physical address bit toggled";
    mk("text",{x:L-38,y:T+ph/2,"text-anchor":"middle",class:"axname",transform:"rotate(-90 "+(L-38)+" "+(T+ph/2)+")"},svg).textContent="median latency (cyc)";
  }

  function render(){ histogram(); strip(); funcs(); matrix(); routes(); sweep(); }
  render();
})();
</script>
</body>
</html>
"""
