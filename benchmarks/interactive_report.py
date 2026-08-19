"""Self-contained *interactive* HTML report for cohort correction runs.

Renders each field as a pan/zoom ``<canvas>`` viewer over its Jacobian-determinant
map (before/after toggle), with a hover-readout tooltip (~3-digit display), a toggleable
displacement-vector overlay, and a click-to-locate region-of-interest table that
ranks the worst fold clusters. When the payload carries a ``traj`` (K sampled Jdet
frames from the solver's iterations — see ``cohort_benchmark._sample_trajectory``),
the viewer also gets a play/scrub timeline that animates how the field deforms as
the solver runs. Everything is inlined (base64 float arrays + one shared vanilla-JS
viewer) so the single .html file is portable — no CDNs, no external assets.

Pure-Python, testable pieces:
- ``b64_floats`` / ``fold_clusters_2d`` / ``fold_clusters_3d`` — data prep.
- ``build_interactive_report`` — assembles the document (never raises).

The viewer JS is a plain string; the Python side only prepares its data.
"""

import base64
import html
import json

import numpy as np
from scipy import ndimage as ndi

_F16_MAX = 65504.0


def b64_floats(arr):
    """Base64 of a little-endian float16 buffer (halves report size; the JS side
    decodes it back to a Float32Array).

    float16 (~3-4 significant digits) is display-only precision — these arrays feed
    the canvas/hover/quiver, never a computation. All reported metrics are computed
    in Python at full precision. Values are clipped to the float16 range so a
    divergent field can't overflow to inf. Use ``b64_uint16`` for exact integer
    coordinates (float16 loses integer exactness above 2048).
    """
    a = np.clip(np.asarray(arr, dtype=np.float64), -_F16_MAX, _F16_MAX)
    return base64.b64encode(a.astype("<f2").tobytes()).decode("ascii")


def b64_uint16(arr):
    """Base64 of a little-endian uint16 buffer — exact for integer pixel coords
    (0..65535). Used for correspondence coordinates (float16 would snap ≥2048)."""
    a = np.clip(np.rint(np.asarray(arr, dtype=np.float64)), 0, 65535).astype("<u2")
    return base64.b64encode(a.tobytes()).decode("ascii")


def _clusters(mask, jac, threshold, top, ndim):
    """Rank connected fold regions in *mask* by severity. Returns list of dicts.

    Severity = negative volume within the cluster (sum of threshold-jdet over the
    cluster's folded voxels); ties broken by voxel count. Centroid keys are
    (y, x) for 2D and (z, y, x) for 3D.
    """
    lbl, n = ndi.label(mask)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    # All per-label stats in one vectorized pass each (O(volume), not O(n*volume)).
    sizes = ndi.sum(mask, lbl, idx)
    mins = ndi.minimum(jac, lbl, idx)
    neg = ndi.sum(np.clip(threshold - jac, 0.0, None), lbl, idx)
    coms = ndi.center_of_mass(mask, lbl, idx)
    slices = ndi.find_objects(lbl)
    out = []
    for i in range(n):
        entry = {
            "size": int(sizes[i]),
            "neg_vol": float(neg[i]),
            "min_jdet": float(mins[i]),
            "bbox": [[int(s.start), int(s.stop)] for s in slices[i]],
        }
        coords = [int(v) for v in np.round(coms[i])]
        if ndim == 2:
            entry["y"], entry["x"] = coords
        else:
            entry["z"], entry["y"], entry["x"] = coords
        out.append(entry)
    out.sort(key=lambda e: (-e["neg_vol"], -e["size"]))
    for rank, e in enumerate(out[:top], 1):
        e["rank"] = rank
    return out[:top]


def fold_clusters_2d(jac2d, threshold, top=25):
    """Ranked fold clusters of a 2D Jacobian map (centroids in y, x)."""
    jac2d = np.asarray(jac2d)
    return _clusters(jac2d < threshold, jac2d, threshold, top, ndim=2)


def fold_clusters_3d(jac3d, threshold, top=25):
    """Ranked fold clusters of a 3D Jacobian volume (centroids in z, y, x)."""
    jac3d = np.asarray(jac3d)
    return _clusters(jac3d < threshold, jac3d, threshold, top, ndim=3)


# ---------------------------------------------------------------------------
# Viewer assets (shared across all field blocks in a report)
# ---------------------------------------------------------------------------

_CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --card:#fafafa;
        --good:#1e7e34; --bad:#c0392b; --accent:#2471a3; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161616; --fg:#e8e8e8; --muted:#9a9a9a; --line:#333; --card:#1f1f1f;
          --good:#4cd07d; --bad:#ff6b5e; --accent:#5aa9e6; } }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); margin:0;
       font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:32px 24px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:0; }
.sub { color:var(--muted); margin:0 0 20px; }
dl.hdr-grid { display:grid; grid-template-columns:auto 1fr; gap:2px 16px; margin:0 0 20px; font-size:14px; }
dl.hdr-grid dt { color:var(--muted); } dl.hdr-grid dd { margin:0; font-variant-numeric:tabular-nums; }
.banner { padding:10px 14px; border-radius:6px; margin:0 0 24px; font-weight:600; }
.banner.ok { background:rgba(30,126,52,.12); color:var(--good); }
.banner.warn { background:rgba(192,57,43,.12); color:var(--bad); }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
thead th { color:var(--muted); font-weight:600; cursor:pointer; }
tr.feasible td.status { color:var(--good); } tr.infeasible td.status { color:var(--bad); }
.field { border:1px solid var(--line); border-radius:8px; padding:14px; margin:18px 0; background:var(--card); }
.field h3 { margin:0 0 10px; font-size:16px; }
.viewer-row { display:flex; gap:16px; flex-wrap:wrap; }
.viewer { position:relative; flex:1 1 460px; min-width:320px; }
.viewer canvas { width:100%; height:auto; border:1px solid var(--line); border-radius:4px;
                 image-rendering:pixelated; background:#000; cursor:crosshair; touch-action:none; }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:8px 0; font-size:13px; }
.controls button { font:inherit; padding:3px 10px; border:1px solid var(--line); border-radius:4px;
                   background:var(--bg); color:var(--fg); cursor:pointer; }
.controls button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
.trajgrp { display:inline-flex; gap:8px; align-items:center; padding-left:6px;
           margin-left:2px; border-left:1px solid var(--line); }
.trajgrp input.traj { vertical-align:middle; width:150px; accent-color:var(--accent); }
.trajlbl { font:12px/1.3 monospace; color:var(--muted, #667); min-width:96px; }
.tip { position:absolute; pointer-events:none; background:rgba(0,0,0,.82); color:#fff; padding:4px 7px;
       border-radius:4px; font:12px/1.3 monospace; white-space:pre; display:none; z-index:5; }
.roi { flex:1 1 320px; min-width:280px; max-height:360px; overflow:auto; }
.roi table td { cursor:pointer; } .roi tr.sel td { background:rgba(36,113,163,.18); }
.metrics { font-size:13px; margin:6px 0 12px; }
.metrics table { width:auto; } .metrics td, .metrics th { padding:4px 12px; }
footer { color:var(--muted); font-size:12px; margin-top:40px; }
code { background:var(--line); padding:1px 5px; border-radius:3px; font-size:12px; }
"""

# One viewer instance per field. Data is attached as a JSON <script> per field and
# wired by id. Vanilla JS, no libraries.
_JS = r"""
function halfToFloat(h){  // IEEE 754 half -> JS number
  const s=(h&0x8000)>>15, e=(h&0x7C00)>>10, f=h&0x03FF;
  if (e===0) return (s?-1:1)*Math.pow(2,-14)*(f/1024);
  if (e===0x1F) return f?NaN:((s?-1:1)*Infinity);
  return (s?-1:1)*Math.pow(2,e-15)*(1+f/1024);
}
function b64Bytes(b64){
  const bin = atob(b64); const len = bin.length; const u8 = new Uint8Array(len);
  for (let i=0;i<len;i++) u8[i]=bin.charCodeAt(i);
  return u8;
}
function b64ToF32(b64){  // base64 of little-endian float16 -> Float32Array
  const u8 = b64Bytes(b64); const n = u8.length>>1; const out = new Float32Array(n);
  for (let i=0;i<n;i++) out[i] = halfToFloat(u8[2*i] | (u8[2*i+1]<<8));
  return out;
}
function b64ToU16(b64){  // base64 of little-endian uint16 -> plain array (exact coords)
  const u8 = b64Bytes(b64); const n = u8.length>>1; const out = new Array(n);
  for (let i=0;i<n;i++) out[i] = u8[2*i] | (u8[2*i+1]<<8);
  return out;
}
function divColor(v, thr, vmax){
  // diverging: folded (< thr) reds, feasible blues, |v| scaled by vmax
  const t = Math.max(-1, Math.min(1, v/vmax));
  if (v < thr){ const a=Math.min(1, (thr-v)/vmax); return [190+65*a|0, 60-40*a|0, 50-40*a|0]; }
  const b=Math.min(1, v/vmax); return [70-20*b|0, 110+40*b|0, 180+40*b|0];
}
function buildImage(f32, W, H, thr, vmax){
  const img = new ImageData(W, H);
  for (let i=0;i<W*H;i++){ const c=divColor(f32[i], thr, vmax);
    img.data[4*i]=c[0]; img.data[4*i+1]=c[1]; img.data[4*i+2]=c[2]; img.data[4*i+3]=255; }
  return img;
}
function Viewer(root, data){
  const W=data.w, H=data.h, thr=data.threshold, vmax=data.vmax;
  const jb=b64ToF32(data.jdet_before), ja=b64ToF32(data.jdet_after);
  const dyB=b64ToF32(data.dy_before), dxB=b64ToF32(data.dx_before);
  const dyA=b64ToF32(data.dy_after), dxA=b64ToF32(data.dx_after);
  const hasCorr=!!data.corr_fx;
  const cfy=hasCorr?b64ToU16(data.corr_fy):null, cfx=hasCorr?b64ToU16(data.corr_fx):null;
  const cmy=hasCorr?b64ToU16(data.corr_my):null, cmx=hasCorr?b64ToU16(data.corr_mx):null;
  const coutlier=new Set(hasCorr?data.corr_outlier_idx:[]);
  let showCorr=false;
  const cv=root.querySelector('canvas'), ctx=cv.getContext('2d');
  const tip=root.querySelector('.tip');
  const off=document.createElement('canvas'); off.width=W; off.height=H; const octx=off.getContext('2d');
  let showAfter=false, showQuiver=false, trajIdx=-1, playTimer=null;
  const imgB=buildImage(jb,W,H,thr,vmax), imgA=buildImage(ja,W,H,thr,vmax);
  const hasTraj=!!(data.traj&&data.traj.length);
  const trajF32=hasTraj?data.traj.map(b64ToF32):null;
  const imgTraj=hasTraj?trajF32.map(f=>buildImage(f,W,H,thr,vmax)):null;
  const trajLbls=data.traj_labels||[];
  function curImg(){ return trajIdx>=0?imgTraj[trajIdx]:(showAfter?imgA:imgB); }
  function setTrajLbl(){ const l=root.querySelector('.trajlbl');
    if (l) l.textContent = trajIdx>=0 ? (trajLbls[trajIdx]||('frame '+(trajIdx+1)+'/'+imgTraj.length)) : ''; }
  function stopPlay(){ if (playTimer){ clearInterval(playTimer); playTimer=null; }
    const pb=root.querySelector('[data-act=play]'); if (pb) pb.classList.remove('on'); }
  function exitTraj(){ stopPlay(); trajIdx=-1;
    const s=root.querySelector('.traj'); if (s) s.value=0; setTrajLbl(); }
  cv.width=W; cv.height=H;  // internal res = data res; CSS scales to fit
  let scale=1, ox=0, oy=0, roiBox=null;
  function draw(){
    octx.putImageData(curImg(), 0, 0);
    ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,cv.width,cv.height);
    ctx.imageSmoothingEnabled=false;
    ctx.setTransform(scale,0,0,scale,ox,oy);
    ctx.drawImage(off,0,0);
    if (showQuiver && trajIdx<0){
      const dy=showAfter?dyA:dyB, dx=showAfter?dxA:dxB;  // vectors match the shown view
      const stride=Math.max(1, Math.round(10/scale)); ctx.lineWidth=Math.max(0.4,0.8/scale);
      ctx.strokeStyle='rgba(255,220,0,.8)'; ctx.beginPath();
      for (let y=0;y<H;y+=stride) for (let x=0;x<W;x+=stride){
        const i=y*W+x;  // dy/dx are in pixel (voxel) units -> draw the actual warp vector
        ctx.moveTo(x+0.5, y+0.5); ctx.lineTo(x+0.5+dx[i], y+0.5+dy[i]);
      }
      ctx.stroke();
    }
    if (showCorr && hasCorr){
      ctx.lineWidth=Math.max(0.35,0.6/scale); ctx.strokeStyle='rgba(0,200,255,.45)'; ctx.beginPath();
      for (let i=0;i<cfx.length;i++){ ctx.moveTo(cfx[i]+0.5,cfy[i]+0.5); ctx.lineTo(cmx[i]+0.5,cmy[i]+0.5); }
      ctx.stroke();
      const r=Math.max(0.6,1.3/scale);
      for (let i=0;i<cfx.length;i++){ ctx.fillStyle=coutlier.has(i)?'#ff8a00':'#00d0ff';
        ctx.fillRect(cfx[i]+0.5-r, cfy[i]+0.5-r, 2*r, 2*r); }
    }
    if (roiBox){ ctx.lineWidth=Math.max(1,2/scale); ctx.strokeStyle='#ffd000';
      ctx.strokeRect(roiBox[0]+0.5, roiBox[1]+0.5, roiBox[2], roiBox[3]); }
  }
  function toPix(ev){ const r=cv.getBoundingClientRect();
    const cx=(ev.clientX-r.left)*(cv.width/r.width), cy=(ev.clientY-r.top)*(cv.height/r.height);
    return [ (cx-ox)/scale, (cy-oy)/scale ]; }
  cv.addEventListener('wheel', e=>{ e.preventDefault();
    const [px,py]=toPix(e); const f=e.deltaY<0?1.2:1/1.2; const ns=Math.max(1, Math.min(60, scale*f));
    // keep the pixel under the cursor fixed: screen = p*scale+o must be invariant
    ox=(px*scale+ox) - px*ns; oy=(py*scale+oy) - py*ns; scale=ns; draw();
  }, {passive:false});
  let drag=null;
  cv.addEventListener('pointerdown', e=>{ drag=[e.clientX,e.clientY,ox,oy]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointerup', ()=>drag=null);
  cv.addEventListener('pointermove', e=>{
    if (drag){ const r=cv.getBoundingClientRect(); const sx=cv.width/r.width, sy=cv.height/r.height;
      ox=drag[2]+(e.clientX-drag[0])*sx; oy=drag[3]+(e.clientY-drag[1])*sy; draw(); return; }
    const [px,py]=toPix(e); const x=Math.floor(px), y=Math.floor(py);
    if (x<0||y<0||x>=W||y>=H){ tip.style.display='none'; return; }
    const i=y*W+x; const jv=trajIdx>=0?trajF32[trajIdx][i]:(showAfter?ja:jb)[i];
    let txt='y='+y+' x='+x+'\nJdet ~'+jv.toFixed(3)+(jv<thr?'  (FOLD)':'');
    if (trajIdx<0){ const hdy=(showAfter?dyA:dyB)[i], hdx=(showAfter?dxA:dxB)[i];
      txt+='\ndy='+hdy.toFixed(3)+' dx='+hdx.toFixed(3); }
    tip.textContent=txt;
    const r=cv.getBoundingClientRect();
    tip.style.left=(e.clientX-r.left+12)+'px'; tip.style.top=(e.clientY-r.top+12)+'px'; tip.style.display='block';
  });
  cv.addEventListener('pointerleave', ()=>{ tip.style.display='none'; });
  root.querySelector('[data-act=after]').addEventListener('click', e=>{
    exitTraj(); showAfter=!showAfter; e.target.classList.toggle('on', showAfter);
    e.target.textContent=showAfter?'Showing: after':'Showing: before'; draw(); });
  if (hasTraj){
    const sl=root.querySelector('.traj'), pb=root.querySelector('[data-act=play]');
    sl.addEventListener('input', ()=>{ stopPlay(); trajIdx=+sl.value; setTrajLbl(); draw(); });
    pb.addEventListener('click', ()=>{
      if (playTimer){ stopPlay(); return; }
      pb.classList.add('on'); trajIdx=0; sl.value=0; setTrajLbl(); draw();
      playTimer=setInterval(()=>{
        if (trajIdx>=imgTraj.length-1){ stopPlay(); return; }
        trajIdx++; sl.value=trajIdx; setTrajLbl(); draw();
      }, 450);
    });
  }
  root.querySelector('[data-act=quiver]').addEventListener('click', e=>{
    showQuiver=!showQuiver; e.target.classList.toggle('on', showQuiver); draw(); });
  const corrBtn=root.querySelector('[data-act=corr]');
  if (corrBtn) corrBtn.addEventListener('click', e=>{
    showCorr=!showCorr; e.target.classList.toggle('on', showCorr); draw(); });
  root.querySelector('[data-act=reset]').addEventListener('click', ()=>{ exitTraj(); scale=1; ox=0; oy=0; roiBox=null; draw(); });
  function centerOn(cx, cy, half){  // set zoom+pan to frame a region; caller sets roiBox
    const pad=half*1.5+8; scale=Math.max(1, Math.min(60, cv.width/(2*pad)));
    ox=cv.width/2 - cx*scale; oy=cv.height/2 - cy*scale; draw();
  }
  root.querySelectorAll('.roi tr[data-bbox]').forEach(tr=>{
    tr.addEventListener('click', ()=>{
      root.querySelectorAll('.roi tr').forEach(t=>t.classList.remove('sel')); tr.classList.add('sel');
      const bb=JSON.parse(tr.getAttribute('data-bbox'));  // [y0,y1,x0,x1]
      const w=bb[3]-bb[2], h=bb[1]-bb[0]; roiBox=[bb[2], bb[0], w, h];  // exact bbox
      centerOn(bb[2]+w/2, bb[0]+h/2, Math.max(w, h));
    });
  });
  root.querySelectorAll('.roi tr[data-loc]').forEach(tr=>{
    tr.addEventListener('click', ()=>{
      root.querySelectorAll('.roi tr').forEach(t=>t.classList.remove('sel')); tr.classList.add('sel');
      const yx=JSON.parse(tr.getAttribute('data-loc'));  // [y,x]
      if (corrBtn && !showCorr){ showCorr=true; corrBtn.classList.add('on'); }
      roiBox=[yx[1]-6, yx[0]-6, 12, 12]; centerOn(yx[1], yx[0], 14);
    });
  });
  draw();
}
function initViewers(){
  document.querySelectorAll('.field[data-viewer]').forEach(f=>{
    const data=JSON.parse(document.getElementById('data-'+f.getAttribute('data-viewer')).textContent);
    try { Viewer(f, data); } catch(err){ f.querySelector('.viewer').innerHTML='<p class=sub>viewer failed: '+err+'</p>'; }
  });
}
function sortTable(th){
  const tb=th.closest('table').tBodies[0]; const idx=[...th.parentNode.children].indexOf(th);
  const num=v=>{const f=parseFloat(v.replace(/[^0-9.\-]/g,'')); return isNaN(f)?v:f;};
  const rows=[...tb.rows]; const asc=th._asc=!th._asc;
  rows.sort((a,b)=>{const x=num(a.cells[idx].textContent),y=num(b.cells[idx].textContent);
    return (x>y?1:x<y?-1:0)*(asc?1:-1);}); rows.forEach(r=>tb.appendChild(r));
}
document.addEventListener('DOMContentLoaded', initViewers);
"""


def _esc(v):
    return html.escape("" if v is None else str(v))


def _roi_table(rois, threshold):
    if not rois:
        return '<p class="sub">No fold clusters.</p>'
    head = "".join(
        f"<th onclick='sortTable(this)'>{h}</th>"
        for h in ("#", "z", "y", "x", "voxels", "neg vol", "min Jdet")
    )
    body = []
    for e in rois:
        z = e.get("z", "")
        bb = e["bbox"]
        # bbox as [y0,y1,x0,x1] for the 2D viewer (last two dims)
        yx = bb[-2] + bb[-1]
        body.append(
            f'<tr data-bbox="{yx}"><td>{e["rank"]}</td><td>{_esc(z)}</td>'
            f'<td>{e["y"]}</td><td>{e["x"]}</td><td>{e["size"]:,}</td>'
            f'<td>{e["neg_vol"]:.1f}</td><td>{e["min_jdet"]:.3f}</td></tr>'
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _metrics_table(families):
    """families: list of (name, n_before, n_after, min_before, min_after)."""
    rows = []
    for name, nb, na, mb, ma in families:
        cls = "feasible" if na == 0 else "infeasible"
        rows.append(
            f'<tr class="{cls}"><td>{_esc(name)}</td><td>{nb:,} &rarr; '
            f'<span class="status">{na:,}</span></td>'
            f"<td>{mb:.3f} &rarr; {ma:.3f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>constraint</th><th>folds (before &rarr; after)</th>"
        f"<th>min value</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _corr_stats_html(s):
    return (
        f'<p class="sub">Correspondences on this slice: <b>{s["n"]:,}</b> · '
        f'mean prescribed |move&rarr;fixed| <b>{s["mean_disp"]:.2f}</b> vox (max {s["max_disp"]:.1f}) · '
        f'registration residual (fit) before <b>{s["mean_resid_before"]:.3f}</b> &rarr; '
        f'after <b>{s["mean_resid_after"]:.3f}</b> vox · outliers <b>{s["n_outliers"]}</b> '
        f'(large-disp {s["n_large"]}, high-residual {s["n_high_resid"]}, '
        f'incoherent {s["n_incoherent"]})</p>'
    )


def _corr_outlier_table(outliers):
    if not outliers:
        return ""
    # Only single-value numeric columns get sorting; multi-value cells (move y,x,
    # resid b->a) and text would sort on a mangled concatenated number.
    sortable = {"#", "y", "x", "disp"}
    head = "".join(
        (f"<th onclick='sortTable(this)'>{h}</th>" if h in sortable else f"<th>{h}</th>")
        for h in ("#", "y", "x", "move y,x", "disp", "resid b&rarr;a", "type")
    )
    body = []
    for e in outliers:
        body.append(
            f'<tr data-loc="[{e["y"]},{e["x"]}]"><td>{e["rank"]}</td>'
            f'<td>{e["y"]}</td><td>{e["x"]}</td><td>{e["my"]},{e["mx"]}</td>'
            f'<td>{e["disp"]:.1f}</td><td>{e["resid_before"]:.2f}&rarr;{e["resid_after"]:.2f}</td>'
            f'<td>{_esc(e["types"])}</td></tr>'
        )
    return (
        '<div class="roi"><div class="sub">Correspondence outliers (click to locate)</div>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _field_block(p):
    vid = p["id"]
    has_corr = "corr_fy" in p
    has_traj = bool(p.get("traj"))
    corr_btn = "<button data-act=corr>Correspondences</button>" if has_corr else ""
    traj_ctrl = (
        (
            '<span class="trajgrp"><button data-act=play>▶ Play iterations</button>'
            f'<input type=range class=traj min=0 max="{len(p["traj"]) - 1}" value=0 step=1>'
            '<span class=trajlbl></span></span>'
        )
        if has_traj
        else ""
    )
    controls = (
        '<div class="controls">'
        '<button data-act=after>Showing: before</button>'
        '<button data-act=quiver>Displacement vectors</button>'
        f"{corr_btn}"
        '<button data-act=reset>Reset view</button>'
        f"{traj_ctrl}"
        "<span class=sub>scroll = zoom · drag = pan · hover = value</span></div>"
    )
    viewer = (
        f'<div class="viewer"><canvas></canvas><div class="tip"></div></div>'
        f'<div class="roi">{_roi_table(p["rois"], p["threshold"])}</div>'
        f"{_corr_outlier_table(p.get('corr_outliers')) if has_corr else ''}"
    )
    data = {
        "w": p["w"],
        "h": p["h"],
        "threshold": p["threshold"],
        "vmax": p["vmax"],
        "jdet_before": p["jdet_before"],
        "jdet_after": p["jdet_after"],
        "dy_before": p["dy_before"],
        "dx_before": p["dx_before"],
        "dy_after": p["dy_after"],
        "dx_after": p["dx_after"],
    }
    if has_corr:
        for k in ("corr_fy", "corr_fx", "corr_my", "corr_mx", "corr_outlier_idx"):
            data[k] = p[k]
    if has_traj:
        data["traj"] = p["traj"]
        data["traj_labels"] = p.get("traj_labels") or []
    data_json = json.dumps(data)
    note = f'<p class="sub">{_esc(p["note"])}</p>' if p.get("note") else ""
    corr_stats = _corr_stats_html(p["corr_stats"]) if has_corr else ""
    return (
        f'<div class="field" data-viewer="{vid}"><h3>{_esc(p["label"])}</h3>'
        f'<div class="metrics">{_metrics_table(p["families"])}</div>{note}{corr_stats}'
        f"{controls}"
        f'<div class="viewer-row">{viewer}</div>'
        f'<script type="application/json" id="data-{vid}">{data_json}</script></div>'
    )


def build_interactive_report(out_path, meta, payloads):
    """Write a self-contained interactive ``report.html``. Never raises; returns path.

    *payloads* is a list of per-field dicts (see ``_field_block``); *meta* carries
    run-level header fields.
    """
    from pathlib import Path

    out_path = Path(out_path)
    try:
        n = len(payloads)
        n_feasible = sum(1 for p in payloads if p["families"] and p["families"][0][2] == 0)
        if n == 0:
            bcls, btxt = "warn", "No fields processed (cohort data not found?)."
        elif n_feasible == n:
            bcls, btxt = "ok", f"All {n} fields feasible (0 residual Jdet folds)."
        else:
            bcls, btxt = "warn", f"{n - n_feasible} of {n} fields still have residual Jdet folds."
        hdr = "".join(
            f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>"
            for k, v in [
                ("Corrector", meta.get("corrector")),
                ("Threshold", meta.get("threshold")),
                ("Fields", n),
                ("Feasible", f"{n_feasible} / {n}"),
                ("Generated", meta.get("generated")),
                ("Total wall time", f"{meta.get('total_time_s', 0):.1f}s"),
            ]
            if v not in (None, "")
        )
        blocks = "".join(_field_block(p) for p in payloads)
        body = (
            "<h1>Interactive Cohort Report</h1>"
            '<p class="sub">Pan/zoom Jacobian maps, hover for values (~3-digit display; '
            "tables are exact), toggle "
            "displacement vectors, click a region to focus.</p>"
            f'<dl class="hdr-grid">{hdr}</dl>'
            f'<div class="banner {bcls}">{_esc(btxt)}</div>{blocks}'
        )
        doc = (
            '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"/>'
            '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
            "<title>Interactive Cohort Report</title>"
            f"<style>{_CSS}</style></head><body>"
            f'<div class="wrap">{body}'
            "<footer>Generated by dvfopt cohort_benchmark — self-contained interactive "
            "report (no external assets).</footer></div>"
            f"<script>{_JS}</script></body></html>"
        )
    except Exception as exc:  # never raise
        doc = (
            '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"/>'
            "<title>Interactive Cohort Report</title></head><body>"
            f"<h1>Interactive Cohort Report</h1><p>Report could not be generated: "
            f"{_esc(exc)}</p></body></html>"
        )
    out_path.write_text(doc, encoding="utf-8")
    return out_path
