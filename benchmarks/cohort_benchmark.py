"""Cohort correction benchmark: run a corrector across brain-cohort fields and
emit a self-contained run directory (CSVs, figures, HTML report).

Each ``run_cohort_benchmark`` call creates one timestamped run directory::

    output/cohort/<timestamp>_<run_name>/
        results.csv        one row per field (before/after fold metrics + time)
        summary.json        run-level aggregate + provenance
        figures/*.png        per-field before/after Jdet figure
        report.html         self-contained report (inline CSS, base64 figures)

The HTML report is inspired by RegTools' report module: one portable file, no
external assets/CDNs, collapsible sections, robust (missing pieces are omitted).

Fields are the dvfopt-native cohort loaded via :mod:`benchmark_utils`
(``list_cohort``/``load_cohort_field``); ``dz`` is 0 so the default corrector is
the 2.5D marching pipeline. Pass any ``corrector(phi) -> phi`` to benchmark a
different strategy, or pass ``fields={label: phi}`` to bypass cohort loading
(used by the self-check and by tests, which run without the gitignored data).
"""

import base64
import csv
import datetime
import html
import io
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no GUI, safe in notebooks / CI
import benchmark_utils as bu
import matplotlib.pyplot as plt
import numpy as np

from dvfopt import jacobian_det2D, jacobian_det3D

# ---------------------------------------------------------------------------
# Correctors
# ---------------------------------------------------------------------------


def make_25d_corrector(**kw):
    """Return a ``corrector(phi) -> phi`` wrapping ``correct_dvf_25d`` (needs dz==0)."""
    from dvfopt import correct_dvf_25d

    def _corrector(phi):
        out, _report = correct_dvf_25d(phi, **kw)
        return out

    _corrector.label = "correct_dvf_25d(" + ", ".join(f"{k}={v}" for k, v in kw.items()) + ")"
    return _corrector


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _neg_volume(jac, threshold):
    """Fold severity: summed depth below threshold over folded voxels (>= 0)."""
    return float(np.clip(threshold - jac, 0.0, None).sum())


def _n_clusters(jac, threshold):
    """Number of connected fold regions (face-connectivity) in the negative mask."""
    from scipy.ndimage import label

    return int(label(jac < threshold)[1])


def _measure(phi_init, phi, elapsed, threshold):
    """Before/after 3D-Jacobian fold metrics for one field."""
    jac_init = jacobian_det3D(phi_init)
    jac_final = jacobian_det3D(phi)
    return {
        "n_neg_init": int((jac_init < threshold).sum()),
        "n_neg_final": int((jac_final < threshold).sum()),
        "neg_vol_init": _neg_volume(jac_init, threshold),
        "neg_vol_final": _neg_volume(jac_final, threshold),
        "n_clusters_init": _n_clusters(jac_init, threshold),
        "n_clusters_final": _n_clusters(jac_final, threshold),
        "min_jdet_init": float(jac_init.min()),
        "min_jdet_final": float(jac_final.min()),
        "l2_err": float(
            np.sqrt(np.sum((phi.astype(np.float64) - phi_init.astype(np.float64)) ** 2))
        ),
        "time_s": float(elapsed),
        "_jac_init": jac_init,
        "_jac_final": jac_final,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


_C_BEFORE = "#c0392b"
_C_AFTER = "#2471a3"


def _field_figure(label, jac_init, jac_final, threshold):
    """Per-field folding figure (PNG bytes): worst-slice Jdet before/after,
    Jdet distribution, and per-slice fold profile along z."""
    per_slice_min = jac_init.reshape(jac_init.shape[0], -1).min(axis=1)
    z = int(np.argmin(per_slice_min))
    a, b = jac_init[z], jac_final[z]
    vmax = float(max(1.0, np.percentile(np.abs(np.concatenate([a.ravel(), b.ravel()])), 99)))

    fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
    for k, (arr, ttl) in enumerate(((a, "Before"), (b, "After"))):
        im = ax[0, k].imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax[0, k].set_title(f"{ttl}  Jdet @ worst z={z}")
        ax[0, k].set_xticks([])
        ax[0, k].set_yticks([])
        fig.colorbar(im, ax=ax[0, k], fraction=0.046, pad=0.04)

    lo = min(jac_init.min(), jac_final.min())
    bins = np.linspace(min(lo, -1.0), 2.0, 60)
    ax[1, 0].hist(jac_init.ravel(), bins=bins, alpha=0.5, label="before", color=_C_BEFORE)
    ax[1, 0].hist(jac_final.ravel(), bins=bins, alpha=0.5, label="after", color=_C_AFTER)
    ax[1, 0].axvline(threshold, color="k", lw=0.8, ls="--")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("Jdet distribution")
    ax[1, 0].legend(fontsize=8)

    zf_i = (jac_init < threshold).reshape(jac_init.shape[0], -1).sum(axis=1)
    zf_f = (jac_final < threshold).reshape(jac_final.shape[0], -1).sum(axis=1)
    zs = np.arange(jac_init.shape[0])
    ax[1, 1].plot(zs, zf_i, color=_C_BEFORE, lw=1.0, label="before")
    ax[1, 1].plot(zs, zf_f, color=_C_AFTER, lw=1.0, label="after")
    ax[1, 1].set_title("Folds per z-slice")
    ax[1, 1].set_xlabel("z")
    ax[1, 1].set_ylabel("fold voxels")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(label, fontsize=11)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _cohort_figure(rows):
    """Aggregate: folds before/after per brain (grouped bars). PNG bytes, or None."""
    labeled = [r for r in rows if r.get("brain")]
    if not labeled:
        return None
    names = [r["label"] for r in labeled]
    before = [r["n_neg_init"] for r in labeled]
    after = [r["n_neg_final"] for r in labeled]
    x = np.arange(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4))
    ax.bar(x - w / 2, before, w, label="before", color=_C_BEFORE)
    ax.bar(x + w / 2, after, w, label="after", color=_C_AFTER)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("fold voxels")
    ax.set_title("Folds before / after by field")
    ax.legend(fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _png_data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# HTML report (self-contained, inspired by RegTools' report module)
# ---------------------------------------------------------------------------

_CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --card:#fafafa;
        --good:#1e7e34; --bad:#c0392b; --accent:#2471a3; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161616; --fg:#e8e8e8; --muted:#9a9a9a; --line:#333; --card:#1f1f1f;
          --good:#4cd07d; --bad:#ff6b5e; --accent:#5aa9e6; } }
* { box-sizing: border-box; }
body { background:var(--bg); color:var(--fg); margin:0;
       font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1000px; margin:0 auto; padding:32px 24px 64px; }
h1 { font-size:24px; margin:0 0 4px; }
h2 { font-size:18px; margin:0; }
.sub { color:var(--muted); margin:0 0 20px; }
dl.hdr-grid { display:grid; grid-template-columns:auto 1fr; gap:2px 16px;
              margin:0 0 24px; font-size:14px; }
dl.hdr-grid dt { color:var(--muted); }
dl.hdr-grid dd { margin:0; font-variant-numeric:tabular-nums; }
.banner { padding:10px 14px; border-radius:6px; margin:0 0 24px; font-weight:600; }
.banner.ok { background:rgba(30,126,52,.12); color:var(--good); }
.banner.warn { background:rgba(192,57,43,.12); color:var(--bad); }
nav.toc { border:1px solid var(--line); border-radius:6px; padding:12px 16px; margin:0 0 24px; }
nav.toc .toc-h { font-weight:600; margin-bottom:6px; }
nav.toc ul { margin:0; padding-left:18px; columns:2; }
nav.toc a { color:var(--accent); text-decoration:none; }
section { margin:0 0 20px; }
details { border:1px solid var(--line); border-radius:6px; padding:0 16px; background:var(--card); }
summary { cursor:pointer; padding:12px 0; list-style:none; }
summary::-webkit-details-marker { display:none; }
.sect-body { padding:0 0 16px; }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
thead th { color:var(--muted); font-weight:600; }
tr.feasible td.status { color:var(--good); }
tr.infeasible td.status { color:var(--bad); }
.card { border:1px solid var(--line); border-radius:6px; padding:12px; margin:12px 0; }
.card h3 { margin:0 0 8px; font-size:15px; }
.card img { width:100%; height:auto; border-radius:4px; }
footer { color:var(--muted); font-size:12px; margin-top:40px; }
code { background:var(--line); padding:1px 5px; border-radius:3px; font-size:12px; }
"""


def _esc(v):
    return html.escape("" if v is None else str(v))


def _fmt(x, nd=4):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return _esc(x)
    if not math.isfinite(f):  # nan / inf: render as text, never int(nan)->ValueError
        return _esc(f)
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    return f"{f:,.{nd}f}"


def _arrow(before, after, nd=0):
    return f"{_fmt(before, nd)} &rarr; {_fmt(after, nd)}"


def _summary_table(rows):
    # Compact before→after cells keep every folding metric in one readable table.
    head = (
        "Field",
        "folds",
        "% removed",
        "neg volume",
        "clusters",
        "min Jdet",
        "L2 move",
        "time (s)",
    )
    body = []
    for r in rows:
        ni, nf = r["n_neg_init"], r["n_neg_final"]
        pct = 100.0 * (ni - nf) / ni if ni else 0.0
        feas = "feasible" if nf == 0 else "infeasible"
        body.append(
            f'<tr class="{feas}"><td>{_esc(r["label"])}</td>'
            f'<td class="status">{_arrow(ni, nf)}</td>'
            f"<td>{pct:.2f}%</td>"
            f"<td>{_arrow(r['neg_vol_init'], r['neg_vol_final'], 1)}</td>"
            f"<td>{_arrow(r['n_clusters_init'], r['n_clusters_final'])}</td>"
            f"<td>{_arrow(r['min_jdet_init'], r['min_jdet_final'], 3)}</td>"
            f"<td>{_fmt(r['l2_err'], 2)}</td>"
            f"<td>{_fmt(r['time_s'], 1)}</td></tr>"
        )
    ths = "".join(f"<th>{_esc(h)}</th>" for h in head)
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _cards(rows):
    # Figures only — the numeric detail lives once in the summary table above.
    out = []
    for r in rows:
        if not r.get("fig_uri"):
            continue
        out.append(
            f'<div class="card"><h3>{_esc(r["label"])}</h3>'
            f'<img src="{r["fig_uri"]}" alt="{_esc(r["label"])}"/></div>'
        )
    return "".join(out)


def _section(sid, title, body):
    if not body or not body.strip():
        return ""
    return (
        f'<section id="{_esc(sid)}"><details open>'
        f"<summary><h2>{_esc(title)}</h2></summary>"
        f'<div class="sect-body">{body}</div></details></section>'
    )


def build_cohort_report(run_dir, meta, rows, cohort_fig_uri=None):
    """Write a self-contained ``report.html`` in *run_dir*. Never raises; returns its path."""
    out_path = Path(run_dir) / "report.html"
    try:
        n = len(rows)
        n_feasible = sum(1 for r in rows if r["n_neg_final"] == 0)
        tot_before = sum(r["n_neg_init"] for r in rows)
        tot_after = sum(r["n_neg_final"] for r in rows)
        hdr_pairs = [
            ("Corrector", meta.get("corrector")),
            ("Threshold", meta.get("threshold")),
            ("Fields", n),
            ("Feasible (0 folds)", f"{n_feasible} / {n}"),
            ("Total folds", f"{tot_before:,} → {tot_after:,}"),
            ("Generated", meta.get("generated")),
            ("Total wall time", f"{meta.get('total_time_s', 0):.1f}s"),
        ]
        hdr = "".join(
            f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in hdr_pairs if v not in (None, "")
        )
        if n == 0:
            banner_cls, banner_txt = ("warn", "No fields processed (cohort data not found?).")
        elif n_feasible == n:
            banner_cls, banner_txt = ("ok", f"All {n} fields feasible (0 residual folds).")
        else:
            banner_cls = "warn"
            banner_txt = f"{n - n_feasible} of {n} fields still have residual folds."
        overview = (
            f'<img src="{cohort_fig_uri}" alt="folds before/after by field"/>'
            if cohort_fig_uri
            else ""
        )
        overview_link = '<li><a href="#overview">Cohort overview</a></li>' if overview else ""
        toc = (
            '<nav class="toc"><div class="toc-h">Contents</div><ul>'
            f"{overview_link}"
            '<li><a href="#summary">Summary</a></li>'
            '<li><a href="#fields">Per-field results</a></li>'
            '<li><a href="#repro">Reproduce</a></li></ul></nav>'
        )
        repro = (
            f"<p>Corrector: <code>{_esc(meta.get('corrector'))}</code></p>"
            f"<p>Fields from <code>benchmark_utils.load_cohort_field</code>; "
            f"metrics via <code>jacobian_det3D</code> at threshold "
            f"<code>{_esc(meta.get('threshold'))}</code>. Negative volume = "
            f"&sum;(threshold&minus;Jdet) over folded voxels; clusters = connected "
            f"fold regions.</p>"
            f"<p>Raw rows: <code>results.csv</code> · aggregate: <code>summary.json</code>.</p>"
        )
        body = (
            f"<h1>Cohort Correction Report</h1>"
            f'<p class="sub">Self-contained dvfopt cohort benchmark report.</p>'
            f'<dl class="hdr-grid">{hdr}</dl>'
            f'<div class="banner {banner_cls}">{_esc(banner_txt)}</div>'
            f"{toc}"
            f"{_section('overview', 'Cohort overview', overview)}"
            f"{_section('summary', 'Summary', _summary_table(rows))}"
            f"{_section('fields', 'Per-field results', _cards(rows))}"
            f"{_section('repro', 'Reproduce', repro)}"
        )
        doc = (
            '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"/>'
            '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
            "<title>Cohort Correction Report</title>"
            f"<style>{_CSS}</style></head><body>"
            f'<div class="wrap">{body}'
            "<footer>Generated by dvfopt cohort_benchmark — self-contained "
            "(no external assets).</footer></div></body></html>"
        )
    except Exception as exc:  # never raise — mirror RegTools' robust contract
        doc = (
            '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"/>'
            "<title>Cohort Correction Report</title></head><body>"
            f"<h1>Cohort Correction Report</h1><p>Report could not be fully "
            f"generated: {_esc(exc)}</p></body></html>"
        )
    out_path.write_text(doc, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_CSV_COLS = [
    "label",
    "brain",
    "variant",
    "n_neg_init",
    "n_neg_final",
    "neg_vol_init",
    "neg_vol_final",
    "n_clusters_init",
    "n_clusters_final",
    "min_jdet_init",
    "min_jdet_final",
    "l2_err",
    "time_s",
]


def run_cohort_benchmark(
    corrector=None,
    *,
    fields=None,
    items=None,
    variant="laplacian_exterior",
    run_name="run",
    out_base="output/cohort",
    threshold=0.01,
    make_figures=True,
    verbose=0,
):
    """Run *corrector* over cohort fields; write a run directory. Returns its Path.

    Parameters
    ----------
    corrector : callable ``phi -> phi`` or None
        Field corrector. Defaults to ``make_25d_corrector()`` (2.5D marching).
    fields : dict ``{label: phi}``, optional
        Explicit fields, bypassing cohort loading (for tests / ad-hoc volumes).
    items : list of ``(brain, variant)``, optional
        Cohort items to load. Default: every cohort brain at *variant*.
    variant : str
        Cohort variant to benchmark when *items* is not given
        (default ``"laplacian_exterior"`` — the folding-benchmark focus).
    run_name, out_base, threshold, make_figures, verbose
        Run knobs. ``out_base`` is cwd-relative unless absolute.
    """
    if corrector is None:
        corrector = make_25d_corrector(threshold=threshold)
    corrector_label = getattr(corrector, "label", getattr(corrector, "__name__", "corrector"))

    # Resolve the (label, brain, variant, loader) work list.
    if fields is not None:
        work = [(lbl, None, None, (lambda p=phi: p)) for lbl, phi in fields.items()]
    else:
        if items is None:
            items = [(b, v) for (b, v) in bu.list_cohort() if v == variant]
        work = [
            (f"{b}/{v}", b, v, (lambda b=b, v=v: bu.load_cohort_field(b, v))) for (b, v) in items
        ]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_base) / f"{stamp}_{run_name}"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t_run = time.perf_counter()
    for label, brain, variant, load in work:
        if verbose:
            print(f"[cohort] {label} ...", flush=True)
        phi_init = np.asarray(load()).astype(np.float64)
        t0 = time.perf_counter()
        phi = corrector(phi_init.copy())
        elapsed = time.perf_counter() - t0
        m = _measure(phi_init, phi, elapsed, threshold)

        fig_uri = None
        if make_figures:
            png = _field_figure(label, m.pop("_jac_init"), m.pop("_jac_final"), threshold)
            fig_path = fig_dir / (label.replace("/", "__") + ".png")
            fig_path.write_bytes(png)
            fig_uri = _png_data_uri(png)
        else:
            m.pop("_jac_init", None)
            m.pop("_jac_final", None)

        rows.append({"label": label, "brain": brain, "variant": variant, "fig_uri": fig_uri, **m})

    total_time_s = time.perf_counter() - t_run
    _write_run_artifacts(
        run_dir, fig_dir, rows, corrector_label, threshold, total_time_s, make_figures
    )
    if verbose:
        print(f"[cohort] wrote {run_dir}", flush=True)
    return run_dir


def _write_run_artifacts(
    run_dir, fig_dir, rows, corrector_label, threshold, total_time_s, make_figures
):
    """Shared: write results.csv, summary.json, cohort figure, and report.html."""
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _CSV_COLS})

    meta = {
        "corrector": corrector_label,
        "threshold": threshold,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_fields": len(rows),
        "n_feasible": sum(1 for r in rows if r["n_neg_final"] == 0),
        "total_folds_before": sum(r["n_neg_init"] for r in rows),
        "total_folds_after": sum(r["n_neg_final"] for r in rows),
        "total_time_s": total_time_s,
    }
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cohort_fig_uri = None
    if make_figures:
        png = _cohort_figure(rows)
        if png is not None:
            (fig_dir / "_cohort.png").write_bytes(png)
            cohort_fig_uri = _png_data_uri(png)
    build_cohort_report(run_dir, meta, rows, cohort_fig_uri=cohort_fig_uri)


# ---------------------------------------------------------------------------
# 2D-section runner (isolated z-slices of the cohort — the folding focus)
# ---------------------------------------------------------------------------


def _measure_2d(sec_init, sec_out, elapsed, threshold):
    """Before/after 2D-Jacobian fold metrics for one (3, 1, H, W) section."""
    jac_init = np.asarray(jacobian_det2D(np.stack([sec_init[1, 0], sec_init[2, 0]]))).squeeze()
    jac_final = np.asarray(jacobian_det2D(np.stack([sec_out[1, 0], sec_out[2, 0]]))).squeeze()
    return {
        "n_neg_init": int((jac_init < threshold).sum()),
        "n_neg_final": int((jac_final < threshold).sum()),
        "neg_vol_init": _neg_volume(jac_init, threshold),
        "neg_vol_final": _neg_volume(jac_final, threshold),
        "n_clusters_init": _n_clusters(jac_init, threshold),
        "n_clusters_final": _n_clusters(jac_final, threshold),
        "min_jdet_init": float(jac_init.min()),
        "min_jdet_final": float(jac_final.min()),
        "l2_err": float(
            np.sqrt(np.sum((sec_out.astype(np.float64) - sec_init.astype(np.float64)) ** 2))
        ),
        "time_s": float(elapsed),
        "_jac_init": jac_init,
        "_jac_final": jac_final,
    }


def _section_figure(label, jac_init, jac_final, threshold):
    """2D-section folding figure (PNG bytes): before/after Jdet + Jdet distribution."""
    vmax = float(
        max(1.0, np.percentile(np.abs(np.concatenate([jac_init.ravel(), jac_final.ravel()])), 99))
    )
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, (arr, ttl) in enumerate(((jac_init, "Before"), (jac_final, "After"))):
        im = ax[k].imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax[k].set_title(f"{ttl}  Jdet")
        ax[k].set_xticks([])
        ax[k].set_yticks([])
        fig.colorbar(im, ax=ax[k], fraction=0.046, pad=0.04)
    lo = min(jac_init.min(), jac_final.min())
    bins = np.linspace(min(lo, -1.0), 2.0, 60)
    ax[2].hist(jac_init.ravel(), bins=bins, alpha=0.5, label="before", color=_C_BEFORE)
    ax[2].hist(jac_final.ravel(), bins=bins, alpha=0.5, label="after", color=_C_AFTER)
    ax[2].axvline(threshold, color="k", lw=0.8, ls="--")
    ax[2].set_yscale("log")
    ax[2].set_title("Jdet distribution")
    ax[2].legend(fontsize=8)
    fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _folds_per_slice(field, threshold):
    """2D fold count per z-slice, computed slice-by-slice (memory-light).

    Uses the 2D Jacobian on each slice's [dy, dx] — the metric the 2D-section
    corrector actually targets — instead of the full 3D determinant, which would
    allocate ~GBs of float64 temporaries over the whole (3, D, H, W) volume.
    """
    field = np.asarray(field)
    d = field.shape[1]
    counts = np.empty(d, dtype=np.int64)
    for z in range(d):
        jac = np.asarray(jacobian_det2D(np.stack([field[1, z], field[2, z]]))).squeeze()
        counts[z] = int((jac < threshold).sum())
    return counts


def _worst_slices(field, threshold, n):
    """Return the *n* z-indices with the most 2D folds (descending), folded only."""
    counts = _folds_per_slice(field, threshold)
    order = np.argsort(counts)[::-1]
    return [int(z) for z in order[:n] if counts[z] > 0]


class _CorrectDvfCorrector:
    """Picklable 2D-section corrector built on ``correct_dvf`` (needed for n_workers>1)."""

    def __init__(self, label, **kw):
        self.label = label
        self._kw = kw

    def __call__(self, section):
        from dvfopt import correct_dvf

        h, w = section.shape[-2:]
        return correct_dvf(section, shape=(h, w), **self._kw).corrected


def make_jdet2d_corrector(**kw):
    """Return a picklable 2D corrector via ``correct_dvf(constraint='jdet')``."""
    return _CorrectDvfCorrector(
        "correct_dvf(constraint='jdet', strategy='auto')", constraint="jdet", **kw
    )


def _process_section(corrector, brain, z, sec_init, threshold, make_figures):
    """Solve + measure one 2D section (module-level so it is process-pool picklable).

    Returns ``(brain, z, metrics_dict, png_bytes_or_None)``. The figure PNG is
    built in the worker so only ~50 KB of bytes cross the process boundary.
    """
    sec_init = np.asarray(sec_init).astype(np.float64)
    t0 = time.perf_counter()
    sec_out = corrector(sec_init.copy())
    elapsed = time.perf_counter() - t0
    m = _measure_2d(sec_init, sec_out, elapsed, threshold)
    png = None
    if make_figures:
        png = _section_figure(f"{brain}/z{z}", m.pop("_jac_init"), m.pop("_jac_final"), threshold)
    else:
        m.pop("_jac_init", None)
        m.pop("_jac_final", None)
    return brain, z, m, png


def run_cohort_2d_sections(
    corrector=None,
    *,
    brains=None,
    sections=None,
    variant="laplacian_exterior",
    n_worst=3,
    n_workers=1,
    run_name="sections2d",
    out_base="output/cohort_2d",
    threshold=0.01,
    make_figures=True,
    verbose=0,
):
    """Correct isolated 2D z-sections of the exterior cohort; write a run directory.

    Parameters
    ----------
    corrector : callable ``(3,1,H,W) -> (3,1,H,W)`` or None
        2D section corrector. Default: ``make_jdet2d_corrector()``.
    brains : list[str], optional
        Cohort brains to pull sections from. Default: all brains present at *variant*.
    sections : list[(brain, z)], optional
        Explicit sections; overrides ``brains``/``n_worst`` auto-selection.
    n_worst : int
        When auto-selecting, the number of worst-folding z-slices per brain.
    n_workers : int
        Sections run in parallel processes when > 1. Each section is an
        independent multi-minute solve, so this scales near-linearly with cores.
        Requires a *picklable* corrector (the defaults are); a lambda/closure
        raises. Serial (n_workers=1) accepts any callable. On spawn platforms
        (Windows/macOS) a script calling this with n_workers>1 MUST guard the
        call under ``if __name__ == "__main__":`` — workers re-import the caller
        module, and an unguarded top-level call re-spawns recursively.
    variant, run_name, out_base, threshold, make_figures, verbose
        Run knobs.
    """
    if corrector is None:
        corrector = make_jdet2d_corrector(threshold=threshold)
    corrector_label = getattr(corrector, "label", getattr(corrector, "__name__", "corrector"))

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_base) / f"{stamp}_{run_name}"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the section work list, loading each brain's field at most once
    # (both the auto and explicit paths — cohort fields are multi-GB each).
    if sections is None:
        if brains is None:
            brains = [b for (b, v) in bu.list_cohort() if v == variant]
        by_brain = {b: None for b in brains}  # None => auto-pick worst slices
    else:
        by_brain = {}
        for b, z in sections:
            by_brain.setdefault(b, []).append(z)

    work = []
    for b, zs in by_brain.items():
        field = bu.load_cohort_field(b, variant)
        if zs is None:
            zs = _worst_slices(field, threshold, n_worst)
        for z in zs:
            work.append((b, z, field[:, z : z + 1].copy()))
        del field

    t_run = time.perf_counter()
    if n_workers > 1 and len(work) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if verbose:
            print(f"[2d] {len(work)} sections across {n_workers} workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_id = {
                ex.submit(_process_section, corrector, b, z, sec, threshold, make_figures): (b, z)
                for (b, z, sec) in work
            }
            done = {}
            for fut in as_completed(fut_id):
                b, z = fut_id[fut]
                done[(b, z)] = fut.result()
                if verbose:
                    print(f"[2d] {b}/z{z} done ({len(done)}/{len(work)})", flush=True)
            results = [done[(b, z)] for (b, z, _sec) in work]  # restore submission order
    else:
        results = []
        for b, z, sec in work:
            if verbose:
                print(f"[2d] {b}/z{z} ...", flush=True)
            results.append(_process_section(corrector, b, z, sec, threshold, make_figures))

    rows = []
    for brain, z, m, png in results:
        label = f"{brain}/z{z}"
        fig_uri = None
        if png is not None:
            (fig_dir / (label.replace("/", "__") + ".png")).write_bytes(png)
            fig_uri = _png_data_uri(png)
        rows.append({"label": label, "brain": brain, "variant": f"z{z}", "fig_uri": fig_uri, **m})

    total_time_s = time.perf_counter() - t_run
    _write_run_artifacts(
        run_dir, fig_dir, rows, corrector_label, threshold, total_time_s, make_figures
    )
    if verbose:
        print(f"[2d] wrote {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    # Self-check: tiny synthetic fields + a trivial corrector, no cohort data needed.
    def _make_folded(seed):
        rng = np.random.default_rng(seed)
        phi = np.zeros((3, 3, 12, 12))
        phi[1:] = rng.normal(0, 3.0, size=(2, 3, 12, 12))  # big in-plane -> folds; dz stays 0
        return phi

    demo = {"synthA": _make_folded(0), "synthB": _make_folded(1)}
    rd = run_cohort_benchmark(
        corrector=lambda p: p,  # no-op corrector: folds in == folds out
        fields=demo,
        run_name="selfcheck",
        out_base="output/cohort_selfcheck",
        verbose=1,
    )
    assert (rd / "results.csv").is_file()
    assert (rd / "summary.json").is_file()
    assert (rd / "report.html").is_file()
    assert (rd / "figures").is_dir() and any((rd / "figures").iterdir())
    doc = (rd / "report.html").read_text(encoding="utf-8")
    assert "Cohort Correction Report" in doc and "synthA" in doc
    summ = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    assert summ["n_fields"] == 2
    print("self-check OK:", rd)
