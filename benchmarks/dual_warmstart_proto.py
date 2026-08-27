"""Does DUAL warm-starting help the windowed isqp engine? (research prototype)

The engine already warm-starts the QP PRIMAL twice over: OSQP's own ``x`` across
SQP iterations inside a window, and (since ``coarse_to_fine``) a prolongated
coarse-grid correction into the fine field. The duals are a different story, and
this harness measures the two places they could be reused:

(a) **Within a window, across SQP iterations.** ``isqp_solve`` sets up ONE QP
    object per window and then ``update``s values in place, so OSQP
    (``warm_starting=True``) carries BOTH ``x`` and ``y`` from one SQP iteration
    to the next for free. ``--mode zeroy`` explicitly resets ``y`` to zero before
    every ADMM solve to price what that already buys.

(b) **Coarse level -> fine level.** ``--mode c2f`` captures each COARSE window's
    final row duals, maps them onto the fine rows they cover, and injects them as
    the initial ``y`` of each fine window's first QP. ``--mode c2ffirst`` transfers
    the coarse window's FIRST-QP duals instead (the cold, heavily-violated
    linearization) -- a converged window's last QP is the trivial ``d = 0`` one,
    whose duals are ~0, so the obvious "final duals" transfer is vacuous.

(c) **Interior point -> ADMM, inside a window.** ``--mode ipdual``. The hybrid
    backend's Clarabel leg hands OSQP only ``warm_start(x=...)``; this mode hands
    over the mapped Clarabel dual too. Included because the micro-benchmark says
    ``x`` is the half that does not matter. ``--mode ipdualcold`` narrows it to the
    window's cold first IP solve only.

Everything is a monkeypatch over the shipped engine (``_make_qp``,
``solve_window_inner``, ``_coarse_warm_start``, ``_HybridQP._solve_ip``) -- no
library edit -- so the baseline path is the real default path, byte for byte.

Row structure (verified, see ``--selfcheck``): ``SimplexConstraint2DBilinear``
lays its ``4*(H-1)*(W-1)`` rows out BLOCK-major, not cell-major::

    row = b * (H-1)*(W-1) + i * (W-1) + j        b in {T1, T2, U1, U2}

and the windowed engine's enforced-row set is built the same way
(``_influenced_2tri``: ``concat([b*m + cell_flat for b in range(k)])``). The QP
OSQP actually sees stacks three row groups::

    [0 : m]           J d + s >= -c      <- the constraint duals we care about
    [m : 2m]          0 <= s <= s_up
    [2m : 2m + nf]    -delta <= d <= delta

so an injected ``y`` has length ``2m + nf`` with only the first ``m`` entries set.

Dual scaling under restriction. Triangle areas are DIMENSIONLESS (``_restrict``
divides displacements by ``factor`` so the coarse field lives in coarse pixel
units and the same ``threshold`` means the same thing), hence ``c`` and ``J`` are
O(1) on both grids while the step rescales as ``d_coarse = d_fine / factor``. QP
stationarity ``H d = J^T y_c + y_tr`` then gives ``y_fine ~ factor * y_coarse``.
BUT the elastic formulation makes most duals bang-bang: a row whose slack is
strictly interior has ``y = -rho`` exactly (complementarity on ``s``), and an
inactive row has ``y = 0`` -- both scale-free. Only the strictly-active,
zero-slack rows carry the ``factor``. Every run prints ``dual_diagnostics`` so
the derivation is checked, not assumed; ``--scale`` overrides the multiplier
applied to the injected duals.

Usage (each mode in its own process, SERIALLY -- the box is shared and wall time
is contention-sensitive; iteration counts are the contention-proof metric)::

    python -u benchmarks/dual_warmstart_proto.py --selfcheck
    python -u benchmarks/dual_warmstart_proto.py --mode base
    python -u benchmarks/dual_warmstart_proto.py --mode zeroy
    python -u benchmarks/dual_warmstart_proto.py --mode ipdual
    python -u benchmarks/dual_warmstart_proto.py --mode ipdualcold
    python -u benchmarks/dual_warmstart_proto.py --mode c2ffirst
    python -u benchmarks/dual_warmstart_proto.py --summary

Findings + the verdict: ``docs/superpowers/notes/dual-warmstart-findings.md``.
Short version: (a) already happens and is worth 2.3x; (b) is structurally vacuous
(the coarse solve's terminal duals are all zero); (c) is a real gap whose per-QP
win the SQP loop refuses to bank. Nothing promoted.
"""

import os

# Thread pinning must precede numpy / osqp / clarabel import.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import osqp  # noqa: E402

import dvfopt.core.primitives.isqp as _isqp_mod  # noqa: E402
import dvfopt.core.windowed._common as _common  # noqa: E402
from dvfopt.constraints import SimplexConstraint2DBilinear  # noqa: E402
from dvfopt.core.primitives.tri import _triangle_areas_2d  # noqa: E402
from dvfopt.core.windowed import min_field, windowed_correct  # noqa: E402
from dvfopt.objectives import NoneObjective  # noqa: E402

VOL = Path(
    r"C:\Users\Andy\Documents\GitHub\UCI-iGravi\deformation-field-processing"
    r"\data\dvfs\b0039\b0039_laplacian_deformation_field.npy"
)
OUT = Path(__file__).resolve().parent / "results" / "dual_warmstart"
THR = 0.01
MAXITER = 600
Z = 16


# ---------------------------------------------------------------------------
# Spy state
# ---------------------------------------------------------------------------


class Spy:
    """Per-run capture state shared by the two monkeypatches."""

    def __init__(self):
        self.mode = "base"
        self.scale = 1.0
        self.phase = "fine"
        self.factor = 4
        self.records = []  # one per solve_window_inner call
        self.cur = None
        self.pending_y = None
        self.coarse_dual = None  # (4, Hc-1, Wc-1) -- the one actually injected
        self.coarse_seen = None
        self.coarse_first = None  # diagnostics: duals of the coarse cold QP
        self.coarse_last = None  # diagnostics: duals at the coarse solution
        self.injected = 0
        self.inject_rows = 0
        self.inject_hits = 0
        self.ip_dual_seeds = 0


SPY = Spy()
_T0 = time.perf_counter()


def _note(msg):
    """Progress to stderr so a long run is observable (stdout stays pure JSON)."""
    print(f"  [{time.perf_counter() - _T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _enforced_of(sub):
    """The window's enforced constraint-row indices (patch-local).

    ``WindowSub`` does not carry them -- ``build_subproblem`` closes over
    ``enforced_idx`` inside ``cons``. Read it back out of the closure rather than
    re-deriving (exact, and free).
    """
    fv = dict(zip(sub.cons.__code__.co_freevars, (c.cell_contents for c in sub.cons.__closure__)))
    enf = np.asarray(fv["enforced_idx"])
    assert enf.size == sub.n_enforced, "closure enforced_idx disagrees with n_enforced"
    return enf


def _rows_to_cells(enf, patch_box, pw):
    """Enforced rows -> ``(block, global_cell_i, global_cell_j)`` (see module doc)."""
    py0, py1, px0, _px1 = patch_box
    ph = py1 - py0
    m = (ph - 1) * (pw - 1)
    b, cf = enf // m, enf % m
    return b, py0 + cf // (pw - 1), px0 + cf % (pw - 1)


# ---------------------------------------------------------------------------
# Monkeypatch 1: wrap the QP object (records ADMM iters, injects / zeroes y)
# ---------------------------------------------------------------------------

_orig_make_qp = _isqp_mod._make_qp


class _SpyQP:
    """Pass-through wrapper over the engine's QP object with dual instrumentation."""

    def __init__(self, inner):
        self._inner = inner
        self.m_total = None

    def _osqp(self):
        # 'hybrid' hands back _HybridQP, which keeps the real OSQP as ._real.
        return self._inner if isinstance(self._inner, osqp.OSQP) else self._inner._real

    def setup(self, p, q, a, lo, up, **kw):
        self.m_total = a.shape[0]
        self._inner.setup(p, q, a, lo, up, **kw)
        y0 = SPY.pending_y
        SPY.pending_y = None
        if y0 is not None and y0.size == self.m_total:
            self._osqp().warm_start(y=y0)
            SPY.injected += 1

    def update(self, **kw):
        self._inner.update(**kw)

    def solve(self):
        if SPY.mode == "zeroy":
            self._osqp().warm_start(y=np.zeros(self.m_total))
        res = self._inner.solve()
        rec = SPY.cur
        if rec is not None:
            status = str(getattr(res.info, "status", "?"))
            rec["solves"].append((int(res.info.iter), status.startswith("clarabel")))
            y = getattr(res, "y", None)  # _HybridQP's IP result carries no dual
            if y is not None:
                rec["last_y"] = np.asarray(y, dtype=np.float64).copy()
                if rec["first_y"] is None:
                    rec["first_y"] = rec["last_y"]
        return res


def _spy_make_qp(qp_backend, ip_cold, ip_after_admm_iters):
    return _SpyQP(_orig_make_qp(qp_backend, ip_cold, ip_after_admm_iters))


# ---------------------------------------------------------------------------
# Monkeypatch 2: wrap the window inner (window bookkeeping + dual injection)
# ---------------------------------------------------------------------------

_orig_swi = _common.solve_window_inner


def _build_fine_y(sub, enf):
    """Fine-window initial ``y``: each fine row takes its parent coarse cell's dual."""
    if SPY.coarse_dual is None:
        return None
    py0, py1, px0, px1 = sub.patch_box
    pw = px1 - px0
    b, gi, gj = _rows_to_cells(enf, sub.patch_box, pw)
    f = SPY.factor
    _nb, hc, wc = SPY.coarse_dual.shape
    ci = np.clip(gi // f, 0, hc - 1)
    cj = np.clip(gj // f, 0, wc - 1)
    yc = SPY.coarse_dual[b, ci, cj] * SPY.scale
    SPY.inject_rows += enf.size
    SPY.inject_hits += int((SPY.coarse_seen[b, ci, cj]).sum())
    m = enf.size
    nf = int(np.asarray(sub.free_idx).size)
    y0 = np.zeros(2 * m + nf)  # [constraint rows | slack bounds | trust-region box]
    y0[:m] = yc
    return y0


def _spy_swi(sub, inner, maxiter, **kw):
    prev = SPY.cur
    # A "new window" = not an immediate retry of the same patch (the no-TR and
    # backend fallbacks re-enter with the same patch_box; grow-on-failure and the
    # giant tiler's next tile do not).
    first = prev is None or prev["phase"] != SPY.phase or prev["patch_box"] != sub.patch_box
    enf = _enforced_of(sub)
    rec = {
        "phase": SPY.phase,
        "patch_box": tuple(sub.patch_box),
        "n_enf": int(sub.n_enforced),
        "nf": int(np.asarray(sub.free_idx).size),
        "first": bool(first),
        "solves": [],
        "last_y": None,
        "first_y": None,
        "enf": enf,
    }
    SPY.cur = rec
    SPY.records.append(rec)
    if first:
        _note(
            f"[{SPY.phase}] window {sub.patch_box} m={sub.n_enforced} nf={rec['nf']} "
            f"(#{sum(r['first'] for r in SPY.records)})"
        )
    if SPY.mode in ("c2f", "c2ffirst") and SPY.phase == "fine" and first:
        SPY.pending_y = _build_fine_y(sub, enf)
    return _orig_swi(sub, inner, maxiter, **kw)


# ---------------------------------------------------------------------------
# Monkeypatch 3: mark the coarse stage and harvest its duals when it ends
# ---------------------------------------------------------------------------

_orig_cws = _common._coarse_warm_start


def _harvest_coarse(shape_c, which="last_y"):
    """(4, Hc-1, Wc-1) coarse row duals, last writer wins across rounds/sweeps.

    ``which='last_y'`` takes each coarse window's FINAL QP duals (the dual at the
    coarse solution); ``'first_y'`` takes its FIRST ADMM solve's duals (the dual of
    the cold, heavily-violated linearization). They are very different objects --
    see the findings note.
    """
    hc, wc = shape_c
    dual = np.zeros((4, hc - 1, wc - 1))
    seen = np.zeros((4, hc - 1, wc - 1), bool)
    for rec in SPY.records:
        if rec["phase"] != "coarse" or rec[which] is None:
            continue
        y = rec[which][: rec["n_enf"]]
        py0, py1, px0, px1 = rec["patch_box"]
        b, gi, gj = _rows_to_cells(rec["enf"], rec["patch_box"], px1 - px0)
        ok = (gi < hc - 1) & (gj < wc - 1)  # _restrict drops a partial trailing block
        dual[b[ok], gi[ok], gj[ok]] = y[ok]
        seen[b[ok], gi[ok], gj[ok]] = True
    return dual, seen


def _spy_cws(phi, constraint, objective, threshold, factor, margin, ring, inner, sub_kw):
    SPY.phase, SPY.factor = "coarse", factor
    try:
        return _orig_cws(phi, constraint, objective, threshold, factor, margin, ring, inner, sub_kw)
    finally:
        SPY.phase, SPY.cur = "fine", None
        shape_c = (phi.shape[1] // factor, phi.shape[2] // factor)
        which = "first_y" if SPY.mode == "c2ffirst" else "last_y"
        SPY.coarse_dual, SPY.coarse_seen = _harvest_coarse(shape_c, which)
        SPY.coarse_last, _ = _harvest_coarse(shape_c, "last_y")
        SPY.coarse_first, _ = _harvest_coarse(shape_c, "first_y")
        _note(
            f"coarse stage done ({which}): "
            f"{sum(r['phase'] == 'coarse' for r in SPY.records)} inner calls, "
            f"{int(SPY.coarse_seen.sum())} coarse rows captured "
            f"({100 * SPY.coarse_seen.mean():.2f}% of the coarse grid); "
            f"nonzero duals: first {int((SPY.coarse_first != 0).sum())} / "
            f"last {int((SPY.coarse_last != 0).sum())}"
        )


# ---------------------------------------------------------------------------
# Monkeypatch 4 ('ipdual'): let the interior-point leg hand OSQP its DUAL too
# ---------------------------------------------------------------------------
#
# The shipped ``_HybridQP._solve_ip`` ends with ``self._real.warm_start(x=x)`` --
# it gives OSQP the Clarabel PRIMAL and leaves the dual at whatever it was
# (zero, on the window's cold first solve). The micro-benchmark in the findings
# note says that is the half that does not matter: zeroing ``y`` alone reverts a
# warm resolve to full cold cost, while zeroing ``x`` alone costs nothing. So
# hand over Clarabel's dual as well.
#
# Convention. ``_solve_ip`` builds ``A_ip = [A[fu]; -A[fl]]``, ``b_ip = [u[fu];
# -l[fl]]`` with ``A_ip z + s = b_ip``, ``s >= 0``, ``z >= 0``, whose
# stationarity is ``Px + q + A[fu]' z_u - A[fl]' z_l = 0``. OSQP's is
# ``Px + q + A' y = 0``. Hence ``y[fu] += z_u`` and ``y[fl] -= z_l`` -- and for
# the engine's constraint rows (``l = -c``, ``u = +inf``) that yields ``y <= 0``,
# which is exactly OSQP's sign for a lower-bound-active row.


def _solve_ip_with_dual(self):
    import clarabel
    from scipy import sparse

    try:
        fu, fl = np.isfinite(self._up), np.isfinite(self._lo)
        a_csr = self._a.tocsr()
        a_ip = sparse.vstack([a_csr[fu], -a_csr[fl]], format="csc")
        b_ip = np.concatenate([self._up[fu], -self._lo[fl]])
        st = clarabel.DefaultSettings()
        st.verbose = False
        st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-3
        sol = clarabel.DefaultSolver(
            self._p, self._q, a_ip, b_ip, [clarabel.NonnegativeConeT(b_ip.size)], st
        ).solve()
        x = np.asarray(sol.x, dtype=np.float64)
        if str(sol.status) != "Solved" or not np.all(np.isfinite(x)):
            return None
        z = np.asarray(sol.z, dtype=np.float64)
        y = np.zeros(self._a.shape[0])
        nu = int(fu.sum())
        y[fu] += z[:nu]
        y[fl] -= z[nu:]
        # 'ipdualcold' seeds ONLY the window's cold first IP solve (where OSQP's
        # dual is literal zero and the per-solve saving is largest), leaving the
        # tail-triggered IP solves exactly as shipped -- far fewer seeded solves,
        # so far less perturbation of the SQP trajectory. `solve()` sets
        # _last_admm_iters AFTER calling us, so None here still means "cold".
        cold_only = SPY.mode == "ipdualcold" and self._last_admm_iters is not None
        if not cold_only and np.all(np.isfinite(y)):
            self._real.warm_start(x=x, y=y)  # <-- the whole point of this mode
            SPY.ip_dual_seeds += 1
        else:
            self._real.warm_start(x=x)
        from types import SimpleNamespace

        return SimpleNamespace(
            x=x,
            info=SimpleNamespace(iter=int(sol.iterations), status=f"clarabel-{sol.status}"),
        )
    except Exception as exc:
        _note(f"ipdual: interior-point solve failed ({exc!r}); falling through to ADMM")
        return None


def install():
    _isqp_mod._make_qp = _spy_make_qp
    _common.solve_window_inner = _spy_swi
    _common._coarse_warm_start = _spy_cws
    if SPY.mode in ("ipdual", "ipdualcold"):
        _isqp_mod._HybridQP._solve_ip = _solve_ip_with_dual


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def simplex_folds(phi):
    return int((np.minimum(*_triangle_areas_2d(phi[0], phi[1])) < THR).sum())


def _iter_stats(phase):
    """ADMM / IP iteration totals for one phase.

    ``post_ip_admm_iters`` is the size of the opportunity the ``ipdual`` mode
    targets: ADMM iterations spent on solves that come DIRECTLY after an
    interior-point solve, i.e. the ones whose OSQP dual the IP leg leaves stale
    (it warm-starts ``x`` only). ``cold_admm_iters`` is the same for the very
    first ADMM solve of a window, whose dual starts at literal zero.
    """
    admm = ip = first_admm = n_first = 0
    post_ip = n_post_ip = n_ip = cold_admm = n_cold = 0
    for rec in SPY.records:
        if rec["phase"] != phase:
            continue
        seen_admm = prev_ip = False
        for it, is_ip in rec["solves"]:
            if is_ip:
                ip += it
                n_ip += 1
            else:
                admm += it
                if prev_ip:
                    post_ip += it
                    n_post_ip += 1
                if not seen_admm:
                    cold_admm += it
                    n_cold += 1
                    if rec["first"]:
                        first_admm += it
                        n_first += 1
                seen_admm = True
            prev_ip = is_ip
    return {
        f"{phase}_admm_iters": admm,
        f"{phase}_ip_iters": ip,
        f"{phase}_n_ip_solves": n_ip,
        f"{phase}_post_ip_admm_iters": post_ip,
        f"{phase}_n_post_ip_solves": n_post_ip,
        f"{phase}_cold_admm_iters": cold_admm,
        f"{phase}_n_cold_solves": n_cold,
        f"{phase}_first_admm_iters": first_admm,
        f"{phase}_n_first_solves": n_first,
    }


def _spread(y, rho=1e3):
    """Where a dual vector sits between the two bang-bang values 0 and -rho."""
    z = np.abs(y) < 1e-6
    r = np.abs(np.abs(y) - rho) < 1.0
    return {
        "n": int(y.size),
        "frac_zero": float(z.mean()),
        "frac_rho": float(r.mean()),
        "frac_interior": float((~(z | r)).mean()),
        "absmax": float(np.abs(y).max(initial=0.0)),
        "absmean": float(np.abs(y).mean()) if y.size else 0.0,
    }


def _dual_diagnostics():
    """Is there any coarse dual information to transfer, and does it predict the fine one?

    Three questions, one pass over the captured duals:

    1. What do the COARSE duals look like at the coarse solution (``last_y``) vs at
       the coarse cold linearization (``first_y``)? A converged SQP's last QP is the
       trivial ``d = 0`` one, so its duals are ~0 -- which would make transfer vacuous.
    2. Same for the fine windows.
    3. Regression of each fine window's FIRST-QP dual on the mapped coarse dual --
       the only pairing where a transfer could possibly help.
    """
    out = {}
    for name, arr in (("coarse_last", SPY.coarse_last), ("coarse_first", SPY.coarse_first)):
        if arr is not None:
            out[name] = _spread(arr.ravel())
    for tag in ("first_y", "last_y"):
        ys = [
            r[tag][: r["n_enf"]] for r in SPY.records if r["phase"] == "fine" and r[tag] is not None
        ]
        if ys:
            out[f"fine_{tag}"] = _spread(np.concatenate(ys))
    if SPY.coarse_first is None:
        return out
    yc_all, yf_all = [], []
    for rec in SPY.records:
        if rec["phase"] != "fine" or rec["first_y"] is None or not rec["first"]:
            continue
        yf = rec["first_y"][: rec["n_enf"]]
        _py0, _py1, _px0, px1 = rec["patch_box"]
        b, gi, gj = _rows_to_cells(rec["enf"], rec["patch_box"], px1 - rec["patch_box"][2])
        _nb, hc, wc = SPY.coarse_first.shape
        f = SPY.factor
        idx = (b, np.clip(gi // f, 0, hc - 1), np.clip(gj // f, 0, wc - 1))
        yc_all.append(SPY.coarse_first[idx])
        yf_all.append(yf)
    if yf_all:
        yc, yf = np.concatenate(yc_all), np.concatenate(yf_all)
        den = float(yc @ yc)
        out["fine_first_vs_coarse_first"] = {
            "n_rows": int(yf.size),
            "ls_slope_fine_over_coarse": (float(yc @ yf / den) if den > 0 else None),
            "corr": (
                float(np.corrcoef(yc, yf)[0, 1]) if yc.std() > 1e-12 and yf.std() > 1e-12 else None
            ),
            "coarse_nonzero_frac": float((np.abs(yc) > 1e-6).mean()),
        }
    return out


def _nenf(phase):
    return [r["n_enf"] for r in SPY.records if r["phase"] == phase and r["first_y"] is not None]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def load_slice(crop=None):
    sl = np.array(np.load(VOL, mmap_mode="r")[1:, Z], dtype=np.float64)
    return sl[:, :crop, :crop] if crop else sl


def run(mode, scale, crop):
    SPY.mode, SPY.scale = mode, scale
    phi = load_slice(crop)
    H, W = phi.shape[1:]
    c = SimplexConstraint2DBilinear(shape=(H, W))
    t = time.perf_counter()
    out, rep = windowed_correct(
        phi, "isqp", constraint=c, objective=NoneObjective(), threshold=THR, maxiter=MAXITER
    )
    wall = time.perf_counter() - t
    move = out - phi
    res = {
        "mode": mode,
        "scale": scale,
        "crop": crop,
        "shape": [H, W],
        "wall_s": wall,
        "coarse_solve_s": rep.coarse_solve_s,
        "fine_s": wall - rep.coarse_solve_s,
        "coarse_sqp_iters": rep.coarse_iters,
        "fine_sqp_iters": int(sum(w.inner_iters for w in rep.windows)),
        "n_windows": rep.n_windows,
        "rounds": rep.rounds,
        "giant_regions": rep.giant_regions,
        "mop_windows": rep.mop_windows,
        "backend_fallbacks": rep.backend_fallbacks,
        "folds_before_simplex": simplex_folds(phi),
        "folds_after_simplex": simplex_folds(out),
        "folds_before_bilinear": rep.folds_before,
        "folds_after_bilinear": rep.folds_after,
        "warm_folds": rep.warm_folds,
        "coarse_folds_before": rep.coarse_folds_before,
        "coarse_folds_after": rep.coarse_folds_after,
        "damage": rep.damage,
        "residual_in_window": rep.residual_in_window,
        "min_after_bilinear": float(min_field(c, out).min()),
        "l1_move": float(np.abs(move).sum()),
        "l2_move": float(np.linalg.norm(move.ravel())),
        "injected_windows": SPY.injected,
        "inject_rows": SPY.inject_rows,
        "inject_coarse_hit_frac": (SPY.inject_hits / SPY.inject_rows) if SPY.inject_rows else None,
        "ip_dual_seeds": SPY.ip_dual_seeds,
    }
    res.update(_iter_stats("coarse"))
    res.update(_iter_stats("fine"))
    res["dual_diagnostics"] = _dual_diagnostics()
    return res


# ---------------------------------------------------------------------------
# Self-check (the one runnable check behind the row mapping)
# ---------------------------------------------------------------------------


def selfcheck():
    """Assert the block-major row layout the dual mapping depends on."""
    H, W = 6, 7
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.05, (2, H, W))
    c = SimplexConstraint2DBilinear(shape=(H, W))
    m = (H - 1) * (W - 1)
    v0 = np.asarray(c.values(c.flatten(phi)))
    assert v0.size == 4 * m, v0.size
    # Perturb pixel (pi, pj): only cells touching it may change, i.e. cell rows
    # (i, j) with i in {pi-1, pi} and j in {pj-1, pj}, in EVERY one of the 4 blocks.
    for pi, pj in ((0, 0), (2, 3), (H - 1, W - 1)):
        p2 = phi.copy()
        p2[0, pi, pj] += 0.3
        changed = np.abs(np.asarray(c.values(c.flatten(p2))) - v0) > 1e-12
        b, cf = np.nonzero(changed)[0] // m, np.nonzero(changed)[0] % m
        ci, cj = cf // (W - 1), cf % (W - 1)
        assert np.all(np.abs(ci - pi + 0.5) <= 0.5), (pi, pj, ci)
        assert np.all(np.abs(cj - pj + 0.5) <= 0.5), (pi, pj, cj)
        assert set(b.tolist()) <= {0, 1, 2, 3}
    # And the reverse map used by _rows_to_cells agrees for a whole-grid "patch".
    enf = np.arange(4 * m)
    b, gi, gj = _rows_to_cells(enf, (0, H, 0, W), W)
    assert (b == enf // m).all() and (gi == (enf % m) // (W - 1)).all()
    assert (gj == (enf % m) % (W - 1)).all()
    # Restriction rescaling keeps triangle areas dimensionless (the scaling claim).
    big = rng.normal(0, 0.3, (2, 64, 64))
    a_f = np.minimum(*_triangle_areas_2d(big[0], big[1])).mean()
    co = _common._restrict(big, 4)
    a_c = np.minimum(*_triangle_areas_2d(co[0], co[1])).mean()
    assert abs(a_f - 0.5) < 0.2 and abs(a_c - 0.5) < 0.2, (a_f, a_c)
    _check_ip_dual_map()
    print(
        "selfcheck OK: block-major rows, cell-major within block, areas scale-free, "
        "clarabel->osqp dual map exact"
    )


def _check_ip_dual_map():
    """The 'ipdual' mapping must reproduce OSQP's own dual on the engine's row stack.

    Same row shape the engine builds: ``m`` lower-bounded-only constraint rows
    (``u = +inf``) stacked over ``n`` two-sided trust-region box rows.
    """
    import clarabel
    from scipy import sparse

    rng = np.random.default_rng(3)
    n, m = 30, 45
    p_mat = sparse.diags(np.full(n, 2.0)).tocsc()
    q = rng.normal(size=n)
    a = sparse.random(m, n, density=0.25, random_state=2, format="csc")
    a.data[:] = rng.normal(size=a.nnz)
    a_full = sparse.vstack([a, sparse.eye(n, format="csc")], format="csc")
    lo = np.concatenate([-rng.random(m), np.full(n, -2.0)])
    up = np.concatenate([np.full(m, np.inf), np.full(n, 2.0)])
    prob = osqp.OSQP()
    prob.setup(
        p_mat, q, a_full, lo, up, verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=200_000
    )
    y_osqp = np.asarray(prob.solve().y)

    fu, fl = np.isfinite(up), np.isfinite(lo)
    a_csr = a_full.tocsr()
    a_ip = sparse.vstack([a_csr[fu], -a_csr[fl]], format="csc")
    b_ip = np.concatenate([up[fu], -lo[fl]])
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-10
    sol = clarabel.DefaultSolver(
        p_mat, q, a_ip, b_ip, [clarabel.NonnegativeConeT(b_ip.size)], st
    ).solve()
    z = np.asarray(sol.z)
    y = np.zeros(a_full.shape[0])
    nu = int(fu.sum())
    y[fu] += z[:nu]
    y[fl] -= z[nu:]
    assert np.allclose(y, y_osqp, atol=1e-6), np.abs(y - y_osqp).max()
    assert y[:m].max() <= 1e-9, y[:m].max()  # lower-bounded rows -> non-positive dual


# ---------------------------------------------------------------------------


def summary():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    if not rows:
        print("no results yet")
        return
    cols = [
        ("mode", "{}"),
        ("wall_s", "{:.0f}"),
        ("fine_s", "{:.0f}"),
        ("fine_sqp_iters", "{}"),
        ("coarse_sqp_iters", "{}"),
        ("fine_admm_iters", "{}"),
        ("fine_post_ip_admm_iters", "{}"),
        ("fine_cold_admm_iters", "{}"),
        ("fine_ip_iters", "{}"),
        ("folds_after_simplex", "{}"),
        ("folds_after_bilinear", "{}"),
        ("damage", "{}"),
        ("l2_move", "{:.1f}"),
    ]
    print(" | ".join(c for c, _ in cols))
    for r in rows:
        print(" | ".join(f.format(r.get(c)) if r.get(c) is not None else "-" for c, f in cols))
    for r in rows:
        if r.get("dual_diagnostics"):
            print(f"\ndual diagnostics ({r['mode']}):", json.dumps(r["dual_diagnostics"], indent=2))


MODES = ("base", "zeroy", "c2f", "c2ffirst", "ipdual", "ipdualcold")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES, default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on injected duals")
    ap.add_argument("--crop", type=int, default=None, help="crop to NxN for a fast smoke run")
    ap.add_argument("--tag", default="", help="suffix for the result filename")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if a.summary:
        return summary()
    if a.mode is None:
        ap.error("--mode required (or --selfcheck / --summary)")
    SPY.mode, SPY.scale = a.mode, a.scale  # install() branches on the mode
    install()
    res = run(a.mode, a.scale, a.crop)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{a.mode}{a.tag}" + (f"_crop{a.crop}" if a.crop else "")
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
