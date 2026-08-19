"""Part XXIII probe: can the ~33 geometric-floor folds left by
correct_dvf_25d on B0039 be eliminated by a MIXED (per-cell) tet
decomposition?

Input: output/b0039_FULL_marching25d_mop3d.npy  (3, 528, 320, 456), dz==0,
33 residual negative 6-tet volumes under the fixed (C0,C7) Kuhn diagonal.

Phases
------
A. Banded full-volume scan with the fused all-diagonals kernel
   (six_tet_volumes_all_diagonals): locate every cell whose fixed-diagonal
   min tet volume is <= 0, and compute the global best-diagonal
   ("exists a positive triangulation") fold count.
B. Per-cell detail: per-tet volumes under each of the 4 diagonals,
   feasible-diagonal set, negative-tet counts, connected clusters,
   and the feasible-diagonal sets of the 6 face neighbours.
C. Conformity analysis (derived from the tet tables, then verified):
   - induced face diagonal per (cube diagonal, face),
   - pairwise compatibility rule across a shared face,
   - GF(2) proof that a finite-support conforming flip region around a
     single cell does NOT exist (minimal supports are full axis planes),
   - verification that full-plane flips ARE conforming.
D. Conforming plane-flip cost: for each cell fixable by diagonal d, count
   how many currently-feasible cells the mandatory plane flip would break.
E. Minimal-displacement probe for cells NOT fixable under any diagonal:
   per-cell SLSQP, in-plane (dy,dx; dz==0 preserved) and full-3D,
   all-8-corner and best-single-corner variants, under the best diagonal
   (and fixed diagonal for reference).

Writes output/mixed_decomp_floor33.json and prints a markdown-ready
per-cell table. Read-only with respect to the field: no .npy is modified.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # repo root

from dvfopt.jacobian.tetrahedron_sign import (
    _ALL_DIAG_SIGNS,
    _ALL_DIAG_TETS,
    _MAIN_DIAGONALS,
    six_tet_volumes_all_diagonals,
)

DEFAULT_INPUT = HERE / 'output' / 'b0039_FULL_marching25d_mop3d.npy'
FALLBACK_INPUT = Path(
    r'C:\Users\Andy\Documents\GitHub\UCI-iGravi\deformation-field-processing'
    r'\research\strict_feasibility_3d\runners\output\b0039_FULL_marching25d_mop3d.npy'
)
OUT_JSON = HERE / 'output' / 'mixed_decomp_floor33.json'

CORNER_OFFSETS = np.array(
    [[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], dtype=np.int64
)  # (8, 3) as (oz, oy, ox)


# ---------------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------------

def cell_corner_positions(vol, cz, cy, cx):
    """Warped (z, y, x) positions of the 8 corners of cell (cz, cy, cx)."""
    P = np.empty((8, 3), dtype=np.float64)
    for i in range(8):
        oz, oy, ox = CORNER_OFFSETS[i]
        z, y, x = cz + oz, cy + oy, cx + ox
        P[i, 0] = z + vol[0, z, y, x]
        P[i, 1] = y + vol[1, z, y, x]
        P[i, 2] = x + vol[2, z, y, x]
    return P


def tet_volumes_from_positions(P):
    """(4, 6) signed tet volumes (identity-normalised) for one cell."""
    out = np.empty((4, 6), dtype=np.float64)
    for d in range(4):
        for k in range(6):
            i0, i1, i2, i3 = _ALL_DIAG_TETS[d, k]
            AB = P[i1] - P[i0]
            AC = P[i2] - P[i0]
            AD = P[i3] - P[i0]
            det = (
                AB[0] * (AC[1] * AD[2] - AC[2] * AD[1])
                - AB[1] * (AC[0] * AD[2] - AC[2] * AD[0])
                + AB[2] * (AC[0] * AD[1] - AC[1] * AD[0])
            )
            out[d, k] = _ALL_DIAG_SIGNS[d, k] * det / 6.0
    return out


# ---------------------------------------------------------------------------
# Phase A: banded full-volume all-diagonals scan
# ---------------------------------------------------------------------------

def phase_a(vol, band=48):
    D, H, W = vol.shape[1:]
    n_fix_neg = 0
    n_best_neg = 0
    min_fix = np.inf
    min_best = np.inf
    residual = []  # (cz, cy, cx, mins[4])
    t0 = time.time()
    for z0 in range(0, D - 1, band):
        z1 = min(z0 + band, D - 1)  # cells z0..z1-1 need slices z0..z1
        sub = np.ascontiguousarray(vol[:, z0:z1 + 1])
        ad = six_tet_volumes_all_diagonals(sub)  # (4, z1-z0, H-1, W-1)
        fix = ad[0]
        best = ad.max(axis=0)
        n_fix_neg += int((fix <= 0).sum())
        n_best_neg += int((best <= 0).sum())
        min_fix = min(min_fix, float(fix.min()))
        min_best = min(min_best, float(best.min()))
        zz, yy, xx = np.nonzero(fix <= 0)
        for lz, cy, cx in zip(zz, yy, xx):
            residual.append((int(z0 + lz), int(cy), int(cx),
                             ad[:, lz, cy, cx].copy()))
        print(f'  band z[{z0}:{z1}] done ({time.time()-t0:.1f}s) '
              f'fix_neg so far={n_fix_neg}', flush=True)
    return dict(n_cells_neg_fixed=n_fix_neg, n_cells_neg_best=n_best_neg,
                min_T_fixed=min_fix, min_T_best=min_best,
                residual=residual)


# ---------------------------------------------------------------------------
# Phase C: conformity machinery (all derived from _ALL_DIAG_TETS)
# ---------------------------------------------------------------------------

AXIS_SHIFT = {'z': 2, 'y': 1, 'x': 0}  # corner-bit shift per axis


def induced_face_diagonal(d, axis, side):
    """Face diagonal (frozenset of 2 local corners) induced on face
    (axis, side) by cube-diagonal d's Kuhn fan. Derived from the tet
    table: the edge shared by the two face triangles."""
    sh = AXIS_SHIFT[axis]
    face = frozenset(i for i in range(8) if ((i >> sh) & 1) == side)
    tris = []
    for k in range(6):
        verts = [int(v) for v in _ALL_DIAG_TETS[d, k]]
        for tri in combinations(verts, 3):
            if frozenset(tri) <= face:
                tris.append(frozenset(tri))
    tris = list(set(tris))
    assert len(tris) == 2, (d, axis, side, tris)
    shared = tris[0] & tris[1]
    assert len(shared) == 2
    a, b = sorted(shared)
    diff = a ^ b
    assert bin(diff).count('1') == 2 and not (diff >> sh) & 1, \
        f'shared edge is not a face diagonal: {a},{b}'
    return frozenset(shared)


def corner_global(base, i):
    oz, oy, ox = CORNER_OFFSETS[i]
    return (base[0] + oz, base[1] + oy, base[2] + ox)


def compat_table():
    """compat[axis][dL] = set of dR such that cube L (diag dL) and its
    +axis neighbour R (diag dR) induce the SAME global diagonal on the
    shared face. Verified in global coordinates."""
    table = {}
    for axis in 'zyx':
        sh = AXIS_SHIFT[axis]
        e = [0, 0, 0]
        e[{'z': 0, 'y': 1, 'x': 2}[axis]] = 1
        table[axis] = {}
        for dL in range(4):
            gL = frozenset(corner_global((0, 0, 0),
                                         i) for i in induced_face_diagonal(dL, axis, 1))
            ok = set()
            for dR in range(4):
                gR = frozenset(corner_global(tuple(e), i)
                               for i in induced_face_diagonal(dR, axis, 0))
                if gL == gR:
                    ok.add(dR)
            table[axis][dL] = ok
    return table


# Diagonal label -> (u, w) bits over GF(2).  Chosen so the linear
# conformity constraints below hold; verified against compat_table().
# d0=(0,0), and flips: along x only u may change; along y only w; along z
# u and w change together.  We DERIVE the bit assignment from the compat
# table rather than hard-coding it.

def derive_bits(compat):
    """Assign (u, w) in GF(2)^2 to each diagonal such that:
    across x: w equal (u free); across y: u equal (w free);
    across z: u^w equal (u free).  Returns dict d -> (u, w)."""
    # d=0 -> (0,0). Across x, compat['x'][0] = {0, dx} with dx != 0 -> dx=(1,0).
    dx = (compat['x'][0] - {0}).pop()
    dy = (compat['y'][0] - {0}).pop()
    dz = (compat['z'][0] - {0}).pop()
    bits = {0: (0, 0), dx: (1, 0), dy: (0, 1), dz: (1, 1)}
    assert len(bits) == 4
    # verify: the linear rules reproduce the compat table exactly
    for axis, rule in (('x', lambda a, b: a[1] == b[1]),
                       ('y', lambda a, b: a[0] == b[0]),
                       ('z', lambda a, b: (a[0] ^ a[1]) == (b[0] ^ b[1]))):
        for dL in range(4):
            pred = {dR for dR in range(4) if rule(bits[dL], bits[dR])}
            assert pred == compat[axis][dL], (axis, dL, pred, compat[axis][dL])
    return bits


def gf2_solvable(A, b):
    """Solve A x = b over GF(2); return True iff consistent."""
    A = A.copy() % 2
    b = b.copy() % 2
    m, n = A.shape
    row = 0
    for col in range(n):
        piv = None
        for r in range(row, m):
            if A[r, col]:
                piv = r
                break
        if piv is None:
            continue
        A[[row, piv]] = A[[piv, row]]
        b[[row, piv]] = b[[piv, row]]
        for r in range(m):
            if r != row and A[r, col]:
                A[r] ^= A[row]
                b[r] ^= b[row]
        row += 1
    for r in range(row, m):
        if b[r] and not A[r].any():
            return False
    # also rows beyond rank with zero A but b=1
    zero_rows = ~A.any(axis=1)
    return not bool((b[zero_rows] == 1).any())


def finite_flip_unsat(n=7):
    """On an n^3 cube lattice with boundary cells pinned to diag 0, is
    there ANY conforming assignment with the centre cell != diag 0?
    Constraints are linear over GF(2) in (u, w) per cell. Returns dict
    target_bits -> solvable(bool)."""
    idx = {}
    for z in range(n):
        for y in range(n):
            for x in range(n):
                idx[(z, y, x)] = len(idx)
    N = len(idx)
    rows = []
    rhs = []

    def add_row(pairs, b):
        r = np.zeros(2 * N, dtype=np.uint8)
        for var in pairs:
            r[var] ^= 1
        rows.append(r)
        rhs.append(b)

    U = lambda c: 2 * idx[c]
    Wv = lambda c: 2 * idx[c] + 1
    for (z, y, x) in idx:
        c = (z, y, x)
        if x + 1 < n:  # across x: w equal
            add_row([Wv(c), Wv((z, y, x + 1))], 0)
        if y + 1 < n:  # across y: u equal
            add_row([U(c), U((z, y + 1, x))], 0)
        if z + 1 < n:  # across z: u^w equal
            c2 = (z + 1, y, x)
            add_row([U(c), Wv(c), U(c2), Wv(c2)], 0)
    # boundary pinned to 0
    for c in idx:
        if 0 in c or (n - 1) in c:
            add_row([U(c)], 0)
            add_row([Wv(c)], 0)
    A0 = np.array(rows, dtype=np.uint8)
    b0 = np.array(rhs, dtype=np.uint8)
    centre = (n // 2,) * 3
    out = {}
    for tgt in [(1, 0), (0, 1), (1, 1)]:
        rows2 = [np.zeros(2 * N, dtype=np.uint8), np.zeros(2 * N, dtype=np.uint8)]
        rows2[0][U(centre)] = 1
        rows2[1][Wv(centre)] = 1
        A = np.vstack([A0, np.array(rows2, dtype=np.uint8)])
        b = np.concatenate([b0, np.array(tgt, dtype=np.uint8)])
        out[tgt] = gf2_solvable(A, b)
    return out


def verify_plane_flip_conforming(compat, bits, n=4):
    """Check that flipping one full axis plane to the matching diagonal is
    conforming, and report which plane orientation goes with which diag."""
    inv_bits = {v: k for k, v in bits.items()}
    # plane normal x (all cells with cx = x0): u flips -> diag inv_bits[(1,0)]
    # plane normal y: w flips -> inv_bits[(0,1)]
    # plane normal z: u and w flip -> inv_bits[(1,1)]
    mapping = {'x': inv_bits[(1, 0)], 'y': inv_bits[(0, 1)], 'z': inv_bits[(1, 1)]}
    for axis, d in mapping.items():
        ai = {'z': 0, 'y': 1, 'x': 2}[axis]
        assign = {}
        for z in range(n):
            for y in range(n):
                for x in range(n):
                    c = (z, y, x)
                    assign[c] = d if c[ai] == n // 2 else 0
        for c in assign:
            for adja, ax in ((('z', 0)), (('y', 1)), (('x', 2))):
                c2 = list(c)
                c2[ax] += 1
                c2 = tuple(c2)
                if c2 in assign:
                    assert assign[c2] in compat[adja][assign[c]], \
                        (axis, d, c, c2, assign[c], assign[c2])
    return mapping


# ---------------------------------------------------------------------------
# Phase D: conforming plane-flip cost
# ---------------------------------------------------------------------------

def plane_flip_cost(vol, cell, d, plane_axis):
    """Flip the full plane through `cell` with normal `plane_axis` to
    diagonal d (the unique conforming finite pattern containing the cell).
    Count cells in the plane that are non-positive under d, and among
    those how many were strictly positive under the fixed diagonal
    ("newly broken")."""
    cz, cy, cx = cell
    if plane_axis == 'x':
        sub = np.ascontiguousarray(vol[:, :, :, cx:cx + 2])
    elif plane_axis == 'y':
        sub = np.ascontiguousarray(vol[:, :, cy:cy + 2, :])
    else:
        sub = np.ascontiguousarray(vol[:, cz:cz + 2, :, :])
    ad = six_tet_volumes_all_diagonals(sub)  # (4, ...) plane of cells
    flipped = ad[d]
    fixed = ad[0]
    n_cells = int(np.prod(flipped.shape))
    n_neg_flipped = int((flipped <= 0).sum())
    n_newly_broken = int(((flipped <= 0) & (fixed > 0)).sum())
    return dict(plane_axis=plane_axis, plane_cells=n_cells,
                n_nonpos_under_d=n_neg_flipped, n_newly_broken=n_newly_broken,
                min_under_d=float(flipped.min()))


# ---------------------------------------------------------------------------
# Phase E: minimal-displacement probes (SLSQP)
# ---------------------------------------------------------------------------

def probe_min_displacement(P0, d, free_vertices, in_plane, eps=1e-3):
    """Minimal sum-of-squares corner displacement so that all 6 tets of
    diagonal d have volume >= eps. free_vertices: list of corner ids that
    may move; in_plane: if True only (y, x) coordinates may move (dz==0
    preserved). Returns (max vertex |delta|, total L2, per-vertex norms)
    or None if SLSQP fails."""
    coords = [1, 2] if in_plane else [0, 1, 2]
    dof = [(v, c) for v in free_vertices for c in coords]

    def positions(delta):
        P = P0.copy()
        for k, (v, c) in enumerate(dof):
            P[v, c] += delta[k]
        return P

    def cons(delta):
        V = tet_volumes_from_positions(positions(delta))
        return V[d] - eps

    def obj(delta):
        return float((delta ** 2).sum())

    # start 1: no movement. start 2 (fallback, all-corner in-plane/full-3D
    # only): displacement that restores the identity cube — guaranteed
    # feasible (every tet = +1/6), so SLSQP only has to shrink it.
    starts = [np.zeros(len(dof))]
    if len(free_vertices) == 8:
        ident = np.zeros(len(dof))
        base = P0[0].copy()  # anchor cube at corner 0's warped position
        for k, (v, c) in enumerate(dof):
            target = base[c] + CORNER_OFFSETS[v][c]
            ident[k] = target - P0[v, c]
        starts.append(ident)
    best = None
    for x0 in starts:
        res = minimize(obj, x0, method='SLSQP',
                       constraints=[dict(type='ineq', fun=cons)],
                       options=dict(maxiter=300, ftol=1e-12))
        if not res.success:
            continue
        V = tet_volumes_from_positions(positions(res.x))[d]
        if V.min() < eps - 1e-9:
            continue
        if best is None or obj(res.x) < obj(best.x):
            best = res
    if best is None:
        return None
    res = best
    per_vertex = {}
    for k, (v, c) in enumerate(dof):
        per_vertex.setdefault(v, 0.0)
        per_vertex[v] += res.x[k] ** 2
    norms = {v: float(np.sqrt(s)) for v, s in per_vertex.items()}
    return dict(max_vertex_move=max(norms.values()),
                total_l2=float(np.sqrt((res.x ** 2).sum())),
                per_vertex=norms)


def probe_cell(vol, cell, diag, eps=1e-3):
    """All-corner and best-single-corner minimal displacement, in-plane
    and full 3D, for feasibility of `diag` on `cell`."""
    P0 = cell_corner_positions(vol, *cell)
    out = {}
    for label, in_plane in (('inplane', True), ('full3d', False)):
        allv = probe_min_displacement(P0, diag, list(range(8)), in_plane, eps)
        best_single = None
        for v in range(8):
            r = probe_min_displacement(P0, diag, [v], in_plane, eps)
            if r is not None and (best_single is None
                                  or r['max_vertex_move'] < best_single['move']):
                best_single = dict(vertex=v, move=r['max_vertex_move'])
        out[label] = dict(all_corners=allv, best_single=best_single)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    inp = DEFAULT_INPUT if DEFAULT_INPUT.exists() else FALLBACK_INPUT
    if len(sys.argv) > 1:
        inp = Path(sys.argv[1])
    print(f'input: {inp}', flush=True)
    vol = np.load(inp)
    assert vol.ndim == 4 and vol.shape[0] == 3
    assert float(np.abs(vol[0]).max()) == 0.0, 'expected dz == 0'
    D, H, W = vol.shape[1:]
    print(f'volume: (3, {D}, {H}, {W}) {vol.dtype}', flush=True)

    # -- Phase A ------------------------------------------------------------
    print('\n=== Phase A: full-volume all-diagonals scan ===', flush=True)
    A = phase_a(vol)
    print(f'fixed-diag  : cells with min<=0: {A["n_cells_neg_fixed"]}, '
          f'min_T={A["min_T_fixed"]:.6f}')
    print(f'best-diag   : cells with min<=0: {A["n_cells_neg_best"]}, '
          f'min_T best={A["min_T_best"]:.6f}', flush=True)

    residual = A['residual']

    # -- Phase B ------------------------------------------------------------
    print('\n=== Phase B: per-cell detail ===', flush=True)
    cells = []
    total_neg_tets = 0
    coords_set = {(c[0], c[1], c[2]) for c in residual}
    for (cz, cy, cx, mins) in residual:
        P0 = cell_corner_positions(vol, cz, cy, cx)
        V = tet_volumes_from_positions(P0)  # (4, 6)
        n_neg_tets_fixed = int((V[0] <= 0).sum())
        total_neg_tets += n_neg_tets_fixed
        feas = [d for d in range(4) if V[d].min() > 0]
        best_d = int(np.argmax(V.min(axis=1)))
        # neighbours' feasible sets
        nbrs = {}
        for ax, (dz_, dy_, dx_) in (('z-', (-1, 0, 0)), ('z+', (1, 0, 0)),
                                    ('y-', (0, -1, 0)), ('y+', (0, 1, 0)),
                                    ('x-', (0, 0, -1)), ('x+', (0, 0, 1))):
            nz, ny, nx = cz + dz_, cy + dy_, cx + dx_
            if not (0 <= nz < D - 1 and 0 <= ny < H - 1 and 0 <= nx < W - 1):
                nbrs[ax] = None
                continue
            Vn = tet_volumes_from_positions(cell_corner_positions(vol, nz, ny, nx))
            nbrs[ax] = dict(feasible=[d for d in range(4) if Vn[d].min() > 0],
                            is_residual=(nz, ny, nx) in coords_set)
        cells.append(dict(cell=(cz, cy, cx),
                          min_per_diag=[float(V[d].min()) for d in range(4)],
                          n_neg_tets_fixed=n_neg_tets_fixed,
                          feasible_diags=feas, best_diag=best_d,
                          fixable=bool(feas), neighbours=nbrs))
    print(f'residual cells: {len(cells)}; total negative tets (fixed diag): '
          f'{total_neg_tets}', flush=True)

    # clusters (26-connectivity)
    coords = sorted(coords_set)
    parent = {c: c for c in coords}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for a in coords:
        for b in coords:
            if a < b and all(abs(a[i] - b[i]) <= 1 for i in range(3)):
                parent[find(a)] = find(b)
    clusters = {}
    for c in coords:
        clusters.setdefault(find(c), []).append(c)
    print(f'clusters (26-conn): {len(clusters)}, sizes: '
          f'{sorted((len(v) for v in clusters.values()), reverse=True)}', flush=True)

    # -- Phase C ------------------------------------------------------------
    print('\n=== Phase C: conformity structure ===', flush=True)
    compat = compat_table()
    for axis in 'zyx':
        print(f'  compat across {axis}: '
              + ', '.join(f'd{dL}->{sorted(compat[axis][dL])}' for dL in range(4)))
    bits = derive_bits(compat)
    print(f'  GF(2) labels (u,w): {bits}')
    unsat = finite_flip_unsat(n=7)
    print(f'  finite-support single-cell flip solvable (7^3, pinned boundary): '
          f'{unsat}')
    mapping = verify_plane_flip_conforming(compat, bits)
    print(f'  conforming plane flips: plane normal -> diagonal: {mapping}',
          flush=True)

    # -- Phase D ------------------------------------------------------------
    print('\n=== Phase D: conforming plane-flip cost per fixable cell ===',
          flush=True)
    axis_for_diag = {d: ax for ax, d in mapping.items()}
    for c in cells:
        c['plane_flips'] = {}
        for d in c['feasible_diags']:
            info = plane_flip_cost(vol, c['cell'], d, axis_for_diag[d])
            c['plane_flips'][d] = info
            print(f"  cell {c['cell']} diag{d} plane {info['plane_axis']}: "
                  f"{info['n_newly_broken']} newly broken of "
                  f"{info['plane_cells']} plane cells "
                  f"(min under d = {info['min_under_d']:.4f})", flush=True)

    # -- Phase E ------------------------------------------------------------
    print('\n=== Phase E: minimal-displacement probes ===', flush=True)
    for c in cells:
        targets = {('best', c['best_diag'])}
        targets.add(('fixed', 0))
        c['probes'] = {}
        for label, d in sorted(targets):
            pr = probe_cell(vol, c['cell'], d)
            c['probes'][f'{label}_d{d}'] = pr
            ip = pr['inplane']
            allv = ip['all_corners']
            bs = ip['best_single']
            print(f"  cell {c['cell']} diag{d} ({label}): in-plane "
                  f"all-corners max-move="
                  f"{allv['max_vertex_move']:.4f} px" if allv else
                  f"  cell {c['cell']} diag{d} ({label}): in-plane all-corners FAILED",
                  flush=True)
            if bs:
                print(f"      best single corner: v{bs['vertex']} "
                      f"move={bs['move']:.4f} px", flush=True)

    # -- summary table -------------------------------------------------------
    print('\n=== Per-cell markdown table ===\n', flush=True)
    print('| # | cell (z,y,x) | negT | min d0 | min d1 | min d2 | min d3 | '
          'feasible diags | fixable | probe: best-diag in-plane move (all / '
          '1-corner) |')
    print('|---|---|---:|---:|---:|---:|---:|---|---|---|')
    for i, c in enumerate(sorted(cells, key=lambda c: c['cell'])):
        m = c['min_per_diag']
        bd = c['best_diag']
        pr = c['probes'].get(f'best_d{bd}', {}).get('inplane', {})
        allv = pr.get('all_corners')
        bs = pr.get('best_single')
        alls = f"{allv['max_vertex_move']:.4f}" if allv else 'fail'
        bss = f"v{bs['vertex']}:{bs['move']:.4f}" if bs else '—'
        print(f"| {i} | {c['cell']} | {c['n_neg_tets_fixed']} "
              f"| {m[0]:.4f} | {m[1]:.4f} | {m[2]:.4f} | {m[3]:.4f} "
              f"| {c['feasible_diags']} | {'YES' if c['fixable'] else 'NO'} "
              f"| {alls} / {bss} |")

    # -- persist ---------------------------------------------------------------
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        input=str(inp),
        global_stats={k: A[k] for k in
                      ('n_cells_neg_fixed', 'n_cells_neg_best',
                       'min_T_fixed', 'min_T_best')},
        total_neg_tets_fixed=total_neg_tets,
        clusters=[sorted(v) for v in clusters.values()],
        compat={ax: {str(d): sorted(s) for d, s in compat[ax].items()}
                for ax in compat},
        gf2_bits={str(k): list(v) for k, v in bits.items()},
        finite_flip_solvable={str(k): bool(v) for k, v in unsat.items()},
        plane_for_diag={ax: d for ax, d in mapping.items()},
        cells=cells,
    )
    OUT_JSON.write_text(json.dumps(_clean(payload), indent=1))
    print(f'\nwrote {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
