"""Research prototype: the 3D 'all-tetrahedra' constraint (research branch only).

3D analogue of :class:`dvfopt.constraints.SimplexConstraint2DBilinear`, which
enforces *all four* triangles of a 2D cell (both diagonals) instead of the two
of one fixed diagonal.

Measurements, blind-spot numbers on B0039 and the promotion recommendation:
``docs/superpowers/notes/3d-all-tets-findings.md``. Headline: a `BarrierStrategy`
run that reaches *strict* 6-tet feasibility on a real block still hides 557
inverted cells (1.87%, worst -8.05) that the fixed-diagonal metric cannot see —
and at matched wall-clock, solving under these 24 rows instead leaves 8, with a
smaller move.

Derivation
----------
A hexahedral cell has 4 body diagonals. Each one defines a Kuhn/Freudenthal
6-tetrahedron split (the fan of 6 tets sharing that diagonal), so the four
splits give 4 x 6 = 24 tets. **All 24 are distinct**: every tet of the fan
around diagonal ``(s, e)`` contains both ``s`` and ``e``, plus a cube *edge*
disjoint from them; two different diagonals can never both sit inside one
4-vertex set, because the second diagonal would have to be the "edge" pair and
a body diagonal is not an edge. So the dedup is a no-op: **24 distinct tets**.

The cube also admits two 5-tet splits (central tet = one parity class of
vertices, plus 4 corner tets anchored at the other class). Those add 10 more
distinct tets — 2 central + 8 corner — for **34 distinct tets** in total. The
8 corner tets are exactly the 8 *corner Jacobians* of the trilinear map
(6 * V_corner = det of the three edge vectors at that vertex), i.e. 8 of the
27 Bezier coefficients of the trilinear cell's degree-(2,2,2) Jacobian
determinant.

Certificate strength (what this does and does not prove)
-------------------------------------------------------
In 2D the 4-triangle set is an *exact* certificate: the bilinear Jdet is
biaffine, so its cell minimum equals the min of the four corner values, each
twice a triangle area. **The 3D analogue is not exact.** The trilinear Jdet is
a degree-(2,2,2) polynomial whose minimum is not attained at the corners, and
none of the 34 tets reproduce its interior Bezier coefficients. What the
24-row constraint certifies is: *all four* piecewise-linear interpolants (one
per diagonal choice) are simultaneously orientation-preserving on the cell —
a strictly stronger and decomposition-independent statement than the fixed
6-tet one, and a necessary condition for the 34-row / trilinear statements.

Usage
-----
``AllTetConstraint3D(shape)`` is a drop-in ``Constraint`` (pack ``DX_FIRST``,
``dim`` 3, rows ``K * (D-1)(H-1)(W-1)``) whose first ``6 * n_cells`` rows are
byte-identical to :class:`~dvfopt.constraints.SimplexConstraint3D`. It runs on
any constraint-generic strategy (``BarrierStrategy``).

No sparse ``jacobian()`` — the SLSQP path is impractical at 3D scale anyway
(see ``SimplexConstraint3D.jacobian``'s own note) and barrier/SLP need only the
adjoint. Add one if an SLSQP strategy ever wants these rows.

Run ``python benchmarks/tet_all_constraint.py`` for the self-check
(derivation counts, exact agreement with the 6-tet rows, finite-difference
gradient check).
"""

from __future__ import annotations

import numpy as np

from dvfopt.constraints import Constraint, PhiPack, SimplexConstraint3D
from dvfopt.jacobian.tetrahedron_sign import (
    _MAIN_DIAGONALS,
    _TET_VERTICES,
    _cross,
    _phi_flat_to_dz_dy_dx,
    _tet_volume_from_vertices,
    _tets_for_diagonal,
    _voxel_corner_positions,
)

# Identity-cube corner positions, (8, 3) in (z, y, x) — corner i has offsets
# ((i>>2)&1, (i>>1)&1, i&1), matching the package-wide cube convention.
_ID_POS = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], dtype=np.float64)


def _parity_class(par: int) -> list[int]:
    return [v for v in range(8) if bin(v).count('1') % 2 == par]


def build_tet_table(include_5tet: bool = False):
    """Deduped tet-vertex table for the all-decompositions certificate.

    Returns ``(tets, signs)`` with ``tets`` of shape ``(K, 4)`` (cube-corner
    indices) and ``signs`` of shape ``(K,)`` normalised so the identity field
    gives ``+1/6`` for every row. Rows ``0..5`` are exactly the fixed 6-tet
    split of :class:`SimplexConstraint3D`, in its order.

    ``K == 24`` for the four body-diagonal Kuhn splits, ``K == 34`` with the
    two 5-tet splits added.
    """
    raw: list[tuple[int, ...]] = [tuple(int(x) for x in row) for row in _TET_VERTICES]
    for s, e in _MAIN_DIAGONALS[1:]:
        raw.extend(tuple(t) for t in _tets_for_diagonal(s, e))
    if include_5tet:
        for par in (0, 1):
            raw.append(tuple(_parity_class(par)))  # central tet
            for v in _parity_class(1 - par):  # 4 corner tets
                raw.append((v, v ^ 1, v ^ 2, v ^ 4))

    seen: set[frozenset[int]] = set()
    tets: list[tuple[int, ...]] = []
    for t in raw:
        key = frozenset(t)
        assert len(key) == 4, f'degenerate tet {t}'
        if key not in seen:
            seen.add(key)
            tets.append(t)

    signs = np.array(
        [1.0 if _tet_volume_from_vertices(*(_ID_POS[i] for i in t)) > 0 else -1.0 for t in tets],
        dtype=np.float64,
    )
    return np.asarray(tets, dtype=np.int64), signs


TETS_24, SIGNS_24 = build_tet_table(False)
TETS_34, SIGNS_34 = build_tet_table(True)


class AllTetConstraint3D(Constraint):
    """Every tet of every considered hex decomposition, ``V_k >= threshold``.

    Same decision vector as :class:`SimplexConstraint3D` (``[dx, dy, dz]``,
    DX_FIRST) — it *delegates* coerce/flatten/unflatten to one rather than
    subclassing it, deliberately: the 6-tet-only strategies gate with
    ``accepts_constraints = (SimplexConstraint3D,)``, so a subclass would be
    silently accepted and then silently solved on 6 of the 24 rows.

    Parameters
    ----------
    shape : (D, H, W)
    include_5tet : bool, default False
        ``False`` -> 24 rows/cell (the 4 body-diagonal 6-tet splits).
        ``True``  -> 34 rows/cell (adds the two 5-tet splits: 2 central tets
        + the 8 trilinear corner Jacobians).

    Output layout: ``[V_0.ravel(), ..., V_{K-1}.ravel()]``, cell-major within
    each tet — so ``values(f)[:6 * n_cells]`` equals
    ``SimplexConstraint3D.values(f)`` exactly.
    """

    pack = PhiPack.DX_FIRST
    dim = 3

    def __init__(self, shape, include_5tet: bool = False):
        super().__init__(shape)
        self.include_5tet = bool(include_5tet)
        self.tets, self.signs = (TETS_34, SIGNS_34) if include_5tet else (TETS_24, SIGNS_24)
        self._base = SimplexConstraint3D(self.shape)

    def coerce(self, phi):
        return self._base.coerce(phi)

    def flatten(self, phi):
        return self._base.flatten(phi)

    def unflatten(self, phi_flat):
        return self._base.unflatten(phi_flat)

    @property
    def n_variables(self) -> int:
        return self._base.n_variables

    @property
    def n_tets(self) -> int:
        return len(self.tets)

    @property
    def n_constraints(self) -> int:
        D, H, W = self.shape
        return self.n_tets * (D - 1) * (H - 1) * (W - 1)

    def values(self, phi_flat: np.ndarray) -> np.ndarray:
        D, H, W = self.shape
        pos = _voxel_corner_positions(*_phi_flat_to_dz_dy_dx(phi_flat, D, H, W))
        out = np.empty((self.n_tets, *pos.shape[2:]), dtype=np.float64)
        for k, (i0, i1, i2, i3) in enumerate(self.tets):
            out[k] = self.signs[k] * _tet_volume_from_vertices(pos[i0], pos[i1], pos[i2], pos[i3])
        return out.ravel()

    def adjoint(self, phi_flat: np.ndarray, v: np.ndarray) -> np.ndarray:
        D, H, W = self.shape
        pos = _voxel_corner_positions(*_phi_flat_to_dz_dy_dx(phi_flat, D, H, W))
        v_per_tet = v.reshape(self.n_tets, D - 1, H - 1, W - 1)
        # accumulators in (z, y, x) component order, matching pos's axis 1
        acc = [np.zeros((D, H, W)) for _ in range(3)]
        for k, (i0, i1, i2, i3) in enumerate(self.tets):
            A, B, C, Dv = pos[i0], pos[i1], pos[i2], pos[i3]
            AB, AC, AD = B - A, C - A, Dv - A
            coef = self.signs[k] / 6.0
            gB = coef * _cross(AC, AD)
            gC = coef * _cross(AD, AB)
            gD = coef * _cross(AB, AC)
            gA = -(gB + gC + gD)
            vk = v_per_tet[k]
            for corner, grad in zip((i0, i1, i2, i3), (gA, gB, gC, gD)):
                oz, oy, ox = (corner >> 2) & 1, (corner >> 1) & 1, corner & 1
                for comp in range(3):
                    acc[comp][oz : D - 1 + oz, oy : H - 1 + oy, ox : W - 1 + ox] += grad[comp] * vk
        g_dz, g_dy, g_dx = acc
        return np.concatenate([g_dx.ravel(), g_dy.ravel(), g_dz.ravel()])

    def jacobian(self, phi_flat: np.ndarray):
        return None  # barrier/SLP need only the adjoint; SLSQP is hopeless at 3D scale

    def __repr__(self) -> str:
        return f'AllTetConstraint3D(shape={self.shape}, n_tets={self.n_tets})'


Tet24Constraint3D = AllTetConstraint3D  # the 24-row default, by its derived count


# For the all-tet fold MAP (per-cell min over the 24 rows) call the library's
# already-fused `six_tet_volumes_all_diagonals(phi).min(axis=0)` — same numbers,
# one pass, no need for this class.


def _selfcheck() -> None:
    from dvfopt.jacobian.tetrahedron_sign import (
        six_tet_min_volume_3d,
        six_tet_volumes_all_diagonals,
    )

    # --- derivation -------------------------------------------------------
    assert len(TETS_24) == 24, len(TETS_24)
    assert len(TETS_34) == 34, len(TETS_34)
    assert (TETS_34[:24] == TETS_24).all()
    print(f'derivation: 4 diagonals x 6 tets = 24 raw -> {len(TETS_24)} distinct')
    print(f'            + 2 five-tet splits (10 more) -> {len(TETS_34)} distinct')

    # the 8 corner tets of the 5-tet splits ARE the trilinear corner Jacobians
    have = {frozenset(t) for t in map(tuple, TETS_34)}
    assert all(frozenset((v, v ^ 1, v ^ 2, v ^ 4)) in have for v in range(8))
    assert not any(
        frozenset((v, v ^ 1, v ^ 2, v ^ 4)) in {frozenset(t) for t in TETS_24} for v in range(8)
    )
    print('            8 corner tets (= trilinear corner Jacobians) absent from the 24  OK')

    rng = np.random.default_rng(0)
    D = H = W = 5
    c24 = AllTetConstraint3D((D, H, W))
    c34 = AllTetConstraint3D((D, H, W), include_5tet=True)
    c6 = SimplexConstraint3D((D, H, W))
    n_cells = (D - 1) * (H - 1) * (W - 1)

    # --- identity field ---------------------------------------------------
    ident = np.zeros((3, D, H, W))
    for c in (c24, c34):
        v = c.values(c.flatten(ident)).reshape(c.n_tets, -1)
        want = np.array(
            [
                s * _tet_volume_from_vertices(*(_ID_POS[i] for i in t))
                for t, s in zip(c.tets, c.signs)
            ]
        )
        assert np.allclose(v, want[:, None], atol=1e-14), (c, v.min(), v.max())
        assert (want > 0).all()
    # 24 Kuhn tets are all 1/6; the 5-tet split adds 8 more 1/6 corners + 2 * 1/3 centrals
    assert np.allclose(c24.values(c24.flatten(ident)), 1.0 / 6.0, atol=1e-14)
    print('identity field: every row positive; 24-row set all = +1/6  OK')

    # --- exact agreement with the 6-tet rows ------------------------------
    phi = rng.normal(scale=0.15, size=(3, D, H, W))
    f24, f6 = c24.flatten(phi), c6.flatten(phi)
    assert np.array_equal(f24, f6)
    v24, v6 = c24.values(f24), c6.values(f6)
    err = np.abs(v24[: 6 * n_cells] - v6).max()
    assert err == 0.0, err
    print(f'shared 6 tets: max |diff| = {err:g} (exact)  OK')

    # per-cell min matches the library's all-diagonal helper
    m_lib = six_tet_volumes_all_diagonals(phi).min(axis=0)
    m_new = v24.reshape(24, D - 1, H - 1, W - 1).min(axis=0)
    assert np.abs(m_lib - m_new).max() < 1e-15
    assert np.abs(six_tet_min_volume_3d(phi) - v6.reshape(6, *m_lib.shape).min(0)).max() < 1e-15
    print('per-cell min == six_tet_volumes_all_diagonals(...).min(0)  OK')

    # --- rows are CUBIC along a line (the 3D exact-line-search fact) ------
    # 2D triangle rows are bilinear -> quadratic along a line; a tet volume is a
    # 3x3 determinant of affine columns -> cubic. Fit at a in {0, 1/3, 2/3, 1},
    # predict a = 0.5, compare.
    d = rng.normal(scale=0.05, size=f24.size)
    va = [c24.values(f24 + a * d) for a in (0.0, 1 / 3, 2 / 3, 1.0)]
    coef = np.linalg.solve(
        np.array([[1, a, a**2, a**3] for a in (0.0, 1 / 3, 2 / 3, 1.0)]), np.array(va)
    )
    pred = coef[0] + 0.5 * coef[1] + 0.25 * coef[2] + 0.125 * coef[3]
    cubic_err = np.abs(pred - c24.values(f24 + 0.5 * d)).max()
    assert cubic_err < 1e-12, cubic_err
    print(f'rows exactly cubic along a line: max |resid| = {cubic_err:.2e}  OK')

    # --- finite-difference gradient check ---------------------------------
    for c in (c24, c34):
        v = rng.normal(size=c.n_constraints)
        g = c.adjoint(f24, v)
        h = 1e-6
        idx = rng.choice(c.n_variables, size=40, replace=False)
        fd = np.empty(idx.size)
        for j, i in enumerate(idx):
            fp, fm = f24.copy(), f24.copy()
            fp[i] += h
            fm[i] -= h
            fd[j] = (v @ c.values(fp) - v @ c.values(fm)) / (2 * h)
        rel = np.abs(fd - g[idx]).max() / max(np.abs(fd).max(), 1e-12)
        assert rel < 1e-6, (c, rel)
        print(f'{c!r}: FD gradient rel err = {rel:.3e}  OK')


if __name__ == '__main__':
    _selfcheck()
