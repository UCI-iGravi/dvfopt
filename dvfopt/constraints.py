"""Constraint objects for DVFopt.

Each constraint encapsulates a single nonlinear inequality of the form
``C(phi) >= threshold`` and the machinery a solver needs:

* :meth:`Constraint.values`  — forward evaluation ``C(phi)``.
* :meth:`Constraint.adjoint` — ``J^T @ v``, used by L-BFGS-B / barrier
  / ALM. Avoids materialising the (potentially huge) dense Jacobian.
* :meth:`Constraint.jacobian` — sparse forward Jacobian ``J``. Optional;
  only the SLSQP path needs it.
* :meth:`Constraint.flatten` / :meth:`Constraint.unflatten` — convert
  between the canonical ``(C, H, W)`` array form and the flat decision
  vector each constraint expects (different families pack channels in
  different orders — see :class:`PhiPack`).

These are pure value objects — no state beyond shape information. A
single constraint instance can be reused across many solver calls.

The constraint family also fixes the **phi-pack convention**:

* All 2-triangle constraints pack as ``[dy, dx]`` (y first).
* All Jdet constraints pack as ``[dx, dy]`` or ``[dx, dy, dz]`` (x first).
  This matches the existing implementations exactly so the new
  constraint classes are drop-in.

See :mod:`dvfopt.strategies` for the algorithms that consume these.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import scipy.sparse as sp


class PhiPack:
    """Marker constants for the two flat-pack conventions in the package."""

    DY_FIRST = 'dy_first'  # 2-triangle family: [dy, dx]
    DX_FIRST = 'dx_first'  # Jdet family: [dx, dy] (2D) or [dx, dy, dz] (3D)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Constraint(ABC):
    """Smooth nonlinear constraint ``C(phi) >= threshold``.

    Subclasses fix the constraint family (2-tri, Jdet, tetrahedral, etc.)
    and the spatial dimensionality. The shape is bound at construction
    so the constraint object knows ``n_variables`` and ``n_constraints``
    without re-deriving them on every call.
    """

    # Convention identifiers — used by callers to assert pack-compat
    # when composing solvers.
    pack: str = ''
    dim: int = 2

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(int(s) for s in shape)

    # ----------------------------- shape -----------------------------
    @property
    @abstractmethod
    def n_variables(self) -> int:
        """Length of the flat decision vector."""

    @property
    @abstractmethod
    def n_constraints(self) -> int:
        """Length of the constraint vector ``C(phi)``."""

    # ----------------------------- flatten ---------------------------
    def coerce(self, phi) -> np.ndarray:
        """Accept loose input shapes and return the canonical ``(C, *shape)``
        ``float64`` ndarray that :meth:`flatten` expects.

        Delegates to :func:`dvfopt.validation.validate_dvf` for the
        cross-cutting concerns (array-coercion, NaN/Inf check, minimum
        spatial size, helpful error messages) and then enforces the
        family-specific shape contract.

        Subclasses may override to broaden the accepted layout family
        (e.g. ``TriConstraint2D`` accepts ``(3, 1, H, W)`` and drops the
        dz channel).
        """
        from dvfopt.validation import validate_dvf

        arr = validate_dvf(
            phi,
            dim=self.dim,
            # The base implementation is strict about layout; subclasses
            # broaden via override.
            accept_3channel_2d=False,
            accept_singleton_d_2d=False,
            name='deformation',
        )
        expected = (self.dim + 1, *self.shape)
        if arr.shape != expected:
            from dvfopt.exceptions import SolverConfigError

            raise SolverConfigError(
                f'deformation shape {arr.shape} does not match the '
                f'{type(self).__name__} shape {self.shape} '
                f'(expected canonical layout {expected})'
            )
        return arr

    @abstractmethod
    def flatten(self, phi: np.ndarray) -> np.ndarray:
        """``(C, *shape)`` ndarray -> flat decision vector."""

    @abstractmethod
    def unflatten(self, phi_flat: np.ndarray) -> np.ndarray:
        """Flat decision vector -> ``(C, *shape)`` ndarray."""

    # ----------------------------- math ------------------------------
    @abstractmethod
    def values(self, phi_flat: np.ndarray) -> np.ndarray:
        """Forward constraint evaluation ``C(phi)`` (length n_constraints)."""

    @abstractmethod
    def adjoint(self, phi_flat: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Adjoint ``J^T @ v`` (length n_variables, same shape as phi_flat)."""

    def jacobian(self, phi_flat: np.ndarray) -> Optional[sp.csr_matrix]:
        """Sparse forward Jacobian ``J`` (shape ``(n_constraints, n_variables)``).

        Returning ``None`` (the default) signals "no analytical Jacobian"
        — SLSQP will fall back to forward-differences, which is slow for
        anything past a tiny problem. Override on the constraint classes
        that have a known sparse pattern.
        """
        return None

    def _cached_jac_builder(self, factory):
        """Memoize a Jacobian *builder* on the instance.

        The builders precompute the sparsity pattern for this shape —
        rebuilding one on every ``jacobian()`` call wastes that work.
        ``factory`` is a zero-arg callable constructing the builder.
        """
        builder = getattr(self, '_jac_builder', None)
        if builder is None:
            builder = self._jac_builder = factory()
        return builder

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(shape={self.shape}, '
            f'n_vars={self.n_variables}, '
            f'n_constr={self.n_constraints})'
        )


# ---------------------------------------------------------------------------
# 2-triangle constraints (PL bijectivity, 2D only)
# ---------------------------------------------------------------------------


class TriConstraint2D(Constraint):
    """Standard per-cell 2-triangle constraint ``T1, T2 >= threshold``.

    Tiles each cell along the TR-BL diagonal into two triangles
    (``T2 = {TL, BL, TR}`` and ``T1 = {TR, BL, BR}``) and enforces both
    signed areas above threshold.

    .. warning::
        **Corner coverage gap.** Under the standard TR-BL split, the
        two diagonally-opposite grid corners ``(0, 0)`` and
        ``(H-1, W-1)`` sit in only one triangle each — they're
        under-constrained relative to interior vertices (4–6 triangles)
        and edge vertices (2–3). In practice fold escapes through these
        corners are rare but possible.

        For new code prefer the registry alias ``'2tri'``, which
        resolves to :class:`TriConstraint2DFullCoverage` (adds two
        opposite-diagonal corner patches at negligible cost). This
        class is retained as ``'2tri_standard'`` for reproducibility
        against benchmark numbers recorded before the default flipped.

    Phi pack: ``[dy.ravel(), dx.ravel()]`` (y-first).
    Output:   ``[T1.ravel(), T2.ravel()]`` of length ``2*(H-1)*(W-1)``.

    Reuses the canonical primitives in
    :mod:`dvfopt.core.primitives.tri` so this class is purely an
    interface wrapper — no math is re-derived.
    """

    pack = PhiPack.DY_FIRST
    dim = 2

    @property
    def n_variables(self) -> int:
        H, W = self.shape
        return 2 * H * W

    @property
    def n_constraints(self) -> int:
        H, W = self.shape
        return 2 * (H - 1) * (W - 1)

    def coerce(self, phi) -> np.ndarray:
        """Accept ``(2, H, W)`` / ``(3, H, W)`` / ``(2, 1, H, W)`` /
        ``(3, 1, H, W)``. Drops the dz channel and any singleton D axis.
        Returns the canonical ``(2, H, W)`` ``float64`` ndarray.
        """
        from dvfopt.exceptions import SolverConfigError
        from dvfopt.validation import validate_dvf

        arr = validate_dvf(
            phi,
            dim=2,
            accept_3channel_2d=True,
            accept_singleton_d_2d=True,
            name='deformation',
        )
        if arr.shape != (2, *self.shape):
            raise SolverConfigError(
                f'deformation spatial shape {arr.shape[1:]} does not match '
                f'this {type(self).__name__} (configured for {self.shape})'
            )
        return arr

    def flatten(self, phi) -> np.ndarray:
        phi = self.coerce(phi)
        return np.concatenate([phi[0].ravel(), phi[1].ravel()])

    def unflatten(self, phi_flat: np.ndarray) -> np.ndarray:
        H, W = self.shape
        return np.stack([phi_flat[: H * W].reshape(H, W), phi_flat[H * W :].reshape(H, W)])

    def values(self, phi_flat: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.tri import tri_areas_flat

        return tri_areas_flat(phi_flat, *self.shape)

    def adjoint(self, phi_flat: np.ndarray, v: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.tri import tri_grad_T_v

        return tri_grad_T_v(phi_flat, *self.shape, v)

    def jacobian(self, phi_flat: np.ndarray) -> sp.csr_matrix:
        # Cached builder (precomputed sparsity pattern). It returns a
        # reused dense buffer (fastest for scipy's SLSQP adapter, which
        # densifies anyway); this convenience API keeps its documented
        # sparse contract.
        from dvfopt.core.primitives.tri import build_full_grid_tri_jac

        builder = self._cached_jac_builder(lambda: build_full_grid_tri_jac(*self.shape, False))
        return sp.csr_matrix(builder(phi_flat))


class TriConstraint2DFullCoverage(Constraint):
    """``TriConstraint2D`` + two opposite-diagonal corner patches.

    This is the **default** for ``'2tri'`` in the registry. Adds two
    extra triangle constraints at cells ``(0, 0)``
    and ``(H-2, W-2)`` using the opposite (TL-BR) diagonal — one
    triangle each — so the grid vertices ``(0, 0)`` and ``(H-1, W-1)``,
    which the standard scheme leaves in only one triangle, end up in
    two. Other cells are unchanged.

    Output: ``[T1, T2, patch_TL, patch_BR]`` of length
    ``2*(H-1)*(W-1) + 2``. The two patches are *additive* — not a
    re-tiling — appended to the standard ``T1``/``T2`` stack.

    The cost over :class:`TriConstraint2D` is 2 scalar evaluations + 6
    extra gradient terms total per call — measurable in microseconds
    on full slices, negligible relative to the rest of a barrier
    iteration. The defensive coverage is essentially free, so this is
    the preferred choice for new code.
    """

    pack = PhiPack.DY_FIRST
    dim = 2

    @property
    def n_variables(self) -> int:
        H, W = self.shape
        return 2 * H * W

    @property
    def n_constraints(self) -> int:
        H, W = self.shape
        return 2 * (H - 1) * (W - 1) + 2

    def coerce(self, phi: np.ndarray) -> np.ndarray:
        return TriConstraint2D.coerce(self, phi)

    def flatten(self, phi: np.ndarray) -> np.ndarray:
        return TriConstraint2D.flatten(self, phi)

    def unflatten(self, phi_flat: np.ndarray) -> np.ndarray:
        return TriConstraint2D.unflatten(self, phi_flat)

    def values(self, phi_flat: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.tri import tri_areas_flat_full_coverage

        return tri_areas_flat_full_coverage(phi_flat, *self.shape)

    def adjoint(self, phi_flat: np.ndarray, v: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.tri import tri_grad_T_v_full_coverage

        return tri_grad_T_v_full_coverage(phi_flat, *self.shape, v)

    def jacobian(self, phi_flat: np.ndarray) -> sp.csr_matrix:
        # See TriConstraint2D.jacobian — cached builder, dense buffer
        # wrapped to keep the documented sparse contract.
        from dvfopt.core.primitives.tri import build_full_grid_tri_jac

        builder = self._cached_jac_builder(lambda: build_full_grid_tri_jac(*self.shape, True))
        return sp.csr_matrix(builder(phi_flat))


# ---------------------------------------------------------------------------
# Jacobian determinant constraints
# ---------------------------------------------------------------------------


class JdetConstraint2D(Constraint):
    """Per-pixel Jacobian determinant ``det(I + ∇phi) >= threshold``.

    Phi pack: ``[dx.ravel(), dy.ravel()]`` (x-first, matching the
    historical SLSQP/barrier convention).
    Output: ``J.ravel()`` of length ``H*W``.

    Reuses :func:`dvfopt.jacobian.numpy_jdet._numpy_jdet_2d` and the
    closed-form adjoint already in
    :mod:`dvfopt.core.primitives.jdet2d`.
    """

    pack = PhiPack.DX_FIRST
    dim = 2

    @property
    def n_variables(self) -> int:
        H, W = self.shape
        return 2 * H * W

    @property
    def n_constraints(self) -> int:
        H, W = self.shape
        return H * W

    def coerce(self, phi) -> np.ndarray:
        """Same shape acceptance as :class:`TriConstraint2D`. Reuses
        the canonical 2D validator."""
        return TriConstraint2D.coerce(self, phi)

    def flatten(self, phi: np.ndarray) -> np.ndarray:
        phi = self.coerce(phi)
        return np.concatenate([phi[1].ravel(), phi[0].ravel()])

    def unflatten(self, phi_flat: np.ndarray) -> np.ndarray:
        H, W = self.shape
        n = H * W
        return np.stack(
            [
                phi_flat[n:].reshape(H, W),  # dy back to channel 0
                phi_flat[:n].reshape(H, W),
            ]
        )  # dx back to channel 1

    def values(self, phi_flat: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.jdet2d import jdet_2d_flat

        return jdet_2d_flat(phi_flat, self.shape)

    def adjoint(self, phi_flat: np.ndarray, v: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.jdet2d import jdet_grad_T_v_2d

        return jdet_grad_T_v_2d(phi_flat, self.shape, v)


class FiniteJdetConstraint2D(Constraint):
    """Per-cell forward-difference determinant ``(1+a)(1+d) - b*c >= threshold``.

    With forward differences at cell ``(i, j)``::

        a = dx[i,j+1]-dx[i,j]   b = dx[i+1,j]-dx[i,j]
        c = dy[i,j+1]-dy[i,j]   d = dy[i+1,j]-dy[i,j]

    the constraint is the deformed-parallelogram area spanned by the two
    forward edges — a local 2-pixel stencil, so unlike the central-diff
    :class:`JdetConstraint2D` it is NOT blind to high-frequency
    (checkerboard) folds. A middle strictness between the central-diff
    Jdet and the exact 2-triangle areas of :class:`TriConstraint2D`.
    Promoted from ``benchmarks/finite_jdet.py`` (PR #64).

    Phi pack: ``[dx.ravel(), dy.ravel()]`` (x-first, matching
    :class:`JdetConstraint2D`).
    Output: cell determinants of length ``(H-1)*(W-1)``.

    Reuses :mod:`dvfopt.core.primitives.finite_jdet` for the flat form,
    analytic sparse Jacobian, and adjoint.
    """

    pack = PhiPack.DX_FIRST
    dim = 2

    @property
    def n_variables(self) -> int:
        H, W = self.shape
        return 2 * H * W

    @property
    def n_constraints(self) -> int:
        H, W = self.shape
        return (H - 1) * (W - 1)

    def coerce(self, phi) -> np.ndarray:
        """Same shape acceptance as :class:`TriConstraint2D`. Reuses
        the canonical 2D validator."""
        return TriConstraint2D.coerce(self, phi)

    def flatten(self, phi: np.ndarray) -> np.ndarray:
        phi = self.coerce(phi)
        return np.concatenate([phi[1].ravel(), phi[0].ravel()])

    def unflatten(self, phi_flat: np.ndarray) -> np.ndarray:
        H, W = self.shape
        n = H * W
        return np.stack(
            [
                phi_flat[n:].reshape(H, W),  # dy back to channel 0
                phi_flat[:n].reshape(H, W),
            ]
        )  # dx back to channel 1

    def values(self, phi_flat: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.finite_jdet import finite_jdet_flat

        return finite_jdet_flat(phi_flat, *self.shape)

    def adjoint(self, phi_flat: np.ndarray, v: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.finite_jdet import finite_jdet_grad_T_v

        return finite_jdet_grad_T_v(phi_flat, *self.shape, v)

    def jacobian(self, phi_flat: np.ndarray) -> sp.csr_matrix:
        # Analytic sparse pattern — 6 nonzeros per cell (3 dx, 3 dy);
        # already CSR, so no wrapping needed.
        from dvfopt.core.primitives.finite_jdet import finite_jdet_jacobian

        return finite_jdet_jacobian(phi_flat, *self.shape)


class JdetConstraint3D(Constraint):
    """Per-voxel 3D Jacobian determinant ``det(I + ∇phi) >= threshold``.

    Phi pack: ``[dx, dy, dz]`` (x-first, length ``3*D*H*W``).
    Output: ``J.ravel()`` of length ``D*H*W``.
    """

    pack = PhiPack.DX_FIRST
    dim = 3

    @property
    def n_variables(self) -> int:
        D, H, W = self.shape
        return 3 * D * H * W

    @property
    def n_constraints(self) -> int:
        D, H, W = self.shape
        return D * H * W

    def coerce(self, phi) -> np.ndarray:
        """Accept the canonical ``(3, D, H, W)`` layout. Validates +
        casts to ``float64`` via :func:`dvfopt.validation.validate_dvf`."""
        from dvfopt.exceptions import SolverConfigError
        from dvfopt.validation import validate_dvf

        arr = validate_dvf(phi, dim=3, name='deformation')
        if arr.shape != (3, *self.shape):
            raise SolverConfigError(
                f'deformation spatial shape {arr.shape[1:]} does not match '
                f'this JdetConstraint3D (configured for {self.shape})'
            )
        return arr

    def flatten(self, phi: np.ndarray) -> np.ndarray:
        phi = self.coerce(phi)
        # Pack as [dx, dy, dz] (x-first) to match the existing
        # barrier_objective convention.
        return np.concatenate([phi[2].ravel(), phi[1].ravel(), phi[0].ravel()])

    def unflatten(self, phi_flat: np.ndarray) -> np.ndarray:
        D, H, W = self.shape
        n = D * H * W
        dx = phi_flat[:n].reshape(D, H, W)
        dy = phi_flat[n : 2 * n].reshape(D, H, W)
        dz = phi_flat[2 * n :].reshape(D, H, W)
        return np.stack([dz, dy, dx])

    def values(self, phi_flat: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.jdet3d import jdet_full

        return jdet_full(phi_flat, self.shape)

    def adjoint(self, phi_flat: np.ndarray, v: np.ndarray) -> np.ndarray:
        from dvfopt.core.primitives.jdet3d import _jdet_grad_T_v

        return _jdet_grad_T_v(phi_flat, self.shape, v)


class Tet6Constraint3D(Constraint):
    """Per-voxel 6-tetrahedron signed volume ``V_k(phi) >= threshold``.

    3D analogue of :class:`TriConstraint2D`. Each cubic voxel cell is
    decomposed into six tetrahedra sharing the main diagonal ``C0``→``C7``
    (see :func:`dvfopt.jacobian.tetrahedron_sign.six_tet_volumes_3d`).
    The constraint enforces every per-tet signed volume above
    ``threshold``; identity field yields exactly ``+1/6`` per tet.

    Compared to :class:`JdetConstraint3D` this gives a more local check
    (6 per voxel instead of 1) and is a smooth function of phi at every
    fold boundary, which is friendlier for the barrier path.

    Phi pack: ``[dx, dy, dz]`` (DX_FIRST), length ``3*D*H*W`` — matches
    :class:`JdetConstraint3D` so 3D barrier plumbing can be shared.
    Output: ``[V0, V1, V2, V3, V4, V5].ravel()`` of length
    ``6 * (D-1) * (H-1) * (W-1)``.
    """

    pack = PhiPack.DX_FIRST
    dim = 3

    @property
    def n_variables(self) -> int:
        D, H, W = self.shape
        return 3 * D * H * W

    @property
    def n_constraints(self) -> int:
        D, H, W = self.shape
        return 6 * (D - 1) * (H - 1) * (W - 1)

    def coerce(self, phi) -> np.ndarray:
        """Validate the input is canonical ``(3, D, H, W)``."""
        from dvfopt.exceptions import SolverConfigError
        from dvfopt.validation import validate_dvf

        arr = validate_dvf(phi, dim=3, name='deformation')
        if arr.shape != (3, *self.shape):
            raise SolverConfigError(
                f'deformation spatial shape {arr.shape[1:]} does not match '
                f'this Tet6Constraint3D (configured for {self.shape})'
            )
        return arr

    def flatten(self, phi: np.ndarray) -> np.ndarray:
        phi = self.coerce(phi)
        # Pack as [dx, dy, dz] (DX_FIRST), same as JdetConstraint3D.
        return np.concatenate([phi[2].ravel(), phi[1].ravel(), phi[0].ravel()])

    def unflatten(self, phi_flat: np.ndarray) -> np.ndarray:
        D, H, W = self.shape
        n = D * H * W
        dx = phi_flat[:n].reshape(D, H, W)
        dy = phi_flat[n : 2 * n].reshape(D, H, W)
        dz = phi_flat[2 * n :].reshape(D, H, W)
        return np.stack([dz, dy, dx])

    def values(self, phi_flat: np.ndarray) -> np.ndarray:
        from dvfopt.jacobian.tetrahedron_sign import tet_volumes_flat

        return tet_volumes_flat(phi_flat, *self.shape)

    def adjoint(self, phi_flat: np.ndarray, v: np.ndarray) -> np.ndarray:
        from dvfopt.jacobian.tetrahedron_sign import tet_grad_T_v

        return tet_grad_T_v(phi_flat, *self.shape, v)

    def jacobian(self, phi_flat: np.ndarray) -> sp.csr_matrix:
        """Sparse forward Jacobian for the SLSQP path.

        Note: the resulting matrix has ``6 * (D-1)(H-1)(W-1)`` rows and
        ``3 * D * H * W`` columns — for a 32^3 voxel grid that's
        ~1.8e5 × ~9.8e4 with ~2.2M non-zeros. SLSQP at this scale is
        impractical (the active-set QP step dominates). Use barrier
        for any realistic 3D problem; this exists for symmetry with
        ``TriConstraint2D`` and for tiny-grid debugging.
        """
        from dvfopt.jacobian.tetrahedron_sign import build_tet_sparse_jac

        # Cached builder — the (rows, cols) pattern is precomputed once
        # per instance instead of on every call.
        builder = self._cached_jac_builder(lambda: build_tet_sparse_jac(*self.shape))
        return builder(phi_flat)


# ---------------------------------------------------------------------------
# Registry — string label -> Constraint class
# ---------------------------------------------------------------------------

_CONSTRAINT_REGISTRY: dict[str, type] = {}


def register_constraint(label: str):
    """Decorator that registers a Constraint subclass under ``label``.

    External packages adding new constraint families plug in by
    decorating their class::

        @register_constraint('6tet')
        class Tet6Constraint3D(Constraint): ...

    They become available via :func:`make_constraint('6tet', ...)`.

    Re-registering the *same* class object under an existing label is a
    silent no-op (idempotent, e.g. module re-import). Registering a
    *different* class under an already-taken label raises
    :class:`ValueError` instead of silently replacing the original.
    """

    def deco(cls: type) -> type:
        if not issubclass(cls, Constraint):
            raise TypeError(f'{cls.__name__} is not a Constraint subclass')
        existing = _CONSTRAINT_REGISTRY.get(label)
        if existing is not None and existing is not cls:
            raise ValueError(
                f'constraint label {label!r} is already registered to '
                f'{existing.__module__}.{existing.__qualname__}; refusing to '
                f'silently overwrite it with {cls.__module__}.{cls.__qualname__}. '
                f'Pick a different label.'
            )
        _CONSTRAINT_REGISTRY[label] = cls
        return cls

    return deco


# Register built-ins.
#
# Note on the '2tri' default
# --------------------------
# ``'2tri'`` resolves to :class:`TriConstraint2DFullCoverage` — the
# variant that adds two opposite-diagonal corner patches so every grid
# vertex is in ≥ 2 triangles. The "standard" TR-BL-only scheme is
# still available as ``'2tri_standard'`` for benchmark reproducibility
# (existing recorded runs used that variant). The corner gap closure
# costs 2 extra scalar constraints (~6 extra gradient terms) and is
# essentially free.
register_constraint('2tri')(TriConstraint2DFullCoverage)
register_constraint('2tri_standard')(TriConstraint2D)
register_constraint('jdet')(JdetConstraint2D)  # alias
register_constraint('jdet_2d')(JdetConstraint2D)
register_constraint('finite')(FiniteJdetConstraint2D)
register_constraint('jdet_3d')(JdetConstraint3D)
register_constraint('6tet')(Tet6Constraint3D)
register_constraint('6tet_3d')(Tet6Constraint3D)  # explicit alias


def make_constraint(name: str, shape: tuple[int, ...]) -> Constraint:
    """Construct a Constraint by name.

    Examples
    --------
    >>> c = make_constraint('2tri', (10, 10))
    >>> c.values(c.flatten(phi))
    """
    try:
        cls = _CONSTRAINT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f'unknown constraint: {name!r}; valid: {sorted(_CONSTRAINT_REGISTRY)}'
        ) from exc
    return cls(shape)


__all__ = [
    'Constraint',
    'FiniteJdetConstraint2D',
    'JdetConstraint2D',
    'JdetConstraint3D',
    'PhiPack',
    'TriConstraint2D',
    'TriConstraint2DFullCoverage',
    'make_constraint',
    'register_constraint',
]
