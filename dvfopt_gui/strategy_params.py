"""Auto-generated per-strategy parameter editing.

Every dvfopt Strategy is a dataclass, so its knobs are introspectable:
``editable_fields`` maps dataclass fields to simple widget kinds and
``StrategyParamsTab`` renders them. Overrides are stored per-method by the
window and applied at worker construction — no bespoke UI per method.
"""

from __future__ import annotations

import dataclasses
import math

from PySide6 import QtWidgets

from dvfopt._defaults import DEFAULT_PARAMS as _DVFOPT_DEFAULTS

# Fields the toolbar already owns, or that make no sense to override here.
# ``supports_3d`` is a typed dataclass field on every Strategy (used for
# compatibility checks at Solver construction) but it is not a solver knob:
# unchecking it in the UI makes an otherwise-valid 3D run raise
# IncompatibleConstraintError, so it's excluded rather than rendered.
_EXCLUDED_FIELDS = {'time_budget_s', 'supports_3d'}
# Literal-choice fields (dataclasses can't express Literal defaults cleanly).
_CHOICE_FIELDS = {'accuracy': ('fast', 'max')}
# Spin SEEDS for ``float | None`` knobs when the user first ticks the
# override checkbox. Detection is by dataclass ANNOTATION (any
# float|None field with default None renders as optfloat); this table
# only picks the initial spin value. Derived from the canonical
# dvfopt default threshold (single source) — note the true None
# derivation uses the RUN threshold at solve time, so these are seeds,
# not the effective defaults.
_OPTFLOAT_SPIN_DEFAULTS = {
    'injectivity_threshold': float(_DVFOPT_DEFAULTS['threshold']),
    'band_threshold': 1.2 * float(_DVFOPT_DEFAULTS['threshold']),
    'recover_threshold': 1.2 * float(_DVFOPT_DEFAULTS['threshold']),
}
_OPTFLOAT_ANNOTATIONS = {'float|None', 'Optional[float]', 'None|float'}


def _is_optional_float(field) -> bool:
    ann = str(field.type).replace(' ', '')
    return ann in _OPTFLOAT_ANNOTATIONS


# Per-algo field whitelist. The 2D windowed path drives iterative_serial
# directly (toolbar owns its iteration knobs), so only the constraint-mode
# toggles are exposed; everything else would be editable-but-ignored.
_INCLUDED_FIELDS_BY_ALGO = {
    'slsqp_windowed': {'enforce_shoelace', 'enforce_injectivity', 'injectivity_threshold'},
}
# Per-algo greyed-out fields (config-time gating of run-time errors).
_DISABLED_FIELDS_BY_ALGO = {
    # enforce_shoelace is 2D-only; the 3D run would raise. Grey it out.
    'slsqp_windowed@jdet3d': {'enforce_shoelace'},
}


def _valid_override(kind: str, name: str, value) -> bool:
    """Reject persisted override values that would corrupt the widgets:
    wrong-typed ints, non-finite floats, unknown choice strings. The
    overrides come from a JSON round-trip in QSettings, so anything a
    past version (or a hand-edited settings file) wrote can show up.

    ``kind`` is one of the strings ``editable_fields`` emits: 'int' |
    'float' | 'bool' | 'choice' | 'str' | 'readonly'. Every kind gets an
    explicit branch rather than falling through to a permissive default,
    so a future kind added to ``editable_fields`` without a matching
    branch here fails loudly (as "always valid") instead of silently.
    """
    if kind == 'int':
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == 'float':
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
    if kind == 'bool':
        return isinstance(value, bool)
    if kind == 'choice':
        return isinstance(value, str) and value in _CHOICE_FIELDS.get(name, ())
    if kind == 'optfloat':
        if value is None:
            return True
        # Real numbers only: bools pass float() (True -> 1.0) and strings
        # like '0.01' float() fine but crash downstream — both rejected.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return math.isfinite(float(value))
    if kind == 'str':
        return isinstance(value, str)
    if kind == 'readonly':
        # Not editable and not applied to a widget from ``value`` (``build``
        # always renders ``repr(default)`` for readonly fields) — nothing to
        # corrupt, so any persisted value is harmless.
        return True
    return True


def strategy_class_for(algo: str):
    """Strategy class for a GUI method algo tag, or None for non-dataclass
    methods (auto / pipelines / marching).

    2D algo tags (e.g. ``'m14'``) and 3D algo tags are ambiguous on their
    own — the 3D classes have different knobs than their 2D namesakes — so
    the window family-qualifies 3D lookups as ``f'{algo}@tet3d'`` or
    ``f'{algo}@jdet3d'`` (see ``LiveSolverWindow._current_params_algo``).
    Plain algo tags resolve against the 2D mapping.
    """
    import dvfopt

    mapping = {
        'slp': dvfopt.SLPStrategy,
        # 2D windowed exposes ONLY the constraint-mode toggles (see
        # _INCLUDED_FIELDS_BY_ALGO) — the worker threads them into
        # iterative_serial; the toolbar owns the iteration knobs.
        'slsqp_windowed': dvfopt.SLSQPWindowedStrategy,
        'm14': dvfopt.HarmonicALMRefineRepairStrategy,
        'm14_schwarz': dvfopt.SchwarzHarmonicALMRefineRepairStrategy,
        'm10': dvfopt.HarmonicALMBarrierStrategy,
        'barrier': dvfopt.BarrierStrategy,
        'slsqp_fullgrid': dvfopt.SLSQPFullGridStrategy,
        'schwarz': dvfopt.SchwarzStrategy,
        'nmvf': dvfopt.NMVFStrategy,
        'barrier_torch': dvfopt.BarrierTet3DTorchStrategy,
    }
    tet3d = {
        'slp@tet3d': dvfopt.SLPStrategy,
        'm14@tet3d': dvfopt.HarmonicALMRefineRepair3DStrategy,
        'm14_schwarz@tet3d': dvfopt.SchwarzHarmonicALMRefineRepair3DStrategy,
        'm10@tet3d': dvfopt.HarmonicALMBarrier3DStrategy,
        'slsqp_fullgrid@tet3d': dvfopt.SLSQPFullGrid3DStrategy,
        'active_band@tet3d': dvfopt.ActiveBandALM3DStrategy,
        'coupled_kring@tet3d': dvfopt.CoupledKRing3DStrategy,
        # barrier_torch's GUI method id is already tet3d-only
        # (`barrier_torch_tet3d`), so `_current_params_algo` produces
        # 'barrier_torch@tet3d' in 3D mode — map it here too, alongside
        # the unqualified key above (used if a caller ever looks it up
        # without the family suffix).
        'barrier_torch@tet3d': dvfopt.BarrierTet3DTorchStrategy,
    }
    mapping.update(tet3d)
    jdet3d = {
        'barrier@jdet3d': dvfopt.BarrierStrategy,
        'slsqp_windowed@jdet3d': dvfopt.SLSQPWindowedStrategy,
    }
    mapping.update(jdet3d)
    return mapping.get(algo)


def editable_fields(cls) -> list:
    """``(name, kind, default)`` for each editable dataclass field.

    kind: 'int' | 'float' | 'bool' | 'choice' | 'str' | 'readonly'.
    """
    out = []
    for f in dataclasses.fields(cls):
        if f.name in _EXCLUDED_FIELDS or f.name.startswith('_'):
            continue
        default = (
            f.default
            if f.default is not dataclasses.MISSING
            else (f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
        )
        if default is None and _is_optional_float(f):
            out.append((f.name, 'optfloat', None))
        elif f.name in _CHOICE_FIELDS:
            choices = _CHOICE_FIELDS[f.name]
            assert str(default) in choices, (
                f'{cls.__name__}.{f.name}: default {default!r} not in {choices}. '
                f'_CHOICE_FIELDS is keyed by bare field name across ALL strategy '
                f'dataclasses — a colliding field with different choices needs '
                f'per-strategy keying.'
            )
            out.append((f.name, 'choice', default))
        elif isinstance(default, bool):
            out.append((f.name, 'bool', default))
        elif isinstance(default, int):
            out.append((f.name, 'int', default))
        elif isinstance(default, float):
            out.append((f.name, 'float', default))
        elif isinstance(default, str):
            out.append((f.name, 'str', default))
        else:  # tuples, None, other — visible but not editable
            out.append((f.name, 'readonly', default))
    return out


class StrategyParamsTab(QtWidgets.QWidget):
    """Form of widgets for one strategy class; returns only overrides."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QtWidgets.QFormLayout(self)
        self._widgets: dict = {}
        self._defaults: dict = {}

    def build(self, algo: str, overrides: dict) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()
        self._defaults.clear()
        cls = strategy_class_for(algo)
        if cls is None:
            self._form.addRow(QtWidgets.QLabel('<i>No editable parameters for this method.</i>'))
            return
        # Exact-algo keying ONLY: a base-algo fallback would bleed the 2D
        # windowed whitelist onto 'slsqp_windowed@jdet3d' and hide the 3D
        # tab's iteration knobs (their only input route).
        include = _INCLUDED_FIELDS_BY_ALGO.get(algo)
        disabled = _DISABLED_FIELDS_BY_ALGO.get(algo, set())
        for name, kind, default in editable_fields(cls):
            if include is not None and name not in include:
                continue
            value = overrides.get(name, default)
            if name in overrides and not _valid_override(kind, name, value):
                value = default
            if kind == 'int':
                w = QtWidgets.QSpinBox()
                w.setRange(-1_000_000_000, 1_000_000_000)
                w.setValue(int(value))
            elif kind == 'float':
                w = QtWidgets.QDoubleSpinBox()
                decimals = 6
                d_abs = abs(float(default)) if isinstance(default, (int, float)) else 0.0
                if 0.0 < d_abs < 1e-4:
                    # Tiny defaults (ftol=1e-10 etc.) need enough precision to
                    # survive the widget round-trip — decimals=6 clamps them
                    # to 0.0 and corrupts the solver settings.
                    decimals = min(14, math.ceil(-math.log10(d_abs)) + 2)
                w.setDecimals(decimals)
                w.setRange(-1e12, 1e12)
                w.setValue(float(value))
            elif kind == 'bool':
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(value))
            elif kind == 'choice':
                w = QtWidgets.QComboBox()
                for c in _CHOICE_FIELDS[name]:
                    w.addItem(c)
                w.setCurrentText(str(value))
            elif kind == 'optfloat':
                # Checkbox-enabled spinbox: unchecked = None (derive from
                # threshold), checked = explicit override value.
                w = QtWidgets.QWidget()
                lay = QtWidgets.QHBoxLayout(w)
                lay.setContentsMargins(0, 0, 0, 0)
                cb = QtWidgets.QCheckBox('override')
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(6)
                spin.setRange(-1e12, 1e12)
                spin.setValue(
                    float(value)
                    if value is not None
                    else float(_OPTFLOAT_SPIN_DEFAULTS.get(name, 0.01))
                )
                cb.setChecked(value is not None)
                spin.setEnabled(value is not None)
                cb.toggled.connect(spin.setEnabled)
                lay.addWidget(cb)
                lay.addWidget(spin)
                w._opt_check = cb  # type: ignore[attr-defined]
                w._opt_spin = spin  # type: ignore[attr-defined]
            elif kind == 'str':
                w = QtWidgets.QLineEdit(str(value))
            else:  # readonly
                w = QtWidgets.QLabel(repr(default))
                self._form.addRow(f'{name}:', w)
                continue
            if name in disabled:
                w.setEnabled(False)
                w.setToolTip('Not applicable for this constraint family.')
            self._widgets[name] = (kind, w)
            self._defaults[name] = default
            self._form.addRow(f'{name}:', w)

    def values(self) -> dict:
        """Only the values that differ from the dataclass defaults."""
        out = {}
        for name, (kind, w) in self._widgets.items():
            if not w.isEnabled():
                # Disabled = not applicable for this constraint family;
                # never (re-)emit an override for it.
                continue
            if kind == 'int':
                v = int(w.value())
            elif kind == 'float':
                v = float(w.value())
            elif kind == 'bool':
                v = bool(w.isChecked())
            elif kind == 'choice':
                v = str(w.currentText())
            elif kind == 'optfloat':
                v = float(w._opt_spin.value()) if w._opt_check.isChecked() else None
            else:
                v = str(w.text())
            if kind == 'float':
                default = self._defaults[name]
                if isinstance(default, (int, float)) and math.isclose(
                    v, float(default), rel_tol=1e-9, abs_tol=1e-300
                ):
                    continue
            elif v == self._defaults[name]:
                continue
            out[name] = v
        return out
