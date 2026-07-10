"""Auto-generated per-strategy parameter editing.

Every dvfopt Strategy is a dataclass, so its knobs are introspectable:
``editable_fields`` maps dataclass fields to simple widget kinds and
``StrategyParamsTab`` renders them. Overrides are stored per-method by the
window and applied at worker construction — no bespoke UI per method.
"""

from __future__ import annotations

import dataclasses
import math

from PyQt5 import QtWidgets

# Fields the toolbar already owns, or that make no sense to override here.
# ``supports_3d`` is a typed dataclass field on every Strategy (used for
# compatibility checks at Solver construction) but it is not a solver knob:
# unchecking it in the UI makes an otherwise-valid 3D run raise
# IncompatibleConstraintError, so it's excluded rather than rendered.
_EXCLUDED_FIELDS = {'time_budget_s', 'supports_3d'}
# Literal-choice fields (dataclasses can't express Literal defaults cleanly).
_CHOICE_FIELDS = {'accuracy': ('fast', 'max')}


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
        'm14': dvfopt.HarmonicALMRefineRepairStrategy,
        'm14_schwarz': dvfopt.SchwarzHarmonicALMRefineRepairStrategy,
        'm10': dvfopt.HarmonicALMBarrierStrategy,
        'barrier': dvfopt.BarrierStrategy,
        # 2D windowed-SLSQP never constructs a Strategy (run() routes it to
        # _run_windowed_slsqp, driven by the toolbar's max_iter spinbox), so
        # a params tab here would be editable-but-ignored. The 3D variant
        # ('slsqp_windowed@jdet3d') IS Strategy-constructed and editable.
        'slsqp_windowed': None,
        'slsqp_fullgrid': dvfopt.SLSQPFullGridStrategy,
        'schwarz': dvfopt.SchwarzStrategy,
        'nmvf': dvfopt.NMVFStrategy,
        'barrier_torch': dvfopt.BarrierTet3DTorchStrategy,
    }
    tet3d = {
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
        if f.name in _CHOICE_FIELDS:
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
        for name, kind, default in editable_fields(cls):
            value = overrides.get(name, default)
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
            elif kind == 'str':
                w = QtWidgets.QLineEdit(str(value))
            else:  # readonly
                w = QtWidgets.QLabel(repr(default))
                self._form.addRow(f'{name}:', w)
                continue
            self._widgets[name] = (kind, w)
            self._defaults[name] = default
            self._form.addRow(f'{name}:', w)

    def values(self) -> dict:
        """Only the values that differ from the dataclass defaults."""
        out = {}
        for name, (kind, w) in self._widgets.items():
            if kind == 'int':
                v = int(w.value())
            elif kind == 'float':
                v = float(w.value())
            elif kind == 'bool':
                v = bool(w.isChecked())
            elif kind == 'choice':
                v = str(w.currentText())
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
