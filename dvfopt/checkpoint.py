"""Resumable-run checkpoint shared by the pipelines (2.5D, per-slice 2D, 3D, CLI).

``<dir>/field.npy`` is a memmap mirror of the ``(C, *shape)`` output, written
one *unit* at a time (a z-slice, a stage, the whole array); ``<dir>/state.json``
holds the validated ``meta`` (always ``shape`` + ``input_sha256`` of the input,
plus the caller's knobs), the ``done`` unit ids in completion order, optional
per-unit ``rows`` (JSON-serialisable dicts the caller needs to rebuild its
report), and ``stage`` (``'run'`` | ``'done'``). Pure numpy + stdlib.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


class RunCheckpoint:
    """``slab(unit)`` maps a unit id to an index into the field; the default
    is a z-slice's ``[dy, dx]`` planes, ``field[1:3, z]``."""

    def __init__(self, checkpoint_dir, phi_in, meta, *, slab=None):
        self.dir = Path(checkpoint_dir)
        self.meta = dict(
            meta,
            shape=list(phi_in.shape),
            input_sha256=hashlib.sha256(np.ascontiguousarray(phi_in)).hexdigest(),
        )
        self._slab = slab or (lambda z: (slice(1, 3), z))
        self.field: np.ndarray  # the memmap mirror, bound by open()
        self.state: dict = {}

    def open(self):
        """Create the checkpoint, or validate and load an existing one (a
        mismatch on any ``meta`` key raises ``ValueError`` naming the keys)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        sp, fp = self.dir / 'state.json', self.dir / 'field.npy'
        if sp.exists() and fp.exists():
            state = json.loads(sp.read_text(encoding='utf-8'))
            bad = {k: (state.get(k), v) for k, v in self.meta.items() if state.get(k) != v}
            if bad:
                raise ValueError(
                    f'checkpoint {self.dir} does not match this run (stored, this): {bad}'
                )
            self.state = state
            self.field = np.lib.format.open_memmap(fp, mode='r+')
            return self
        self.field = np.lib.format.open_memmap(
            fp, mode='w+', dtype=np.float64, shape=tuple(self.meta['shape'])
        )
        self.state = dict(self.meta, done=[], rows={}, stage='run')
        self._save()
        return self

    @property
    def finished(self) -> bool:
        return self.state.get('stage') == 'done'

    @property
    def done(self) -> list:
        return self.state['done']

    @property
    def rows(self) -> dict:
        """Per-unit rows, keyed by ``str(unit)`` (JSON object keys)."""
        return self.state['rows']

    def is_done(self, unit) -> bool:
        return unit in self.state['done']

    def restore_into(self, out):
        """Copy every done unit's slab from the mirror into ``out``."""
        for u in self.state['done']:
            idx = self._slab(u)
            out[idx] = self.field[idx]

    def mark(self, unit, slab=None, row=None):
        """Mirror ``slab`` (if given) under ``unit``, record ``row``, and
        append ``unit`` to ``done`` — atomically, so an interruption leaves
        either the previous state or this one."""
        if slab is not None:
            self.field[self._slab(unit)] = slab
            self.field.flush()
        if row is not None:
            self.state['rows'][str(unit)] = row
        self.state['done'].append(unit)
        self._save()

    def finish(self, out=None):
        """Mirror the whole ``out`` (if given) and mark the run ``done``."""
        if out is not None:
            self.field[...] = out
            self.field.flush()
        self.state['stage'] = 'done'
        self._save()

    def _save(self):
        sp = self.dir / 'state.json'
        tmp = sp.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.state, default=_json_scalar), encoding='utf-8')
        os.replace(tmp, sp)


def _json_scalar(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f'not JSON-serialisable: {type(o).__name__}')


__all__ = ['RunCheckpoint']
