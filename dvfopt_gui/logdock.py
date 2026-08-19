"""Live solver-log dock.

Since the logging migration, every solver progress line and warning
routes through the ``dvfopt`` logger — this dock attaches a Qt-aware
handler so long runs show their per-phase output inside the GUI instead
of vanishing (the worker historically ran everything ``verbose=0`` and
prints went to a console nobody watches).

Thread-safety: solver logging happens on the worker QThread; the handler
re-emits each formatted record through a Qt signal, which Qt delivers to
the GUI thread via a queued connection — the text widget is only ever
touched from the GUI thread.

Level semantics: the dock's level combo controls BOTH the handler filter
and the ``verbose`` value the window passes to the next worker (the
solver call sites guard their log lines with ``if verbose:``, so a
DEBUG handler alone would still see nothing at ``verbose=0``).
WARNING-level messages (`log_warning` — cluster failures, swallowed
callbacks) are emitted unconditionally by the solvers and therefore
always appear, regardless of verbose.
"""

from __future__ import annotations

import logging

from PyQt5 import QtCore, QtGui, QtWidgets

from dvfopt._logging import logger as _dvfopt_logger

_MAX_LINES = 5000

# (combo label, handler level, worker verbose)
_LEVELS = [
    ('Warnings', logging.WARNING, 0),
    ('Info', logging.INFO, 1),
    ('Debug', logging.DEBUG, 2),
]


class _LogEmitter(QtCore.QObject):
    """Qt signal relay owned by the dock (dies with it)."""

    message = QtCore.pyqtSignal(str, int)  # text, levelno


class _QtLogHandler(logging.Handler):
    """logging.Handler that re-emits formatted records via a Qt signal.

    Deliberately a PLAIN Handler (not a QObject): logging.shutdown()
    touches every registered handler at interpreter exit, and a
    QObject-derived handler whose C++ side Qt already deleted raises
    RuntimeError there. The QObject half lives in :class:`_LogEmitter`;
    if it dies first, ``emit`` swallows the RuntimeError.
    """

    def __init__(self, emitter: _LogEmitter):
        logging.Handler.__init__(self)
        self._emitter = emitter
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):  # logging.Handler API
        try:
            self._emitter.message.emit(self.format(record), record.levelno)
        except Exception:  # never let logging crash the solver thread
            pass


class LogDock(QtWidgets.QDockWidget):
    """Collapsible dock: level selector + capped plain-text log view."""

    #: emitted when the user changes the level; carries the worker
    #: ``verbose`` value the window should use for the NEXT run.
    verboseChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__('Solver log', parent)
        self.setObjectName('solver_log_dock')

        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(4, 4, 4, 4)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Level:'))
        self._level_combo = QtWidgets.QComboBox()
        for label, _lvl, _verb in _LEVELS:
            self._level_combo.addItem(label)
        row.addWidget(self._level_combo)
        clear_btn = QtWidgets.QPushButton('Clear')
        row.addWidget(clear_btn)
        row.addStretch(1)
        lay.addLayout(row)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(_MAX_LINES)
        self._text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        lay.addWidget(self._text)
        self.setWidget(body)

        self._emitter = _LogEmitter(self)
        self._handler = _QtLogHandler(self._emitter)
        self._emitter.message.connect(self._append)  # queued across threads
        # Belt-and-braces: if Qt destroys the dock without closeEvent
        # (tests, embedding), still unhook the handler from the logger.
        self.destroyed.connect(lambda *_: self.detach())
        clear_btn.clicked.connect(self._text.clear)
        self._level_combo.currentIndexChanged.connect(self._on_level_changed)
        self._attached = False
        self._on_level_changed(0)

    # ----- logger attachment -------------------------------------------------
    def attach(self) -> None:
        """Attach the handler to the ``dvfopt`` logger.

        The logger level is opened to DEBUG (filtering happens at this
        handler); a real handler on the logger also suppresses
        ``_logging``'s stdout auto-install, so the dock becomes the
        single sink for solver output in the GUI process.
        """
        if self._attached:
            return
        _dvfopt_logger.addHandler(self._handler)
        _dvfopt_logger.setLevel(logging.DEBUG)
        self._attached = True

    def detach(self) -> None:
        if self._attached:
            _dvfopt_logger.removeHandler(self._handler)
            self._handler.close()  # deregister from logging's shutdown list
            self._attached = False

    # ----- level -------------------------------------------------------------
    @property
    def worker_verbose(self) -> int:
        """``verbose`` value for the next solver run at the current level."""
        return _LEVELS[self._level_combo.currentIndex()][2]

    def _on_level_changed(self, idx: int) -> None:
        _label, level, verbose = _LEVELS[idx]
        self._handler.setLevel(level)
        self.verboseChanged.emit(verbose)

    # ----- sink --------------------------------------------------------------
    def _append(self, text: str, levelno: int) -> None:
        if levelno >= logging.WARNING:
            import html

            self._text.appendHtml(f'<span style="color:#c0392b;">{html.escape(text)}</span>')
        else:
            self._text.appendPlainText(text)


__all__ = ['LogDock']
