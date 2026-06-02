@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ----------------------------------------------------------------
REM  dvfopt_gui — live PyQtGraph visualisation of the SLSQP solver
REM
REM  Usage:
REM      run_gui.bat                       a default 20x20 synthetic case
REM      run_gui.bat --canonical 03d       a specific canonical synthetic
REM      run_gui.bat --b0039 12            the B0039 z=12 dense extreme
REM      run_gui.bat --b0039 100 --max-iter 30
REM
REM  Any args you pass are forwarded verbatim to ``dvfopt_gui.demo``.
REM ----------------------------------------------------------------

REM Prefer the project venv when present; fall back to "python" on PATH.
set PYTHON=python
if exist .venv\Scripts\python.exe set PYTHON=.venv\Scripts\python.exe

REM Detect that the GUI extras are installed; tell the user how to fix
REM if not. We probe with a tiny stub instead of importing pyqtgraph
REM directly (its import is slow).
%PYTHON% -c "import importlib.util as u; raise SystemExit(0 if u.find_spec('pyqtgraph') and u.find_spec('PyQt5') else 1)"
if errorlevel 1 (
    echo.
    echo [run_gui] PyQt5 and pyqtgraph are not installed in this Python.
    echo           Install them with:
    echo               %PYTHON% -m pip install -e ".[gui]"
    echo.
    exit /b 2
)

echo [run_gui] launching dvfopt_gui.demo %*
%PYTHON% -m dvfopt_gui.demo %*

endlocal
