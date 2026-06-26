@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM   dvfopt — virtual environment setup / reinstall
REM
REM   Usage:
REM       setup_venv.bat                fresh install (prompts if .venv exists)
REM       setup_venv.bat --reinstall    delete + recreate .venv without prompting
REM       setup_venv.bat uninstall      delete .venv and stop
REM       setup_venv.bat --gpu          install CUDA 12.8 PyTorch first
REM
REM   Pip-installs the editable package + all optional extras
REM   ([dev,benchmarks,gui]) so a single command sets up everything
REM   needed for tests, benchmarks, and the live-viz GUI.
REM ============================================================

echo.
echo ============================================================
echo   dvfopt - virtual environment setup
echo ============================================================
echo.

if /i "%1"=="uninstall" goto :uninstall
if /i "%1"=="--uninstall" goto :uninstall

set FORCE=0
set USE_GPU=0
:argloop
if "%1"=="" goto :argloop_done
if /i "%1"=="--reinstall" set FORCE=1
if /i "%1"=="--gpu" set USE_GPU=1
shift
goto :argloop
:argloop_done

REM ---- Create venv ----
if exist .venv (
    if "%FORCE%"=="1" (
        echo Removing existing .venv ^(--reinstall^) ...
        rmdir /s /q .venv
    ) else (
        echo A .venv already exists.
        set /p OVERWRITE="Delete it and start fresh? [y/N]: "
        if /i not "!OVERWRITE!"=="y" (
            echo Keeping existing .venv. Re-running pip install on top of it.
            goto :install_packages
        )
        rmdir /s /q .venv
    )
)

echo Creating .venv with the python on PATH ...
python -m venv .venv
if !ERRORLEVEL! neq 0 (
    echo ERROR: ``python -m venv .venv`` failed. Make sure Python 3.10+
    echo        is on your PATH ^(``python --version`` from this shell^).
    goto :done
)

:install_packages
echo Upgrading pip ...
.venv\Scripts\python -m pip install --upgrade pip wheel setuptools

if "%USE_GPU%"=="1" (
    echo.
    echo Installing CUDA 12.8 PyTorch first ^(per --gpu^) ...
    .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if !ERRORLEVEL! neq 0 (
        echo WARNING: GPU torch install failed - the editable install below
        echo          will fall back to CPU-only torch via the [benchmarks] extra.
    )
)

echo.
echo Installing dvfopt + all extras ^([dev,benchmarks,gui]^) in editable mode ...
.venv\Scripts\python -m pip install -e ".[dev,benchmarks,gui]"
if !ERRORLEVEL! neq 0 (
    echo ERROR: pip install failed. See the messages above.
    goto :done
)

echo.
echo ============================================================
echo   Done.
echo
echo   Quick checks:
echo     .venv\Scripts\python -m pytest tests/ -q
echo     .venv\Scripts\python -m dvfopt_gui.demo --canonical 03d
echo
echo   VS Code:
echo     Command Palette ^(Ctrl+Shift+P^) ^> "Python: Select Interpreter"
echo     pick  .\.venv\Scripts\python.exe
echo ============================================================
goto :done

:uninstall
if not exist .venv (
    echo No .venv found - nothing to remove.
    goto :done
)
echo This will permanently delete the .venv directory.
set /p CONFIRM="Are you sure? [y/N]: "
if /i not "!CONFIRM!"=="y" (
    echo Aborted.
    goto :done
)
echo Removing .venv ...
rmdir /s /q .venv
echo Done.

:done
echo.
endlocal
