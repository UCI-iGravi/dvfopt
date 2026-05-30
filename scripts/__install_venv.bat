@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Deformation Field Processing - Virtual Environment Setup
echo ============================================================
echo.

if "%1"=="uninstall" goto :uninstall

:: ---- Install ----
if exist .venv (
    echo A .venv already exists in this directory.
    set /p OVERWRITE="Delete it and start fresh? [y/N]: "
    if /i not "!OVERWRITE!"=="y" (
        echo Aborted.
        goto :done
    )
    echo Removing existing .venv ...
    rmdir /s /q .venv
)

echo.
echo Creating virtual environment ...
python -m venv .venv
if !ERRORLEVEL! neq 0 (
    echo ERROR: python -m venv failed. Make sure Python 3.10+ is on PATH.
    goto :done
)

echo Upgrading pip ...
.venv\Scripts\python -m pip install --upgrade pip

echo.
echo Installing PyTorch with CUDA 12.8 support ...
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if !ERRORLEVEL! neq 0 (
    echo WARNING: GPU install failed - falling back to CPU-only PyTorch.
    .venv\Scripts\python -m pip install torch torchvision
)

echo.
echo Installing remaining dependencies from requirements.txt ...
.venv\Scripts\python -m pip install -r requirements.txt

echo.
echo Installing dvfopt package in editable mode ...
.venv\Scripts\python -m pip install -e .

echo.
echo ============================================================
echo   Done!  To use this environment in VS Code:
echo     1. Open the Command Palette  (Ctrl+Shift+P)
echo     2. "Python: Select Interpreter"
echo     3. Choose  .\.venv\Scripts\python.exe
echo   Then select the .venv kernel in any notebook.
echo ============================================================
goto :done

:: ---- Uninstall ----
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
echo Done - virtual environment removed.

:done
echo.
pause
