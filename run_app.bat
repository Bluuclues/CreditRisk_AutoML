@echo off
title KBA 2026 Credit Risk AutoML Engine
echo =====================================================================
echo  Starting Credit Risk AutoML with PyCaret & TabFM Engine...
echo =====================================================================
echo.

cd /d "%~dp0"

IF EXIST ".venv\Scripts\streamlit.exe" (
    echo [INFO] Launching using Python 3.10 virtual environment (.venv)...
    ".venv\Scripts\streamlit.exe" run app.py
) ELSE (
    echo [WARN] .venv virtual environment not detected!
    echo [INFO] Attempting to launch with system Streamlit...
    streamlit run app.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
