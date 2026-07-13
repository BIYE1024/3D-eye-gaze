@echo off
REM ── Manual Annotator launcher for Windows desktop ──
REM Double-click this file (or run in a terminal) to launch manual_annotate.py.
REM It auto-installs opencv-python if missing, then opens the interactive GUI
REM on your display so you can click pupil/iris points.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found on PATH.
    echo         Install Python 3.10/3.11 from python.org and tick "Add to PATH",
    echo         or use the docker image with an X server.
    pause
    exit /b 1
)

REM ensure opencv (GUI build) is available
python -c "import cv2" >nul 2>nul
if errorlevel 1 (
    echo [setup] installing opencv-python (GUI build) ...
    python -m pip install opencv-python
)

echo.
echo Controls:
echo   Left click = add point (green=pupil, blue=iris)
echo   p = pupil mode   i = iris mode   c = clear
echo   n / SPACE = save & next   b = save & back
echo   q / ESC = quit & save
echo   Output: images/0/annotations.json
echo.

python manual_annotate.py --image_dir images/0
pause
