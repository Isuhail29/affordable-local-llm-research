@echo off
cd /d "%~dp0"
echo ==========================================================
echo   LOCAL IMAGE STUDIO  (uncensored, on your own GPU)
echo.
echo   A simple page with a model dropdown:
echo     Pony V6 XL   - art / illustration / anime
echo     Juggernaut XL - photorealistic
echo     Chroma1-HD   - highest quality (slower)
echo.
echo   Your browser opens at http://127.0.0.1:8090
echo   Switching models reloads (~20s); same model stays warm.
echo.
echo   NOTE: uses the GPU. Close the AI Hub first if it has a
echo   model loaded. KEEP THIS WINDOW OPEN. Close it to stop.
echo ==========================================================
start "" cmd /c "timeout /t 4 /nobreak >nul && start http://127.0.0.1:8090"
python "scripts\image-studio.py"
pause
