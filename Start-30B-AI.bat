@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Starting your LOCAL AI  (Qwen3 30B, the big smart one)
echo.
echo   Step 1: warming up memory (a few seconds)...
echo ==========================================================
copy /b "models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" NUL > nul
echo   Step 2: starting the AI. A chat page will open in your
echo   browser in about 30 seconds.
echo   KEEP THIS BLACK WINDOW OPEN while you chat.
echo   To stop the AI: just close this window.
echo ==========================================================
start "" cmd /c "timeout /t 25 /nobreak >nul && start http://127.0.0.1:8080"
echo   TIP: for full speed, close heavy apps (many Chrome tabs,
echo   video editors) before starting. The AI reserves 18 GB of RAM.
echo ==========================================================
"llama.cpp\bin\llama-server.exe" -m "models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" -ngl 99 --n-cpu-moe 40 -c 8192 -np 1 -t 12 --mlock --host 127.0.0.1 --port 8080
pause
