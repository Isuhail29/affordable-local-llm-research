@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Starting your LOCAL AI  (Qwen3.6 35B, DEEP REASONING)
echo.
echo   The newest-generation model: thinks before it answers.
echo   Expect 30-90 seconds of visible "thinking", then
echo   high-quality answers at ~37-40 words/sec.
echo   For instant answers use Start-30B-AI.bat instead.
echo.
echo   Step 1: warming up memory (~20 GB model)...
echo ==========================================================
copy /b "models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" NUL > nul
echo   Step 2: starting. Browser opens in ~40 seconds.
echo   KEEP THIS WINDOW OPEN. Close it to stop the AI.
echo ==========================================================
start "" cmd /c "timeout /t 35 /nobreak >nul && start http://127.0.0.1:8080"
"llama.cpp\bin-b10068\llama-server.exe" -m "models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" -ngl 99 --n-cpu-moe 32 -fa on -c 16384 --mlock -t 12 -np 1 --host 127.0.0.1 --port 8080
pause
