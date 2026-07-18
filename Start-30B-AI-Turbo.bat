@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Starting your LOCAL AI  (Qwen3 30B TURBO mode)
echo.
echo   Turbo = ~20 percent faster by using 6 of 8 experts
echo   per word. Quality drops very slightly (measured: ~2%%).
echo   For full quality use Start-30B-AI.bat instead.
echo.
echo   Step 1: warming up memory (a few seconds)...
echo ==========================================================
copy /b "models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" NUL > nul
echo   Step 2: starting the AI. Browser opens in ~30 seconds.
echo   KEEP THIS WINDOW OPEN. Close it to stop the AI.
echo ==========================================================
start "" cmd /c "timeout /t 25 /nobreak >nul && start http://127.0.0.1:8080"
"llama.cpp\bin\llama-server.exe" -m "models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" -ngl 99 --n-cpu-moe 40 -c 8192 -np 1 -t 12 --mlock --override-kv qwen3moe.expert_used_count=int:6 --host 127.0.0.1 --port 8080
pause
