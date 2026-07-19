@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Starting your LOCAL AI  (Qwen3 30B, 64K LONG CONTEXT)
echo.
echo   This mode reads BOOKS: up to ~45,000 words of context.
echo   Speed: ~18-24 words/sec depending on how full it gets.
echo   NOTE: feeding it a huge document takes minutes to read
echo   in (one-time per conversation). For normal chats use
echo   Start-30B-AI.bat instead.
echo.
echo   Step 1: warming up memory...
echo ==========================================================
copy /b "models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" NUL > nul
echo   Step 2: starting. Browser opens in ~30 seconds.
echo   KEEP THIS WINDOW OPEN. Close it to stop the AI.
echo ==========================================================
start "" cmd /c "timeout /t 25 /nobreak >nul && start http://127.0.0.1:8080"
"llama.cpp\bin\llama-server.exe" -m "models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" -ngl 99 --cpu-moe -fa on -c 65536 --mlock -t 12 -np 1 --host 127.0.0.1 --port 8080
pause
