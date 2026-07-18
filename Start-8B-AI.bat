@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Starting your LOCAL AI  (Qwen3 8B, the small fast one)
echo.
echo   A chat page will open in your browser in ~20 seconds.
echo   KEEP THIS BLACK WINDOW OPEN while you chat.
echo   To stop the AI: just close this window.
echo ==========================================================
copy /b "models\Qwen3-8B-Q4_K_M.gguf" NUL > nul
start "" cmd /c "timeout /t 15 /nobreak >nul && start http://127.0.0.1:8080"
"llama.cpp\bin\llama-server.exe" -m "models\Qwen3-8B-Q4_K_M.gguf" -ngl 34 -c 4096 -np 1 -t 8 --host 127.0.0.1 --port 8080
pause
