@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Starting your LOCAL AI HUB
echo.
echo   One switchboard, all your models:
echo     qwen-30b / qwen-30b-turbo / qwen-30b-64k
echo     qwen-35b-reasoning / qwen-8b-fast / glm-coder
echo.
echo   API for any app:   http://127.0.0.1:9292/v1
echo   Chat interface:    http://127.0.0.1:8081  (opens soon)
echo.
echo   Models load on first use (30-90s) and swap automatically.
echo   KEEP THIS WINDOW OPEN. Close it to stop everything.
echo ==========================================================
start "llama-swap" /min "tools\llama-swap\llama-swap.exe" --config "llama-swap.yaml" --listen "127.0.0.1:9292"
set OPENAI_API_BASE_URL=http://127.0.0.1:9292/v1
set OPENAI_API_KEY=local
start "" cmd /c "timeout /t 20 /nobreak >nul && start http://127.0.0.1:8081"
"%USERPROFILE%\.local\bin\open-webui.exe" serve --port 8081
pause
