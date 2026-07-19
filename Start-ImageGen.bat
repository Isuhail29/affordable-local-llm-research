@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Starting your LOCAL IMAGE GENERATOR (Pony Diffusion V6 XL)
echo.
echo   Uncensored image generation on your own GPU.
echo   API: http://127.0.0.1:8082/v1  (OpenAI image format)
echo.
echo   To use in the browser (Open WebUI):
echo     Settings ^> Images ^> Engine = OpenAI
echo     Base URL = http://127.0.0.1:8082/v1
echo     then click the image icon in any chat.
echo.
echo   NOTE: uses ~6.5 GB VRAM. Do NOT run a big LLM at the
echo   same time (they share the 8 GB card). Close the AI Hub
echo   first if it is running a model.
echo   KEEP THIS WINDOW OPEN. Close it to stop the generator.
echo ==========================================================
"sd-cpp-src\build\bin\sd-server.exe" -m "models\ponyDiffusionV6XL.safetensors" --vae "models\sdxl_vae.safetensors" --listen-ip 127.0.0.1 --listen-port 8082 --diffusion-fa
pause
