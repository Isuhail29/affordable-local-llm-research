@echo off
cd /d "%~dp0"
echo ==========================================================
echo   LOCAL IMAGE GENERATOR  (uncensored, on your own GPU)
echo.
echo   Pick a model:
echo     [1] Pony V6 XL    - art / illustration / anime (fast)
echo     [2] Juggernaut XL - photorealistic (fast)
echo     [3] Chroma1-HD    - highest quality, Flux-class (slower)
echo.
echo   NOTE: uses most of the 8 GB GPU. Close the AI Hub first
echo   if it has a model loaded. KEEP THIS WINDOW OPEN.
echo ==========================================================
choice /c 123 /n /m "Press 1, 2, or 3: "
set SD=sd-cpp-src\build\bin\sd-server.exe
if errorlevel 3 goto chroma
if errorlevel 2 goto jugg
if errorlevel 1 goto pony

:pony
echo Starting Pony V6 XL on http://127.0.0.1:8082/v1  (prompt tip: start with "score_9, score_8_up,")
"%SD%" -m "models\ponyDiffusionV6XL.safetensors" --vae "models\sdxl_vae.safetensors" --listen-ip 127.0.0.1 --listen-port 8082 --diffusion-fa
goto end

:jugg
echo Starting Juggernaut XL on http://127.0.0.1:8082/v1  (photoreal; normal prompts, no score tags)
"%SD%" -m "models\Juggernaut-XL-v9.safetensors" --listen-ip 127.0.0.1 --listen-port 8082 --diffusion-fa
goto end

:chroma
echo Starting Chroma1-HD on http://127.0.0.1:8082/v1  (Flux-class; T5 on CPU; slower but highest quality)
"%SD%" --diffusion-model "models\Chroma1-HD-Q5_K_S.gguf" --t5xxl "models\t5xxl-Q5_K_M.gguf" --vae "models\flux-ae.safetensors" --backend te=cpu --diffusion-fa --listen-ip 127.0.0.1 --listen-port 8082
goto end

:end
pause
