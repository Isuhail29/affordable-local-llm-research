# How To Use Your Local AI (Browser + Coding)

## Chat in your browser (Open WebUI)

1. Double-click **Start-AI-Hub.bat**. A black window opens and stays open (that is the engine; closing it stops everything).
2. Wait about 20 seconds. Your browser opens **http://127.0.0.1:8081** automatically. If it does not, type that address in yourself.
3. First time only: create a local account (any name/email/password, it never leaves your machine).
4. Top-left, click the **model dropdown** and pick one. First message to a model takes 30-90 seconds while it loads; after that it is fast. Switching models triggers one reload.
5. To use vision: pick **qwen-vision**, click the **attach/image** icon in the message box, upload a picture, and ask about it.

## Which model to pick

| Model in the dropdown | Use it for | Speed |
|---|---|---|
| Qwen3-30B (fast general) | everyday chat, quick answers | ~33 t/s |
| Qwen3-30B turbo | same, a bit faster, tiny quality drop | ~42 t/s |
| Qwen3-30B UNCENSORED | unfiltered general chat | ~30 t/s |
| Qwen3.6-35B reasoning | hardest questions (thinks first, slower to start) | ~37 t/s |
| Qwen3-30B 64K | very long documents (~45,000 words) | ~18 t/s |
| GLM-4.7-Flash coding | strongest coding | ~31 t/s |
| Qwen3-Coder UNCENSORED | unfiltered coding | ~28 t/s |
| Qwen3-VL vision | reading images/screenshots | ~30 t/s |
| Qwen3-8B fast | quick simple tasks | ~50 t/s |

## Coding in a GUI (opencode)

1. Keep **Start-AI-Hub.bat** running.
2. Open the **opencode** app.
3. Pick a model from its dropdown under **Local AI Hub** (GLM-4.7-Flash for hardest coding, Qwen3-Coder UNCENSORED for unfiltered).
4. Open your project folder and start asking it to read, write, and edit code.
5. Ignore the "Free" models at the top of opencode's list: those run on opencode's cloud servers, not your machine.

## The one rule

Everything depends on **Start-AI-Hub.bat** being open. It is the switchboard (address **http://127.0.0.1:9292/v1**) that every app talks to. One engine, all your models, all local, all private.

## Adding any new model from Hugging Face

`.\scripts\Add-Model.ps1 -Repo "org/repo-name" -File "the-file.gguf" -Name "nickname"`

Then restart Start-AI-Hub.bat and the model appears in every app. GGUF format only.
