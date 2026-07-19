# Add any GGUF model from Hugging Face to the AI Hub.
# Usage: .\scripts\Add-Model.ps1 -Repo "unsloth/SomeModel-GGUF" -File "SomeModel-Q4_K_M.gguf" -Name "my-model"
# Then restart Start-AI-Hub.bat and the model appears everywhere.

param(
    [Parameter(Mandatory=$true)][string]$Repo,
    [Parameter(Mandatory=$true)][string]$File,
    [Parameter(Mandatory=$true)][string]$Name,
    [string]$ExtraArgs = "--cpu-moe"
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $root "models\$File"

if (-not (Test-Path $dest)) {
    Write-Host "Downloading $File from $Repo (resumable)..."
    curl.exe -L --fail --retry 5 -C - -o $dest "https://huggingface.co/$Repo/resolve/main/$File"
    if ($LASTEXITCODE -ne 0) { throw "download failed" }
}
Write-Host "Model on disk: $([math]::Round((Get-Item $dest).Length/1GB,1)) GB"

$entry = @"

  "$Name":
    ttl: 900
    cmd: >
      "$root\llama.cpp\bin-b10068\llama-server.exe"
      -m "$dest"
      -ngl 99 $ExtraArgs -fa on -c 8192 --mlock -t 12 -np 1 --port `${PORT}
"@
Add-Content -Path (Join-Path $root "llama-swap.yaml") -Value $entry -Encoding utf8
Write-Host "Registered '$Name' in llama-swap.yaml. Restart Start-AI-Hub.bat to use it."
Write-Host "Tip: for dense (non-MoE) models rerun with -ExtraArgs '' or a specific -ngl split."
