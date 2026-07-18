# E014 harness: start llama-server with given args, run the standard prompt set, record timings, shut down.
# Usage: .\spec-decode-test.ps1 -ServerArgs "-m models\... -ngl 99 ..." -Tag "8b-nodraft"

param(
    [Parameter(Mandatory=$true)][string]$ServerArgs,
    [Parameter(Mandatory=$true)][string]$Tag
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root 'llama.cpp\bin\llama-server.exe'
$outFile = Join-Path $root "benchmarks\e014-$Tag.jsonl"
$errLog = Join-Path $root "benchmarks\e014-$Tag.server-err.log"
$outLog = Join-Path $root "benchmarks\e014-$Tag.server-out.log"
$promptDir = Join-Path $root 'scripts\e014-prompts'

$p = Start-Process -FilePath $server -ArgumentList "$ServerArgs --host 127.0.0.1 --port 8080" -PassThru -WindowStyle Hidden -RedirectStandardError $errLog -RedirectStandardOutput $outLog
try {
    $healthy = $false
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 2
        if ($p.HasExited) { throw "server exited during load, see $errLog" }
        try { $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 2; if ($h.status -eq 'ok') { $healthy = $true; break } } catch {}
    }
    if (-not $healthy) { throw "server not healthy after 120s" }
    # Global warmup: one throwaway generation so GPU clocks ramp before anything is measured
    $warm = Get-ChildItem $promptDir -Filter '*.json' | Sort-Object Name | Select-Object -First 1
    curl.exe -s -X POST 'http://127.0.0.1:8080/v1/chat/completions' -H 'Content-Type: application/json' --data-binary "@$($warm.FullName)" | Out-Null
    $vram = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    $clocks = (nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader | Select-Object -First 1).Trim()
    if (Test-Path $outFile) { Remove-Item $outFile -Force -Confirm:$false }
    foreach ($pf in (Get-ChildItem $promptDir -Filter '*.json' | Sort-Object Name)) {
        foreach ($run in 1, 2) {
            $raw = curl.exe -s -X POST 'http://127.0.0.1:8080/v1/chat/completions' -H 'Content-Type: application/json' --data-binary "@$($pf.FullName)"
            $r = $raw | ConvertFrom-Json
            if (-not $r.choices) { throw "bad response for $($pf.Name): $raw" }
            $text = $r.choices[0].message.content
            if ($text.Length -gt 300) { $text = $text.Substring(0, 300) }
            $rec = [pscustomobject]@{ tag = $Tag; prompt = $pf.BaseName; run = $run; vram_mib = $vram; clocks = $clocks; output_head = $text; timings = $r.timings } | ConvertTo-Json -Compress -Depth 6
            Add-Content -Path $outFile -Value $rec -Encoding utf8
        }
    }
    Write-Host "OK $Tag  (VRAM $vram MiB, clocks $clocks)"
} finally {
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -Confirm:$false }
}
