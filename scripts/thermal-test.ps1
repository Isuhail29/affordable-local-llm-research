# E032: sustained thermal test. Assumes llama-server already healthy on 127.0.0.1:8080.
# Runs identical 600-token generations for $Minutes, logging speed + sensors per request.

param([int]$Minutes = 10)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root "benchmarks\e032-thermal.jsonl"
if (Test-Path $out) { Remove-Item $out -Force -Confirm:$false }

$req = Join-Path $env:TEMP "e032-req.json"
'{"messages":[{"role":"user","content":"Write a detailed 450-word essay about the history of navigation at sea, from the stars to satellites."}],"max_tokens":600,"temperature":0}' | Out-File -Encoding ascii $req

$deadline = (Get-Date).AddMinutes($Minutes)
$i = 0
while ((Get-Date) -lt $deadline) {
    $i++
    $r = curl.exe -s -X POST "http://127.0.0.1:8080/v1/chat/completions" -H "Content-Type: application/json" --data-binary "@$req" | ConvertFrom-Json
    $gpu = (nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader,nounits).Trim() -split ',\s*'
    $cpuPerf = [math]::Round((Get-Counter '\Processor Information(_Total)\% Processor Performance').CounterSamples[0].CookedValue, 1)
    $rec = [pscustomobject]@{
        n = $i; t = (Get-Date).ToString('HH:mm:ss')
        tps = [math]::Round($r.timings.predicted_per_second, 2)
        n_gen = $r.timings.predicted_n
        gpu_temp_c = [int]$gpu[0]; gpu_sm_mhz = [int]$gpu[1]; gpu_mem_mhz = [int]$gpu[2]
        cpu_perf_pct = $cpuPerf
    } | ConvertTo-Json -Compress
    Add-Content -Path $out -Value $rec -Encoding utf8
    Write-Host $rec
}
Write-Host "done: $i generations logged to $out"
