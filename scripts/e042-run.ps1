# E042 condition runner. Starts the 35B on :8080 with the given sampler flags,
# runs the reasoning eval, scores exact integer match, records t/s. Stops its own server.
# Does NOT touch the hub on :9292.
param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [Parameter(Mandatory=$true)][string]$SamplerArgs
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$model = Join-Path $root 'models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'
$server = Join-Path $root 'llama.cpp\bin-b10068\llama-server.exe'
$out = Join-Path $root "benchmarks\e042-$Tag.jsonl"
$errlog = Join-Path $root "benchmarks\e042-$Tag.server.log"
if (Test-Path $out) { Remove-Item $out -Force }

$args = "-m `"$model`" -ngl 99 --n-cpu-moe 32 -fa on -c 8192 --mlock -t 12 -np 1 $SamplerArgs --host 127.0.0.1 --port 8080"
$p = Start-Process -FilePath $server -ArgumentList $args -WindowStyle Hidden -PassThru -RedirectStandardError $errlog -RedirectStandardOutput "$errlog.out"
try {
    $ready = $false
    foreach ($i in 1..60) { Start-Sleep -Seconds 2; if ($p.HasExited) { throw "server died" }; try { $h = Invoke-RestMethod "http://127.0.0.1:8080/health" -TimeoutSec 2; if ($h.status -eq 'ok') { $ready = $true; break } } catch {} }
    if (-not $ready) { throw "not healthy" }
    $correct = 0; $total = 0; $tps = @()
    foreach ($line in (Get-Content (Join-Path $root 'datasets\e042-reasoning.jsonl'))) {
        $prob = $line | ConvertFrom-Json
        $body = @{ messages = @(@{ role = "user"; content = $prob.q }); max_tokens = 3000; temperature = -1 } | ConvertTo-Json -Compress
        # temperature -1 sentinel removed; rely on server-baked sampler flags
        $body = @{ messages = @(@{ role = "user"; content = $prob.q }); max_tokens = 3000 } | ConvertTo-Json -Compress
        $tmp = Join-Path $env:TEMP "e042p.json"; $body | Out-File -Encoding ascii $tmp
        $r = curl.exe -s --max-time 300 -X POST "http://127.0.0.1:8080/v1/chat/completions" -H "Content-Type: application/json" --data-binary "@$tmp" | ConvertFrom-Json
        $c = $r.choices[0].message.content; $rc = $r.choices[0].message.reasoning_content
        $full = if ($c) { $c } else { $rc }
        # take the last 200 chars as the answer zone; exact integer match
        $tail = if ($full.Length -gt 200) { $full.Substring($full.Length - 200) } else { $full }
        $hit = $tail -match ("(?<![0-9])" + [regex]::Escape($prob.answer) + "(?![0-9])")
        if ($hit) { $correct++ }
        $total++
        $tps += $r.timings.predicted_per_second
        ([pscustomobject]@{ id = $prob.id; answer = $prob.answer; hit = [bool]$hit; tps = [math]::Round($r.timings.predicted_per_second,1); tail = $tail } | ConvertTo-Json -Compress) | Add-Content $out
    }
    $avgTps = [math]::Round(($tps | Measure-Object -Average).Average, 1)
    Write-Host "$Tag : $correct / $total correct | avg $avgTps t/s"
} finally {
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -Confirm:$false }
}
