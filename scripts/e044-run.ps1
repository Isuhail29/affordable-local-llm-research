# E044 driver: for each K, start llama-server with -np K, fire K concurrent gens, stop server.
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root 'llama.cpp\bin\llama-server.exe'
$model = Join-Path $root 'models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
$out = Join-Path $root 'benchmarks\e044-results.jsonl'
if (Test-Path $out) { Remove-Item $out -Force }

# warm the model file into page cache once
cmd /c "copy /b `"$model`" NUL > nul"

foreach ($run in @(@(1,'K1-first'), @(2,'K2'), @(4,'K4'), @(8,'K8'), @(1,'K1-reflank'))) {
    $K = $run[0]; $tag = $run[1]
    $args = "-m `"$model`" -ngl 99 --n-cpu-moe 40 -fa on -c 8192 --mlock -t 12 -np $K --host 127.0.0.1 --port 8080"
    $p = Start-Process -FilePath $server -ArgumentList $args -WindowStyle Hidden -PassThru `
        -RedirectStandardError (Join-Path $root "benchmarks\e044-$tag.server.log") `
        -RedirectStandardOutput (Join-Path $root "benchmarks\e044-$tag.server.out")
    try {
        $ready = $false
        foreach ($i in 1..90) {
            Start-Sleep -Seconds 2
            if ($p.HasExited) { throw "server exited (K=$K)" }
            try { $h = Invoke-RestMethod 'http://127.0.0.1:8080/health' -TimeoutSec 2; if ($h.status -eq 'ok') { $ready = $true; break } } catch {}
        }
        if (-not $ready) { throw "server not ready (K=$K)" }
        Push-Location $root
        python (Join-Path $root 'scripts\e044-batch-gate.py') $K $tag
        Pop-Location
    } finally {
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -Confirm:$false }
        Start-Sleep -Seconds 3
    }
}
Write-Host "E044 done. Results in benchmarks/e044-results.jsonl"
