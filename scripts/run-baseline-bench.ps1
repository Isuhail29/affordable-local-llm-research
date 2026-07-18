# Experiment 001: baseline benchmark sweep for Qwen3-8B Q4_K_M
# Produces JSON results in benchmarks/ for the offload sweep and the CPU thread sweep.
# Native tools run via cmd /c so their stderr output never becomes a PowerShell error record.

$ErrorActionPreference = 'Stop'
$root  = Split-Path $PSScriptRoot -Parent
$bin   = Join-Path $root 'llama.cpp\bin'
$model = Join-Path $root 'models\Qwen3-8B-Q4_K_M.gguf'
$outDir = Join-Path $root 'benchmarks'
$stamp = Get-Date -Format 'yyyy-MM-dd_HHmm'

if (-not (Test-Path $model)) { throw "Model not found: $model" }
$bench = Join-Path $bin 'llama-bench.exe'
if (-not (Test-Path $bench)) { throw "llama-bench not found: $bench" }

$envFile = Join-Path $outDir "$stamp-env.txt"
$cli = Join-Path $bin 'llama-cli.exe'

# Record environment
cmd /c "`"$cli`" --version > `"$envFile`" 2>&1"
cmd /c "nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv >> `"$envFile`" 2>&1"

# Sweep 1: GPU offload levels (0 = pure CPU, 99 = all 36 layers + output on GPU)
# Default pp512 (prefill) and tg128 (decode) tests, 5 repetitions each.
$out1 = Join-Path $outDir "$stamp-ngl-sweep.json"
$log1 = Join-Path $outDir "$stamp-ngl-sweep.log"
cmd /c "`"$bench`" -m `"$model`" -ngl 0,8,16,24,32,99 -o json > `"$out1`" 2> `"$log1`""
if ($LASTEXITCODE -ne 0) { throw "ngl sweep failed with exit code $LASTEXITCODE, see $log1" }

# Sweep 2: CPU thread scaling at ngl 0 (tests the bandwidth-saturation hypothesis)
$out2 = Join-Path $outDir "$stamp-thread-sweep.json"
$log2 = Join-Path $outDir "$stamp-thread-sweep.log"
cmd /c "`"$bench`" -m `"$model`" -ngl 0 -t 4,8,12,16,20,24 -o json > `"$out2`" 2> `"$log2`""
if ($LASTEXITCODE -ne 0) { throw "thread sweep failed with exit code $LASTEXITCODE, see $log2" }

Write-Host "Done. Results in $outDir with prefix $stamp"
