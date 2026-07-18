# E025: build patched llama.cpp (CUDA, Blackwell-only) into llama.cpp-src/build
# Requires: VS Build Tools 2022 (VCTools workload), CMake, CUDA toolkit installed.

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$src  = Join-Path $root 'llama.cpp-src'

$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars (is VS Build Tools installed?)" }

$cfg = "cmake -S `"$src`" -B `"$src\build`" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF"
$bld = "cmake --build `"$src\build`" --config Release -j 20 --target llama-bench llama-server llama-cli"

cmd /c "`"$vcvars`" && $cfg && $bld"
if ($LASTEXITCODE -ne 0) { throw "build failed with exit code $LASTEXITCODE" }

Write-Host "Build OK: $src\build\bin\Release"
