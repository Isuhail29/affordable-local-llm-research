# E042 via the hub (:9292), zero extra memory. Runs all 3 sampler conditions against
# qwen-35b-reasoning by varying per-request sampler params. Scores exact integer match.
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$probs = Get-Content (Join-Path $root 'datasets\e042-reasoning.jsonl') | ForEach-Object { $_ | ConvertFrom-Json }

$conditions = @(
    @{ tag = 'A-baseline'; s = @{ temperature = 0.8; top_p = 0.95; top_k = 40; min_p = 0.05 } },
    @{ tag = 'B-vendor';   s = @{ temperature = 1.0; top_p = 0.95; top_k = 20; min_p = 0.0; presence_penalty = 1.5 } },
    @{ tag = 'C-nsigma';   s = @{ temperature = 1.0; top_p = 0.95; top_k = 20; min_p = 0.0; presence_penalty = 1.5; top_n_sigma = 1.0 } }
)

foreach ($cond in $conditions) {
    $out = Join-Path $root "benchmarks\e042-$($cond.tag).jsonl"
    if (Test-Path $out) { Remove-Item $out -Force }
    $correct = 0; $total = 0; $tps = @()
    foreach ($prob in $probs) {
        $req = @{ model = 'qwen-35b-reasoning'; messages = @(@{ role = 'user'; content = $prob.q }); max_tokens = 3000 }
        foreach ($k in $cond.s.Keys) { $req[$k] = $cond.s[$k] }
        $tmp = Join-Path $env:TEMP "e042hub.json"; ($req | ConvertTo-Json -Depth 6 -Compress) | Out-File -Encoding ascii $tmp
        $r = curl.exe -s --max-time 400 -X POST "http://127.0.0.1:9292/v1/chat/completions" -H "Content-Type: application/json" --data-binary "@$tmp" | ConvertFrom-Json
        $c = $r.choices[0].message.content; $rc = $r.choices[0].message.reasoning_content
        $full = if ($c) { "$c" } else { "$rc" }
        $tail = if ($full.Length -gt 250) { $full.Substring($full.Length - 250) } else { $full }
        $hit = $tail -match ("(?<![0-9])" + [regex]::Escape($prob.answer) + "(?![0-9.])")
        if ($hit) { $correct++ }
        $total++; $tps += $r.timings.predicted_per_second
        ([pscustomobject]@{ id = $prob.id; answer = $prob.answer; hit = [bool]$hit; tps = [math]::Round($r.timings.predicted_per_second,1) } | ConvertTo-Json -Compress) | Add-Content $out
    }
    $avg = [math]::Round(($tps | Measure-Object -Average).Average, 1)
    Write-Host "$($cond.tag) : $correct / $total correct | avg $avg t/s"
}
