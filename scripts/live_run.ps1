# One prospective-forecast cycle: fetch -> forecast -> score -> commit ledger.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\live_run.ps1
# Exit codes: 0 ok, 1 fetch failed, 2 forecast refused or failed (score still runs), 3 git commit failed.
# Output is also appended to logs\live_runs.log (git-ignored).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$sfi = Join-Path $root ".venv\Scripts\sfi.exe"
New-Item -ItemType Directory -Force (Join-Path $root "logs") | Out-Null
$log = Join-Path $root "logs\live_runs.log"

function Say([string]$msg) {
    $line = "[{0}] {1}" -f ([DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")), $msg
    Write-Host $line  # not Write-Output: that would leak into Run's return value
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Run([string]$label, [string[]]$cmdArgs) {
    Say "== $label"
    $out = & $sfi @cmdArgs 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    foreach ($l in $out) { Say "   $l" }
    Say "   (exit $code)"
    return $code
}

$status = 0
if ((Run "fetch" @("fetch")) -ne 0) { Say "RESULT: fetch failed; nothing forecast."; exit 1 }
if ((Run "forecast" @("forecast")) -ne 0) { $status = 2 }
[void](Run "score" @("score"))

$changed = git status --porcelain -- ledger
if ($changed) {
    $runId = (Get-Content (Join-Path $root "ledger\forecast_runs.jsonl") -Tail 1 | ConvertFrom-Json).run_id
    git add -- ledger | Out-Null
    $msg = "Archive prospective forecasts (run $runId)`n`nAutomated by scripts/live_run.ps1.`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    git commit -q -m $msg -- ledger
    if ($LASTEXITCODE -ne 0) { Say "RESULT: git commit failed"; exit 3 }
    Say ("committed ledger: " + (git log --format="%h %s" -1))
} else {
    Say "ledger unchanged; no commit"
}
if ($status -eq 2) { Say "RESULT: forecast refused or failed (see above)"; exit 2 }
Say "RESULT: ok"
exit 0
