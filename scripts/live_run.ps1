# One prospective-forecast cycle: fetch (schedules + current-season play-by-play and injuries)
# -> forecast -> score -> commit ledger -> push to origin (a failed push is logged, never fails the run).
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
if ((Run "fetch" @("fetch", "--live")) -ne 0) { Say "RESULT: fetch failed; nothing forecast."; exit 1 }
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

# Back up the ledger: push main to origin (also pushes commits left over from earlier failed pushes).
# Never fails the run: offline, auth or timeout problems are logged as warnings. Runs in a job so that
# git's stderr progress output can't trip $ErrorActionPreference, prompts are disabled, and it can time out.
$remote = git remote 2>$null | Where-Object { $_ -eq "origin" }
if ($remote) {
    Say "== push"
    $job = Start-Job -ArgumentList $root -ScriptBlock {
        param($r)
        $ErrorActionPreference = "Continue"
        Set-Location $r
        $env:GIT_TERMINAL_PROMPT = "0"   # no username/password prompt
        $env:GCM_INTERACTIVE = "never"   # no Git Credential Manager dialog
        $o = git push origin main 2>&1 | ForEach-Object { "$_" }
        [pscustomobject]@{ code = $LASTEXITCODE; out = $o }
    }
    if (Wait-Job $job -Timeout 120) {
        $res = Receive-Job $job
        foreach ($l in $res.out) { Say "   $l" }
        if ($res.code -eq 0) { Say "pushed main to origin" }
        else { Say "WARNING: git push failed (exit $($res.code)); ledger is committed locally and will be pushed next run" }
    } else {
        Stop-Job $job
        Say "WARNING: git push timed out after 120 s; ledger is committed locally and will be pushed next run"
    }
    Remove-Job $job -Force
} else {
    Say "no 'origin' remote configured; skipping push"
}

if ($status -eq 2) { Say "RESULT: forecast refused or failed (see above)"; exit 2 }
Say "RESULT: ok"
exit 0
