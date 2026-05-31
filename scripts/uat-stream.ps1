#requires -Version 5.1
<#
.SYNOPSIS
    Cross-backend UAT for WhisperForge streaming. Certifies CPU, CUDA, and WGPU
    against the LJ001-0001 reference clip.

.DESCRIPTION
    For each --device, runs two passes against test_data/LJ001-0001_16k.wav:

      1. Accuracy (offline, deterministic): --no-realtime --json. Parses the
         endpoint events' transcript and checks LJ keyword coverage (>= MinKeywords
         of 14). Identical decode math across backends => should pass on all three.

      2. Real-time keep-up (strict latency): same command WITHOUT --no-realtime.
         Scans stderr for any "[audio] dropped" line. Zero drops = the backend
         sustains real-time at the configured stride. If WGPU drops (expected, see
         CLAUDE.md ~3.5 s/window), re-run with a larger -StrideSecs until it passes
         and record that value as WGPU's recommended stride.

    Prints a PASS/FAIL matrix. Run on the Windows box (CUDA/WGPU native) and paste
    the output.

.PREREQUISITES
    Build with all backends:  cargo build --release -p whisperforge --features cuda
    Models present:           models/tiny_en_converted.{mpk,cfg}, models/tokenizer.json
    Audio fixture:            test_data/LJ001-0001_16k.wav

.EXAMPLE
    pwsh -File scripts/uat-stream.ps1
    pwsh -File scripts/uat-stream.ps1 -Devices wgpu -StrideSecs 2.0
#>
[CmdletBinding()]
param(
    [string]   $Binary      = ".\target\release\wforge.exe",
    [string]   $Model       = "tiny_en_converted",
    [string]   $Wav         = ".\test_data\LJ001-0001_16k.wav",
    [string[]] $Devices     = @("cpu", "cuda", "wgpu"),
    [int]      $MinKeywords = 10,
    [double]   $StrideSecs  = 1.0,
    # Accuracy pass uses a large window so a single continuous clip (LJ001-0001 is
    # one ~12 s sentence with no pause) does NOT hit the 5 s cap and lose content at
    # trim seams — this is the supported path for continuous monologue. The real-time
    # keep-up pass deliberately stays at the live default (5 s) to test live latency.
    [double]   $AccuracyMaxWindowSecs = 28.0
)

$ErrorActionPreference = "Stop"

# Same 14-keyword set as whisperforge-core/tests/stream_integration.rs.
$Keywords = @(
    "printing", "only", "sense", "which", "present", "concerned", "differs",
    "most", "from", "all", "arts", "crafts", "represented", "exhibition"
)

if (-not (Test-Path $Binary)) {
    throw "Binary not found: $Binary. Build it first: cargo build --release -p whisperforge --features cuda"
}
if (-not (Test-Path $Wav)) { throw "Audio fixture not found: $Wav" }

function Invoke-Stream {
    param([string] $Device, [bool] $Realtime, [double] $Stride, [double] $MaxWindow = 0.0)

    $outFile = [System.IO.Path]::GetTempFileName()
    $errFile = [System.IO.Path]::GetTempFileName()
    $cliArgs = @(
        "stream", "--model", $Model, "--from-file", $Wav,
        "--json", "--device", $Device, "--stride-secs", "$Stride"
    )
    if ($MaxWindow -gt 0.0) { $cliArgs += @("--max-window-secs", "$MaxWindow") }
    if (-not $Realtime) { $cliArgs += "--no-realtime" }

    # Use Start-Process with redirected files: the binary writes progress lines
    # ("Backend: Flex (CPU)", "Loading model…") to stderr, and under
    # `$ErrorActionPreference = 'Stop'` the `&` call operator would otherwise wrap
    # each native stderr line as a terminating RemoteException. Start-Process does not.
    $proc = Start-Process -FilePath $Binary -ArgumentList $cliArgs `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    $stdout = @(Get-Content -LiteralPath $outFile -ErrorAction SilentlyContinue)
    $stderr = (Get-Content -LiteralPath $errFile -Raw -ErrorAction SilentlyContinue)
    Remove-Item -LiteralPath $outFile, $errFile -ErrorAction SilentlyContinue

    [pscustomobject]@{ Stdout = $stdout; Stderr = $stderr; Exit = $proc.ExitCode }
}

function Get-Transcript {
    param([string[]] $Lines)
    $sb = [System.Text.StringBuilder]::new()
    foreach ($line in $Lines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try { $obj = $line | ConvertFrom-Json } catch { continue }
        if ($obj.type -eq "endpoint" -and $obj.text) {
            [void]$sb.Append($obj.text).Append(" ")
        }
    }
    $sb.ToString()
}

$results = @()

foreach ($dev in $Devices) {
    Write-Host "`n=== Device: $dev ===" -ForegroundColor Cyan

    # --- Pass 1: accuracy (offline, large window — no trim seams) ---
    Write-Host "  [1/2] accuracy (offline, max-window=${AccuracyMaxWindowSecs}s)..." -NoNewline
    $acc = Invoke-Stream -Device $dev -Realtime $false -Stride $StrideSecs -MaxWindow $AccuracyMaxWindowSecs
    if ($acc.Exit -ne 0) {
        Write-Host " ERROR (exit $($acc.Exit))" -ForegroundColor Red
        $snippet = ($acc.Stderr -split "`n" | Select-Object -Last 3) -join " | "
        Write-Host "      stderr: $snippet" -ForegroundColor DarkGray
        $results += [pscustomobject]@{ Device = $dev; Accuracy = "ERROR"; Hits = 0; RealTime = "SKIP"; Drops = "-" }
        continue
    }
    $transcript = Get-Transcript -Lines $acc.Stdout
    $lower = $transcript.ToLower()
    $hits = @($Keywords | Where-Object { $lower.Contains($_) })
    $accPass = $hits.Count -ge $MinKeywords
    Write-Host (" {0} ({1}/{2} keywords)" -f $(if ($accPass) { "PASS" } else { "FAIL" }), $hits.Count, $Keywords.Count) `
        -ForegroundColor $(if ($accPass) { "Green" } else { "Red" })
    Write-Host "      transcript: $($transcript.Trim())" -ForegroundColor DarkGray

    # --- Pass 2: real-time keep-up ---
    Write-Host "  [2/2] real-time keep-up (stride=${StrideSecs}s)..." -NoNewline
    $rt = Invoke-Stream -Device $dev -Realtime $true -Stride $StrideSecs
    $dropLines = @()
    if ($rt.Stderr) {
        $dropLines = @($rt.Stderr -split "`n" | Where-Object { $_ -match "\[audio\] dropped" })
    }
    $rtPass = ($rt.Exit -eq 0) -and ($dropLines.Count -eq 0)
    Write-Host (" {0} ({1} drop events)" -f $(if ($rtPass) { "PASS" } else { "FAIL" }), $dropLines.Count) `
        -ForegroundColor $(if ($rtPass) { "Green" } else { "Red" })
    if ($dropLines.Count -gt 0) {
        Write-Host "      hint: raise -StrideSecs until drops hit zero, record that value." -ForegroundColor Yellow
    }

    $results += [pscustomobject]@{
        Device   = $dev
        Accuracy = $(if ($accPass) { "PASS" } else { "FAIL" })
        Hits     = "$($hits.Count)/$($Keywords.Count)"
        RealTime = $(if ($rtPass) { "PASS" } else { "FAIL" })
        Drops    = $dropLines.Count
    }
}

Write-Host "`n===== UAT MATRIX (accuracy: max-window=${AccuracyMaxWindowSecs}s, min keywords=${MinKeywords}; real-time: live 5 s window, stride=${StrideSecs}s) =====" -ForegroundColor Cyan
$results | Format-Table -AutoSize

$failed = @($results | Where-Object { $_.Accuracy -ne "PASS" -or $_.RealTime -ne "PASS" })
if ($failed.Count -eq 0) {
    Write-Host "ALL BACKENDS PASS" -ForegroundColor Green
    exit 0
} else {
    Write-Host "$($failed.Count) backend(s) did not pass cleanly (see above)." -ForegroundColor Red
    exit 1
}
