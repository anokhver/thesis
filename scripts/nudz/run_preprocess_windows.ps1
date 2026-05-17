param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "full",

    [string]$InputRoot = "Z:\Brejtr\ActiveProjects\Microscopy",
    [string]$OutputRoot = "Z:\anokhver_temp\Microscopy",
    [string]$SmokeOutputRoot = "Z:\anokhver_temp\Microscopy_smoketest",

    [int]$Workers = 2,
    [string]$CondaEnv = "microscopy",

    [string]$CondaExe = "",
    [switch]$NoLog
)

$ErrorActionPreference = "Stop"

function Resolve-CondaExe {
    param([string]$ProvidedPath)

    if ($ProvidedPath -and (Test-Path $ProvidedPath)) {
        return (Resolve-Path $ProvidedPath).Path
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "Miniforge3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
    )

    foreach ($c in $candidates) {
        if (Test-Path $c) {
            return (Resolve-Path $c).Path
        }
    }

    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    throw "Could not find conda.exe. Pass -CondaExe explicitly."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

$condaPath = Resolve-CondaExe -ProvidedPath $CondaExe

if (-not (Test-Path $InputRoot)) {
    throw "Input root does not exist: $InputRoot"
}

$targetOutput = if ($Mode -eq "smoke") { $SmokeOutputRoot } else { $OutputRoot }
if (-not (Test-Path $targetOutput)) {
    New-Item -ItemType Directory -Path $targetOutput -Force | Out-Null
}

$batchScript = Join-Path $repoRoot "scripts\batch_preprocess.py"
if (-not (Test-Path $batchScript)) {
    throw "Batch script not found: $batchScript"
}

$args = @(
    "run",
    "-n", $CondaEnv,
    "python", "scripts/batch_preprocess.py",
    "--input_root", $InputRoot,
    "--output_root", $targetOutput,
    "--workers", $Workers
)

if ($Mode -eq "smoke") {
    $args += @("--max_folders", "1", "--max_files_per_folder", "1", "--workers", "1")
}

Write-Host "Mode:         $Mode"
Write-Host "Input root:   $InputRoot"
Write-Host "Output root:  $targetOutput"
Write-Host "Conda env:    $CondaEnv"
Write-Host "Conda exe:    $condaPath"
Write-Host "Repo root:    $repoRoot"

if ($NoLog) {
    $proc = Start-Process -FilePath $condaPath -ArgumentList $args -NoNewWindow -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "Preprocessing failed with exit code $($proc.ExitCode)."
    }
    Write-Host "Completed successfully."
    exit 0
}

$logDir = Join-Path $repoRoot "scripts\nudz\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "preprocess_${Mode}_${ts}.out.log"
$errLog = Join-Path $logDir "preprocess_${Mode}_${ts}.err.log"

Write-Host "Stdout log:   $outLog"
Write-Host "Stderr log:   $errLog"

$proc = Start-Process -FilePath $condaPath -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog
if ($proc.ExitCode -ne 0) {
    throw "Preprocessing failed with exit code $($proc.ExitCode). Check logs in $logDir"
}

Write-Host "Completed successfully. Logs saved in $logDir"
exit 0
