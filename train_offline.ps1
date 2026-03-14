param(
    [switch]$SkipInstall,
    [string]$DatasetPath = "",
    [string]$PreprocessorPath = "",
    [string]$ModelPath = "",
    [string]$MetadataPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-CommandExists {
    param([Parameter(Mandatory = $true)][string]$CommandName)

    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

$projectRoot = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$backendDir = Join-Path -Path $projectRoot -ChildPath "backend"
$requirementsPath = Join-Path -Path $backendDir -ChildPath "requirements.txt"

Set-Location -Path $projectRoot

Write-Host "== Offline Fraud Model Trainer ==" -ForegroundColor Cyan
Write-Host "Project root: $projectRoot"

if (-not (Test-CommandExists -CommandName "python")) {
    throw "Python is not available in PATH. Please install Python 3.13+ and try again."
}

if (-not (Test-Path -Path $requirementsPath -PathType Leaf)) {
    throw "Backend requirements file not found: '$requirementsPath'."
}

if (-not $SkipInstall) {
    Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
    python -m pip install -r $requirementsPath
}

if ([string]::IsNullOrWhiteSpace($DatasetPath)) {
    $resolvedDatasetPath = Join-Path -Path $projectRoot -ChildPath "Bank_Transaction_Fraud_Detection.csv"
}
else {
    $resolvedDatasetPath = $DatasetPath
}

if (-not (Test-Path -Path $resolvedDatasetPath -PathType Leaf)) {
    throw "Training dataset not found at '$resolvedDatasetPath'."
}

$trainingArgs = @("-m", "app.ml.train_offline", "--dataset-path", $resolvedDatasetPath)

if (-not [string]::IsNullOrWhiteSpace($PreprocessorPath)) {
    $trainingArgs += @("--preprocessor-path", $PreprocessorPath)
}

if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    $trainingArgs += @("--model-path", $ModelPath)
}

if (-not [string]::IsNullOrWhiteSpace($MetadataPath)) {
    $trainingArgs += @("--metadata-path", $MetadataPath)
}

Write-Host "Starting offline model training..." -ForegroundColor Green
Set-Location -Path $backendDir
python @trainingArgs
$trainingExitCode = $LASTEXITCODE
Set-Location -Path $projectRoot

if ($trainingExitCode -ne 0) {
    throw "Offline training failed (exit code $trainingExitCode). Review the output above and retry."
}

Write-Host "Offline training finished successfully and saved frozen artifacts." -ForegroundColor Green
Write-Host "You can now launch the app with frozen artifacts using:" -ForegroundColor Magenta
Write-Host "powershell -ExecutionPolicy Bypass -File .\run_with_trained_model.ps1" -ForegroundColor Magenta
