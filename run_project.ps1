param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$SkipInstall,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-CommandExists {
    param([Parameter(Mandatory = $true)][string]$CommandName)

    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

$projectRoot = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
Set-Location -Path $projectRoot

Write-Host "== Bank Fraud Detection App Launcher ==" -ForegroundColor Cyan
Write-Host "Project root: $projectRoot"

if (-not (Test-CommandExists -CommandName "python")) {
    throw "Python is not available in PATH. Please install Python 3.13+ and try again."
}

if (-not (Test-CommandExists -CommandName "npm")) {
    throw "npm is not available in PATH. Please install Node.js and try again."
}

$datasetPath = Join-Path -Path $projectRoot -ChildPath "Bank_Transaction_Fraud_Detection.csv"
if (-not (Test-Path -Path $datasetPath -PathType Leaf)) {
    Write-Warning "Dataset not found at '$datasetPath'. Training endpoint will fail until this file exists."
}

if (-not $SkipInstall) {
    Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
    python -m pip install -r (Join-Path -Path $projectRoot -ChildPath "backend/requirements.txt")

    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Set-Location -Path (Join-Path -Path $projectRoot -ChildPath "frontend")
    npm install
    Set-Location -Path $projectRoot
}

$backendCommand = "Set-Location '$projectRoot'; python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $BackendPort --reload"
$frontendCommand = "Set-Location '$projectRoot/frontend'; npm run dev -- --host 0.0.0.0 --port $FrontendPort"

Write-Host "Starting backend on port $BackendPort..." -ForegroundColor Green
Start-Process -FilePath "powershell" -WorkingDirectory $projectRoot -ArgumentList "-NoExit", "-Command", $backendCommand | Out-Null

Write-Host "Starting frontend on port $FrontendPort..." -ForegroundColor Green
Start-Process -FilePath "powershell" -WorkingDirectory (Join-Path -Path $projectRoot -ChildPath "frontend") -ArgumentList "-NoExit", "-Command", $frontendCommand | Out-Null

$backendUrl = "http://127.0.0.1:$BackendPort"
$frontendUrl = "http://127.0.0.1:$FrontendPort"

Write-Host ""
Write-Host "Backend URL:  $backendUrl" -ForegroundColor Cyan
Write-Host "Frontend URL: $frontendUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "Open this link in your browser to use the app:" -ForegroundColor Magenta
Write-Host "$frontendUrl" -ForegroundColor Magenta

if (-not $NoBrowser) {
    Write-Host "Launching browser..." -ForegroundColor DarkCyan
    Start-Process -FilePath $frontendUrl | Out-Null
}

Write-Host "Done. Backend and frontend are running in separate PowerShell windows." -ForegroundColor Green
