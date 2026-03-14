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

function Get-RequiredArtifacts {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)

    $artifactsDir = Join-Path -Path $ProjectRoot -ChildPath "backend/artifacts"
    return @{
        ArtifactsDir = $artifactsDir
        Preprocessor = Join-Path -Path $artifactsDir -ChildPath "preprocessor.joblib"
        Model = Join-Path -Path $artifactsDir -ChildPath "fraud_model.pt"
        Metadata = Join-Path -Path $artifactsDir -ChildPath "model_metadata.json"
    }
}

function Assert-FileExists {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if (-not (Test-Path -Path $Path -PathType Leaf)) {
        throw "Required trained artifact '$Name' is missing at '$Path'. Train and freeze the model first."
    }
}

function Assert-ArtifactMetadata {
    param([Parameter(Mandatory = $true)][hashtable]$Artifacts)

    try {
        $metadata = Get-Content -Path $Artifacts.Metadata -Raw | ConvertFrom-Json -Depth 20
    }
    catch {
        throw "Artifact metadata file is invalid JSON: $($Artifacts.Metadata)"
    }

    $expectedModelHash = [string]$metadata.artifact_hashes.model_sha256
    $expectedPreprocessorHash = [string]$metadata.artifact_hashes.preprocessor_sha256
    if ([string]::IsNullOrWhiteSpace($expectedModelHash) -or [string]::IsNullOrWhiteSpace($expectedPreprocessorHash)) {
        throw "Artifact metadata does not contain required SHA-256 hashes."
    }

    $actualModelHash = (Get-FileHash -Path $Artifacts.Model -Algorithm SHA256).Hash.ToLowerInvariant()
    $actualPreprocessorHash = (Get-FileHash -Path $Artifacts.Preprocessor -Algorithm SHA256).Hash.ToLowerInvariant()

    if ($actualModelHash -ne $expectedModelHash.ToLowerInvariant()) {
        throw "Model artifact hash mismatch. Retrain and republish artifacts before starting the app."
    }

    if ($actualPreprocessorHash -ne $expectedPreprocessorHash.ToLowerInvariant()) {
        throw "Preprocessor artifact hash mismatch. Retrain and republish artifacts before starting the app."
    }
}

$projectRoot = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
Set-Location -Path $projectRoot

Write-Host "== Bank Fraud Detection App Launcher (Trained Model Only) ==" -ForegroundColor Cyan
Write-Host "Project root: $projectRoot"

if (-not (Test-CommandExists -CommandName "python")) {
    throw "Python is not available in PATH. Please install Python 3.13+ and try again."
}

if (-not (Test-CommandExists -CommandName "npm")) {
    throw "npm is not available in PATH. Please install Node.js and try again."
}

$artifacts = Get-RequiredArtifacts -ProjectRoot $projectRoot
Assert-FileExists -Path $artifacts.Preprocessor -Name "preprocessor.joblib"
Assert-FileExists -Path $artifacts.Model -Name "fraud_model.pt"
Assert-FileExists -Path $artifacts.Metadata -Name "model_metadata.json"
Assert-ArtifactMetadata -Artifacts $artifacts

Write-Host "Verified trained model artifacts and integrity checks." -ForegroundColor Green

if (-not $SkipInstall) {
    Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
    python -m pip install -r (Join-Path -Path $projectRoot -ChildPath "backend/requirements.txt")

    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Set-Location -Path (Join-Path -Path $projectRoot -ChildPath "frontend")
    npm install
    Set-Location -Path $projectRoot
}

$backendCommand = "Set-Location '$projectRoot'; python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $BackendPort"
$frontendCommand = "Set-Location '$projectRoot/frontend'; npm run dev -- --host 0.0.0.0 --port $FrontendPort"

Write-Host "Starting backend on port $BackendPort (inference-only)..." -ForegroundColor Green
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

Write-Host "Done. Backend and frontend are running in separate PowerShell windows with frozen model artifacts." -ForegroundColor Green
