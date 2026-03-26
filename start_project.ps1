# start_project.ps1
# NeuroScan AI v2.0 - Prediction-only setup script

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "   NeuroScan AI v2.0 - Starting System" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Setting up Backend..." -ForegroundColor Yellow
$BackendPath = Join-Path $Root "backend"
Set-Location $BackendPath

if (-not (Test-Path "venv")) {
    Write-Host "  Creating virtual environment..."
    python -m venv venv
}

Write-Host "  Installing backend requirements..."
& ".\venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
Write-Host "  Backend dependencies installed." -ForegroundColor Green

$backendCmd = "Set-Location '$BackendPath'; .\venv\Scripts\activate; Write-Host 'Backend starting on http://localhost:8000' -ForegroundColor Green; uvicorn main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

Write-Host "  Waiting 3s for backend to start..."
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "Setting up Frontend..." -ForegroundColor Yellow
$FrontendPath = Join-Path $Root "frontend"
Set-Location $FrontendPath

if (-not (Test-Path "node_modules")) {
    Write-Host "  Installing npm packages..."
    npm install --silent
}

Write-Host "  Frontend dependencies ready." -ForegroundColor Green

$frontendCmd = "Set-Location '$FrontendPath'; Write-Host 'Frontend starting on http://localhost:3000' -ForegroundColor Green; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "  NeuroScan AI is launching!" -ForegroundColor Green
Write-Host "  Landing Page:  http://localhost:3000" -ForegroundColor Green
Write-Host "  Dashboard:     http://localhost:3000/dashboard" -ForegroundColor Green
Write-Host "  API Docs:      http://localhost:8000/docs" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Features available in this version:" -ForegroundColor Cyan
Write-Host "  AI Tumor Detection"
Write-Host "  Tumor Classification"
Write-Host "  Grad-CAM++ and Score-CAM visualisation"
Write-Host "  Uncertainty-aware predictions"
Write-Host "  Clinical PDF report generation"
Write-Host "  Landing page and analysis dashboard"
Write-Host ""
