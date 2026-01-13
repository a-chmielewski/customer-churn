# PowerShell automation script for Customer Churn Prediction
# Windows-compatible alternative to Makefile

param(
    [Parameter(Position=0)]
    [ValidateSet('help', 'setup', 'install', 'test', 'preprocess', 'train', 'evaluate', 'explain', 'all', 'clean')]
    [string]$Command = 'help'
)

function Show-Help {
    Write-Host "`nCustomer Churn Prediction - Available Commands:" -ForegroundColor Green
    Write-Host ""
    Write-Host "  .\run.ps1 setup       - Create directories"
    Write-Host "  .\run.ps1 install     - Install Python dependencies"
    Write-Host "  .\run.ps1 test        - Run unit tests"
    Write-Host "  .\run.ps1 train       - Train models"
    Write-Host "  .\run.ps1 evaluate    - Evaluate on test set"
    Write-Host "  .\run.ps1 explain     - SHAP + business optimization"
    Write-Host "  .\run.ps1 all         - Run complete pipeline"
    Write-Host "  .\run.ps1 clean       - Remove generated files"
    Write-Host ""
}

function Setup {
    Write-Host "`nCreating project directories..." -ForegroundColor Yellow
    if (!(Test-Path "data\processed")) { New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null }
    if (!(Test-Path "models")) { New-Item -ItemType Directory -Force -Path "models" | Out-Null }
    if (!(Test-Path "reports\figures")) { New-Item -ItemType Directory -Force -Path "reports\figures" | Out-Null }
    Write-Host "Done!" -ForegroundColor Green
}

function Install {
    Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
    python -m pip install -q -U pip
    python -m pip install -q pandas numpy scikit-learn matplotlib seaborn joblib "shap<0.50"
    Write-Host "Dependencies installed!" -ForegroundColor Green
}

function Test {
    Write-Host "`nRunning unit tests..." -ForegroundColor Yellow
    python -m unittest discover tests -v
    Write-Host "`nTests complete!" -ForegroundColor Green
}

function Preprocess {
    Write-Host "`nRunning preprocessing pipeline..." -ForegroundColor Yellow
    python -m src.churn.preprocess
    Write-Host "Preprocessing complete!" -ForegroundColor Green
}

function Train {
    Write-Host "`nTraining models..." -ForegroundColor Yellow
    python -m src.churn.train
    Write-Host "Training complete!" -ForegroundColor Green
}

function Evaluate {
    Write-Host "`nEvaluating on test set..." -ForegroundColor Yellow
    python -m src.churn.evaluate
    Write-Host "Evaluation complete!" -ForegroundColor Green
}

function Explain {
    Write-Host "`nGenerating interpretation and business optimization..." -ForegroundColor Yellow
    python -m src.churn.explain
    Write-Host "Interpretation complete!" -ForegroundColor Green
}

function All {
    Preprocess
    Train
    Evaluate
    Explain
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  FULL PIPELINE COMPLETE" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results:"
    Write-Host "  - Models saved to: models/"
    Write-Host "  - Reports in: reports/"
    Write-Host "  - Visualizations in: reports/figures/"
    Write-Host ""
}

function Clean {
    Write-Host "`nCleaning generated files..." -ForegroundColor Yellow
    if (Test-Path "data\processed\*.npy") { Remove-Item "data\processed\*.npy" }
    if (Test-Path "models\*.pkl") { Remove-Item "models\*.pkl" }
    if (Test-Path "reports\*.csv") { Remove-Item "reports\*.csv" }
    if (Test-Path "reports\figures\*.png") { Remove-Item "reports\figures\*.png" }
    Write-Host "Clean complete!" -ForegroundColor Green
}

# Execute command
switch ($Command) {
    'help' { Show-Help }
    'setup' { Setup }
    'install' { Install }
    'test' { Test }
    'preprocess' { Preprocess }
    'train' { Train }
    'evaluate' { Evaluate }
    'explain' { Explain }
    'all' { All }
    'clean' { Clean }
}
