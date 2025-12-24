# E.D.I.T.H. Vehicle Identification System - PowerShell Launcher
# Usage: .\run.ps1 <image_path>

param(
    [Parameter(Mandatory=$true, Position=0, HelpMessage="Path to the image file")]
    [string]$ImagePath
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "E.D.I.T.H. Vehicle Identification System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if image file exists
if (-not (Test-Path $ImagePath)) {
    Write-Host "Error: Image file not found: $ImagePath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage: .\run.ps1 <image_path>" -ForegroundColor Yellow
    Write-Host "Example: .\run.ps1 C:\Users\YourName\Pictures\car.jpg" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Check if virtual environment exists
$venvPath = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & $venvPath
} else {
    Write-Host "Note: Virtual environment not found. Using system Python." -ForegroundColor Yellow
}

Write-Host "Processing image: $ImagePath" -ForegroundColor Green
Write-Host ""

# Run the identification pipeline
$exampleScript = Join-Path $PSScriptRoot "example.py"
python $exampleScript --image $ImagePath --verbose

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Processing complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
