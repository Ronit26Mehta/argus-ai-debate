# BioSage Terminal - Build and Publish Script (PowerShell)
# This script automates the build and publish process

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BioSage Terminal - Build and Publish" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to script directory
Set-Location $PSScriptRoot

Write-Host "[Step 1] Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
Write-Host "Previous builds cleaned." -ForegroundColor Green
Write-Host ""

Write-Host "[Step 2] Building package..." -ForegroundColor Yellow
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Build completed successfully." -ForegroundColor Green
Write-Host ""

Write-Host "[Step 3] Checking distribution..." -ForegroundColor Yellow
twine check dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Distribution check failed!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Distribution check passed." -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Files created in 'dist' folder:" -ForegroundColor Cyan
Get-ChildItem dist
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Test upload: " -NoNewline
Write-Host "twine upload --repository testpypi dist/*" -ForegroundColor White
Write-Host "  2. Production upload: " -NoNewline
Write-Host "twine upload dist/*" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit"
