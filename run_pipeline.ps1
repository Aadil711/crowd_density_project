# AI Crowd Density Prediction - Full Pipeline Runner
# Usage: .\run_pipeline.ps1 raw_video/street1.mp4

param(
    [string]$VideoPath = "raw_video/street1.mp4",
    [float]$FPS = 1.0,
    [string]$StartTime = ""
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Crowd Density Prediction Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Extract location name from video path
$LocationName = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath)
$FramesDir = "anonymized_frames/$LocationName"
$PredsDir = "models/preds/$LocationName"
$CSVPath = "counts_ts/${LocationName}_counts.csv"
$ModelPath = "models/lstm_${LocationName}.pth"

Write-Host "`n[1/5] Extracting frames..." -ForegroundColor Yellow
python scripts/extract_frames.py $VideoPath $FramesDir $FPS
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Frame extraction failed" -ForegroundColor Red; exit 1 }

Write-Host "`n[2/5] Counting crowd density..." -ForegroundColor Yellow
python scripts/count_crowd.py $FramesDir $PredsDir models/csrnet_shanghaitech.pth
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Crowd counting failed" -ForegroundColor Red; exit 1 }

Write-Host "`n[3/5] Building time series..." -ForegroundColor Yellow
if ($StartTime) {
    python scripts/build_timeseries.py "$PredsDir/counts.json" $CSVPath $StartTime $FPS
} else {
    python scripts/build_timeseries.py "$PredsDir/counts.json" $CSVPath
}
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Time series building failed" -ForegroundColor Red; exit 1 }

Write-Host "`n[4/5] Training LSTM forecaster..." -ForegroundColor Yellow
python scripts/train_lstm.py $CSVPath $ModelPath 10 50
if ($LASTEXITCODE -ne 0) { Write-Host "❌ LSTM training failed" -ForegroundColor Red; exit 1 }

Write-Host "`n[5/5] Starting services..." -ForegroundColor Yellow
Write-Host "Starting API server on port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd $PWD; .\venv\Scripts\activate; uvicorn app.api:app --reload --port 8000"

Start-Sleep -Seconds 3

Write-Host "Starting dashboard on port 8501..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd $PWD; .\venv\Scripts\activate; streamlit run app/dashboard.py"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ Pipeline Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nDashboard: http://localhost:8501" -ForegroundColor White
Write-Host "API Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host "`nPress Ctrl+C in API/Dashboard windows to stop" -ForegroundColor Yellow
