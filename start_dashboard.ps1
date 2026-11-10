$env:OPENBLAS_NUM_THREADS = "1"
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Starting Socio-Economic Grant Prioritization Dashboard on http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8000"

$pythonPath = "C:\Users\ashut\AppData\Local\Programs\Python\Python311\python.exe"
if (Test-Path $pythonPath) {
    & $pythonPath (Join-Path $PSScriptRoot "server.py")
} else {
    python (Join-Path $PSScriptRoot "server.py")
}
