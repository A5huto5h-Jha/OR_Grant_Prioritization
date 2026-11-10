@echo off
set OPENBLAS_NUM_THREADS=1
echo ======================================================================
echo  Starting Socio-Economic Grant Prioritization Dashboard on http://127.0.0.1:8000
echo ======================================================================
start http://127.0.0.1:8000
if exist "C:\Users\ashut\AppData\Local\Programs\Python\Python311\python.exe" (
    "C:\Users\ashut\AppData\Local\Programs\Python\Python311\python.exe" server.py
) else (
    python server.py
)
pause
