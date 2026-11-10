@echo off
set OPENBLAS_NUM_THREADS=1
if exist "C:\Users\ashut\AppData\Local\Programs\Python\Python311\python.exe" (
    "C:\Users\ashut\AppData\Local\Programs\Python\Python311\python.exe" main.py %*
) else (
    python main.py %*
)
pause
