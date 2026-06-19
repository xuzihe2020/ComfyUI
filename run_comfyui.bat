@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ComfyUI virtual environment not found at "%CD%\.venv".
    exit /b 1
)

".venv\Scripts\python.exe" "main.py" %*
