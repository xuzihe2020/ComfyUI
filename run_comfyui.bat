@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ComfyUI virtual environment not found at "%CD%\.venv".
    exit /b 1
)

if not defined AUX_ANNOTATOR_CKPTS_PATH (
    set "AUX_ANNOTATOR_CKPTS_PATH=C:\Users\Tony Xu\workspace\comfyui_models\annotators"
)

".venv\Scripts\python.exe" "main.py" %*
