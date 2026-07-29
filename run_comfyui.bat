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

".venv\Scripts\python.exe" "scripts\install_custom_nodes.py" --check-editor-integration
if errorlevel 1 exit /b 1

set "COMFYUI_EDITOR_FRONTEND_ROOT=%~dp0tools\ComfyUI_frontend\dist"
echo Using manifest-installed editor-integrated frontend: "%COMFYUI_EDITOR_FRONTEND_ROOT%"
".venv\Scripts\python.exe" "main.py" --front-end-root "%COMFYUI_EDITOR_FRONTEND_ROOT%" %*
