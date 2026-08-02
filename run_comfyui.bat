@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ComfyUI virtual environment not found at "%CD%\.venv".
    exit /b 1
)

rem Keep the frontend toolchain isolated from the machine-wide Node install.
set "COMFYUI_NODE_RUNTIME=%~sdp0.runtime\node-v25.9.0-win-x64"
set "COMFYUI_PACKAGE_MANAGER=%~sdp0.runtime\package-manager"
if exist "%COMFYUI_NODE_RUNTIME%\node.exe" (
    set "PATH=%COMFYUI_NODE_RUNTIME%;%COMFYUI_PACKAGE_MANAGER%;%PATH%"
    echo Using repository-local Node.js 25.9.0 runtime.
)

if not defined AUX_ANNOTATOR_CKPTS_PATH (
    set "AUX_ANNOTATOR_CKPTS_PATH=C:\Users\Tony Xu\workspace\comfyui_models\annotators"
)

echo Checking manifest-managed custom nodes and dependencies...
".venv\Scripts\python.exe" "scripts\install_custom_nodes.py"
if errorlevel 1 exit /b 1

set "COMFYUI_EDITOR_FRONTEND_ROOT=%~dp0ComfyUI_frontend\dist"
echo Using manifest-installed editor-integrated frontend: "%COMFYUI_EDITOR_FRONTEND_ROOT%"
".venv\Scripts\python.exe" "main.py" --front-end-root "%COMFYUI_EDITOR_FRONTEND_ROOT%" %*
