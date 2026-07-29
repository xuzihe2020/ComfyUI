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

set "COMFYUI_EDITOR_BRIDGE_SOURCE=%~dp0..\comfyui-editor-bridge"
set "COMFYUI_EDITOR_BRIDGE_TARGET=%~dp0custom_nodes\comfyui-editor-bridge"
if exist "%COMFYUI_EDITOR_BRIDGE_SOURCE%\__init__.py" (
    if not exist "%COMFYUI_EDITOR_BRIDGE_TARGET%\__init__.py" (
        echo Linking editor bridge custom node...
        mklink /J "%COMFYUI_EDITOR_BRIDGE_TARGET%" "%COMFYUI_EDITOR_BRIDGE_SOURCE%"
        if errorlevel 1 (
            echo Failed to link the editor bridge custom node.
            exit /b 1
        )
    )
)

if not defined COMFYUI_EDITOR_FRONTEND_ROOT (
    if exist "%~dp0..\ComfyUI_frontend\dist\index.html" (
        set "COMFYUI_EDITOR_FRONTEND_ROOT=%~dp0..\ComfyUI_frontend\dist"
    )
)

if defined COMFYUI_EDITOR_FRONTEND_ROOT (
    echo Using editor-integrated frontend: "%COMFYUI_EDITOR_FRONTEND_ROOT%"
    ".venv\Scripts\python.exe" "main.py" --front-end-root "%COMFYUI_EDITOR_FRONTEND_ROOT%" %*
) else (
    echo Editor-integrated frontend not found; using the packaged ComfyUI frontend.
    ".venv\Scripts\python.exe" "main.py" %*
)
