# Replicate Custom Nodes on Another Machine

This repo keeps custom node source folders out of git. The portable source of
truth is:

- `custom_nodes.manifest.json`
- `script/install_custom_nodes.py`
- `user/__manager/config.ini`

The installer bootstraps ComfyUI-Manager into `custom_nodes/`, then asks Manager
to install any node folders listed in the manifest that are missing locally.

## What Gets Installed

The current manifest installs:

- `ComfyUI-Manager`
- `ComfyUI-EasyOCR`
- `ComfyUI_essentials`
- `was-node-suite-comfyui`
- `comfyui-fast-mosaic-detector`

The node folders remain local machine state under `custom_nodes/`, which is
ignored by git.

## Desktop Setup

From the desktop machine, first sync or clone this ComfyUI repo.

```bash
cd /path/to/ComfyUI
```

Create a Python 3.12 virtual environment if the desktop checkout does not already
have one.

```bash
python3.12 -m venv .venv
```

Activate it.

For zsh/bash:

```bash
source .venv/bin/activate
```

For fish:

```fish
source .venv/bin/activate.fish
```

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install ComfyUI's base Python dependencies.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then install the custom nodes from the tracked manifest.

```bash
python script/install_custom_nodes.py
```

If the node folders already exist and you want Manager to repair/reinstall their
Python dependencies, run:

```bash
python script/install_custom_nodes.py --fix-existing
```

## Verify

You can verify the setup without launching the browser UI:

```bash
python main.py --cpu --disable-auto-launch --quick-test-for-ci
```

The expected result is exit code `0`. In the output, check that these custom
nodes appear in the import-time list:

- `ComfyUI-Manager`
- `ComfyUI-EasyOCR`
- `ComfyUI_essentials`
- `was-node-suite-comfyui`
- `comfyui-fast-mosaic-detector`

Warnings about missing CUDA on CPU machines, ComfyRegistry connectivity, or macOS
OpenCV/AV duplicate classes may appear and are not necessarily fatal. Treat a
Python traceback or nonzero exit code as the real failure signal.

## Updating the Node List

To add another shared custom node later:

1. Add an entry to `custom_nodes.manifest.json`.
2. Run `python script/install_custom_nodes.py` locally.
3. Commit the manifest change, not the downloaded folder under `custom_nodes/`.

The manifest entry should include `name`, `folder`, `repo`, and `reason`.

## Important Notes

- Do not commit full third-party node repos under `custom_nodes/`.
- Do not commit `user/__manager/cache/`; it is Manager registry cache.
- Model files are still separate from this process and belong under `models/` or
  your configured model paths.
- If Manager's UI is available later, it can still update/disable/uninstall the
  nodes. The manifest is just the repo-level record of what this project expects.
