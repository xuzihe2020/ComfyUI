#!/usr/bin/env python3
"""Install the ComfyUI custom nodes listed in custom_nodes.manifest.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "custom_nodes.manifest.json"
DEFAULT_EXTRA_MODEL_PATHS = REPO_ROOT / "extra_model_paths.yaml"
CUSTOM_NODES_DIR = REPO_ROOT / "custom_nodes"
ALWAYS_FIX_DEPENDENCIES = {
    "ComfyUI-EasyOCR",
    "ComfyUI-Watermark-Detection",
    "ComfyUI-qwenmultiangle",
}
EXTRA_PIP_DEPENDENCIES = {
    "ComfyUI-Watermark-Detection": [
        "ultralytics",
        "huggingface_hub",
    ],
}


def run(cmd: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def external_model_base(path: Path = DEFAULT_EXTRA_MODEL_PATHS) -> Path | None:
    if not path.exists():
        return None

    base_path = None
    has_download_model_base = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("base_path:"):
                base_path = line.split(":", 1)[1].strip().strip("'\"")
            elif line.startswith("download_model_base:"):
                has_download_model_base = True

    if base_path and has_download_model_base:
        return Path(base_path).expanduser().resolve()

    return None


def comfy_python() -> str:
    if os.name == "nt":
        candidate = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def clone_repo(repo: str, target: Path) -> None:
    if target.exists():
        if not (target / ".git").exists():
            raise SystemExit(f"{target} exists but is not a git repository")
        print(f"{target} already exists; leaving clone in place")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo, str(target)])


def install_manager(manifest: dict, python_bin: str) -> Path:
    manager = manifest["manager"]
    manager_dir = CUSTOM_NODES_DIR / manager["folder"]
    clone_repo(manager["repo"], manager_dir)

    requirements = manager_dir / "requirements.txt"
    if requirements.exists():
        run([python_bin, "-m", "pip", "install", "-r", str(requirements)])

    return manager_dir / "cm-cli.py"


def manager_env() -> dict[str, str]:
    env = os.environ.copy()
    env["COMFYUI_PATH"] = str(REPO_ROOT)
    model_base = external_model_base()
    if model_base is not None:
        env["COMFYUI_MODEL_PATH"] = str(model_base)
    return env


def manager_install_node(
    *,
    python_bin: str,
    manager_cli: Path,
    node: dict,
    no_deps: bool,
    manager_fix_existing: bool,
) -> None:
    folder = CUSTOM_NODES_DIR / node["folder"]
    repo = node["repo"]
    name = node["name"]
    always_fix_deps = name in ALWAYS_FIX_DEPENDENCIES or node["folder"] in ALWAYS_FIX_DEPENDENCIES
    extra_dependencies = EXTRA_PIP_DEPENDENCIES.get(name, []) + EXTRA_PIP_DEPENDENCIES.get(node["folder"], [])

    base_cmd = [python_bin, str(manager_cli)]
    if folder.exists():
        print(f"{folder} already exists", flush=True)
        requirements = folder / "requirements.txt"
        if requirements.exists() and (always_fix_deps or not no_deps):
            run([python_bin, "-m", "pip", "install", "-r", str(requirements)])
        if extra_dependencies and (always_fix_deps or not no_deps):
            run([python_bin, "-m", "pip", "install", *extra_dependencies])
        if manager_fix_existing or always_fix_deps:
            run(base_cmd + ["fix", name, "--mode", "local"], env=manager_env())
        return

    cmd = base_cmd + ["install", repo, "--mode", "local", "--exit-on-fail"]
    if no_deps and not always_fix_deps:
        cmd.append("--no-deps")
    run(cmd, env=manager_env())
    if extra_dependencies and (always_fix_deps or not no_deps):
        run([python_bin, "-m", "pip", "install", *extra_dependencies])


def apply_post_install_fixes() -> None:
    easyocr_docs = CUSTOM_NODES_DIR / "ComfyUI-EasyOCR" / "docs"
    source_font = easyocr_docs / "PingFangRegular.ttf"
    expected_font = easyocr_docs / "PingFang Regular.ttf"

    if source_font.exists() and not expected_font.exists():
        expected_font.write_bytes(source_font.read_bytes())
        print(f"Created EasyOCR expected font resource: {expected_font}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the custom node manifest.",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Ask ComfyUI-Manager to skip dependency installation for missing nodes.",
    )
    parser.add_argument(
        "--manager-fix-existing",
        action="store_true",
        help="Also run Manager's slower dependency fix for nodes whose folders already exist.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    python_bin = comfy_python()
    manager_cli = install_manager(manifest, python_bin)

    for node in manifest["nodes"]:
        manager_install_node(
            python_bin=python_bin,
            manager_cli=manager_cli,
            node=node,
            no_deps=args.no_deps,
            manager_fix_existing=args.manager_fix_existing,
        )

    apply_post_install_fixes()


if __name__ == "__main__":
    main()
