#!/usr/bin/env python3
"""Install the ComfyUI custom nodes listed in custom_nodes.manifest.json.

Run from the repository root.

Windows PowerShell:

    cd "C:\\Users\\Tony Xu\\workspace\\comfyui"
    python .\\script\\install_custom_nodes.py

macOS/Linux:

    cd /path/to/comfyui
    python3 script/install_custom_nodes.py

Default behavior is diff mode: only custom nodes listed in the manifest whose
folders are missing from custom_nodes/ are installed.

Show help/options only:

    python .\\script\\install_custom_nodes.py --help
    python3 script/install_custom_nodes.py --help

Useful commands on Windows:

    python .\\script\\install_custom_nodes.py
    python .\\script\\install_custom_nodes.py --no-deps
    python .\\script\\install_custom_nodes.py --full
    python .\\script\\install_custom_nodes.py --full --manager-fix-existing

Useful commands on macOS/Linux:

    python3 script/install_custom_nodes.py
    python3 script/install_custom_nodes.py --no-deps
    python3 script/install_custom_nodes.py --full
    python3 script/install_custom_nodes.py --full --manager-fix-existing
"""

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
    "Comfyui-LayerForge",
    "comfyui_face_parsing",
    "comfyui_controlnet_aux",
}
EXTRA_PIP_DEPENDENCIES = {
    "ComfyUI-Watermark-Detection": [
        "ultralytics",
        "huggingface_hub",
    ],
}


def current_os() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    return sys.platform


def node_allowed_here(node: dict) -> bool:
    """A manifest node with a "platforms" allowlist installs only on those OSes
    (e.g. ["linux"] for GPU-only nodes whose deps have no macOS/Windows wheels,
    like comfyui_controlnet_aux -> onnxruntime-gpu). No key = install everywhere."""
    platforms = node.get("platforms")
    return True if not platforms else current_os() in platforms


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


def install_manager(manifest: dict, python_bin: str, *, install_requirements: bool) -> Path:
    manager = manifest["manager"]
    manager_dir = CUSTOM_NODES_DIR / manager["folder"]
    clone_repo(manager["repo"], manager_dir)

    requirements = manager_dir / "requirements.txt"
    if install_requirements and requirements.exists():
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


def missing_manifest_nodes(manifest: dict) -> list[dict]:
    missing = []
    for node in manifest["nodes"]:
        if not node_allowed_here(node):
            print(f"{node['name']}: skipping on {current_os()} "
                  f"(platforms={node['platforms']})", flush=True)
            continue
        folder = CUSTOM_NODES_DIR / node["folder"]
        if folder.exists():
            print(f"{folder} already exists; skipping in diff mode", flush=True)
            continue
        missing.append(node)
    return missing


def apply_post_install_fixes() -> None:
    easyocr_docs = CUSTOM_NODES_DIR / "ComfyUI-EasyOCR" / "docs"
    source_font = easyocr_docs / "PingFangRegular.ttf"
    expected_font = easyocr_docs / "PingFang Regular.ttf"

    if source_font.exists() and not expected_font.exists():
        expected_font.write_bytes(source_font.read_bytes())
        print(f"Created EasyOCR expected font resource: {expected_font}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Windows examples:\n"
            "  python .\\script\\install_custom_nodes.py\n"
            "  python .\\script\\install_custom_nodes.py --no-deps\n"
            "  python .\\script\\install_custom_nodes.py --full\n"
            "  python .\\script\\install_custom_nodes.py --full --manager-fix-existing\n"
            "\n"
            "macOS/Linux examples:\n"
            "  python3 script/install_custom_nodes.py\n"
            "  python3 script/install_custom_nodes.py --no-deps\n"
            "  python3 script/install_custom_nodes.py --full\n"
            "  python3 script/install_custom_nodes.py --full --manager-fix-existing\n"
            "\n"
            "Show help/options only:\n"
            "  python .\\script\\install_custom_nodes.py --help\n"
            "  python3 script/install_custom_nodes.py --help\n"
        ),
    )
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
        "--install-mode",
        choices=("diff", "full"),
        default="diff",
        help=(
            "diff installs only manifest nodes whose custom_nodes folders are missing "
            "(default); full processes every manifest node."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Shortcut for --install-mode full.",
    )
    parser.add_argument(
        "--manager-fix-existing",
        action="store_true",
        help="In full mode, also run Manager's slower dependency fix for nodes whose folders already exist.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    install_mode = "full" if args.full else args.install_mode
    if install_mode == "full":
        nodes_to_install = []
        for node in manifest["nodes"]:
            if not node_allowed_here(node):
                print(f"{node['name']}: skipping on {current_os()} "
                      f"(platforms={node['platforms']})", flush=True)
                continue
            nodes_to_install.append(node)
    else:
        nodes_to_install = missing_manifest_nodes(manifest)

    if not nodes_to_install:
        print("No missing custom nodes found in manifest; diff install is complete.", flush=True)
        return

    python_bin = comfy_python()
    manager_dir = CUSTOM_NODES_DIR / manifest["manager"]["folder"]
    manager_cli = install_manager(
        manifest,
        python_bin,
        install_requirements=install_mode == "full" or not manager_dir.exists(),
    )

    for node in nodes_to_install:
        manager_install_node(
            python_bin=python_bin,
            manager_cli=manager_cli,
            node=node,
            no_deps=args.no_deps,
            manager_fix_existing=args.manager_fix_existing,
        )

    if install_mode == "full" or any(
        node["name"] == "ComfyUI-EasyOCR" or node["folder"] == "ComfyUI-EasyOCR"
        for node in nodes_to_install
    ):
        apply_post_install_fixes()


if __name__ == "__main__":
    main()
