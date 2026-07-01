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
folders are missing from custom_nodes/ are installed. Existing nodes can get a
lightweight optional-accelerator check without running ComfyUI-Manager.

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

# Patched custom nodes maintained from the user's GitHub should be handled first.
# These forks carry repo-specific fixes and compatibility patches, so the install
# pass checks/clones them before normal upstream/community nodes.
#
# Current patched forks:
# - ComfyUI-EasyOCR: robust OCR box handling and sensitivity presets.
# - ComfyUI-qwenmultiangle: Qwen multi-angle camera prompt controls and deps.
# - comfyui-flux2fun-controlnet: ComfyUI Flux timestep_zero_index compatibility.
# - ComfyUI-enricos-nodes: compositor fixes for VN foundation layout workflows.
PATCHED_NODE_FOLDERS = {
    "ComfyUI-EasyOCR",
    "ComfyUI-qwenmultiangle",
    "comfyui-flux2fun-controlnet",
    "ComfyUI-enricos-nodes",
}
ALWAYS_FIX_DEPENDENCIES = {
    "ComfyUI-EasyOCR",
    "ComfyUI-Watermark-Detection",
    "ComfyUI-qwenmultiangle",
    "Comfyui-LayerForge",
    "comfyui_face_parsing",
    "comfyui_controlnet_aux",
    # SeedVR2 pulls a sizable dependency set (omegaconf, einops, rotary
    # embeddings, etc.) that must be present before ComfyUI imports the node;
    # force its requirements.txt even under --no-deps.
    "ComfyUI-SeedVR2_VideoUpscaler",
}
EXTRA_PIP_DEPENDENCIES = {
    "ComfyUI-Watermark-Detection": [
        "ultralytics",
        "huggingface_hub",
    ],
}
# Optional GPU-only accelerators installed best-effort (never fatal) and only on
# the listed platforms. These are NOT required: the node falls back to PyTorch
# sdpa when they are absent, so a build failure or macOS run must not abort the
# rest of the manifest install. SeedVR2 benefits from SageAttention/Triton on
# CUDA machines; try Windows-friendly packages on Windows, Linux packages on
# Linux, and skip macOS/Apple Silicon. Do not build flash-attn on Windows by
# default: if a matching wheel is unavailable, it falls back to a fragile source
# build that requires the local CUDA toolkit to match the PyTorch CUDA version.
# Each platform-specific "pip_args" entry is passed verbatim after `pip install`.
OPTIONAL_ACCELERATORS = {
    "ComfyUI-SeedVR2_VideoUpscaler": {
        "platforms": ["linux", "windows"],
        "platform_pip_args": {
            "linux": [
                {"module": "triton", "pip_args": ["triton"]},
                {"module": "sageattention", "pip_args": ["sageattention==2.2.0", "--no-build-isolation"]},
                {"module": "flash_attn", "pip_args": ["flash-attn", "--no-build-isolation"]},
            ],
            "windows": [
                {"module": "triton", "pip_args": ["triton-windows"]},
                {"module": "sageattention", "pip_args": ["sageattention", "--no-build-isolation"]},
            ],
        },
    },
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


def is_patched_node_from_user_github(node: dict) -> bool:
    return node.get("folder") in PATCHED_NODE_FOLDERS or node.get("name") in PATCHED_NODE_FOLDERS


def manifest_nodes_in_install_order(manifest: dict) -> list[dict]:
    """Process patched user-maintained forks before normal upstream nodes."""
    nodes = list(manifest["nodes"])
    patched = [node for node in nodes if is_patched_node_from_user_github(node)]
    upstream = [node for node in nodes if not is_patched_node_from_user_github(node)]
    return patched + upstream


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


def install_optional_accelerators(python_bin: str, node: dict) -> None:
    """Best-effort install of optional GPU-only accelerators for a node. Never
    fatal: a build failure or an unsupported platform is logged and skipped so
    the node still works on its sdpa fallback and the rest of the install
    continues."""
    spec = OPTIONAL_ACCELERATORS.get(node["name"]) or OPTIONAL_ACCELERATORS.get(node["folder"])
    if not spec:
        return
    if current_os() not in spec["platforms"]:
        print(f"{node['name']}: skipping optional GPU accelerator on {current_os()} "
              f"(installed only on {spec['platforms']})", flush=True)
        return

    def module_available(module: str) -> bool:
        result = subprocess.run(
            [
                python_bin,
                "-c",
                "import importlib.util, sys; "
                "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                module,
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    pip_args_sets = spec.get("platform_pip_args", {}).get(current_os(), spec.get("pip_args", []))
    if pip_args_sets and isinstance(pip_args_sets[0], str):
        pip_args_sets = [pip_args_sets]
    for entry in pip_args_sets:
        if isinstance(entry, dict):
            module = entry.get("module")
            pip_args = entry["pip_args"]
        else:
            module = None
            pip_args = entry
        if module and module_available(module):
            print(f"{node['name']}: optional accelerator already available: {module}", flush=True)
            continue
        try:
            run([python_bin, "-m", "pip", "install", *pip_args])
        except subprocess.CalledProcessError:
            print(f"{node['name']}: optional accelerator install failed "
                  f"({' '.join(pip_args)}); continuing. The node still runs on "
                  f"sdpa if no accelerator is available.", flush=True)


def missing_manifest_nodes(manifest: dict) -> list[dict]:
    missing = []
    for node in manifest_nodes_in_install_order(manifest):
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


def diff_mode_existing_accelerator_nodes(manifest: dict) -> list[dict]:
    nodes = []
    for node in manifest_nodes_in_install_order(manifest):
        if not node_allowed_here(node):
            continue
        name = node["name"]
        folder_name = node["folder"]
        folder = CUSTOM_NODES_DIR / folder_name
        has_optional_accelerators = name in OPTIONAL_ACCELERATORS or folder_name in OPTIONAL_ACCELERATORS
        if folder.exists() and has_optional_accelerators:
            print(f"{folder} already exists; checking optional accelerators", flush=True)
            nodes.append(node)
    return nodes


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
            "and checks optional accelerators for selected existing nodes "
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
    existing_accelerator_nodes: list[dict] = []
    if install_mode == "full":
        nodes_to_install = []
        for node in manifest_nodes_in_install_order(manifest):
            if not node_allowed_here(node):
                print(f"{node['name']}: skipping on {current_os()} "
                      f"(platforms={node['platforms']})", flush=True)
                continue
            nodes_to_install.append(node)
    else:
        nodes_to_install = missing_manifest_nodes(manifest)
        existing_accelerator_nodes = diff_mode_existing_accelerator_nodes(manifest)
        seen = {node["folder"] for node in nodes_to_install}
        existing_accelerator_nodes = [
            node for node in existing_accelerator_nodes
            if node["folder"] not in seen
        ]

    if not nodes_to_install and not existing_accelerator_nodes:
        print("No missing custom nodes or optional accelerator checks found in manifest; diff install is complete.", flush=True)
        return

    python_bin = comfy_python()
    if nodes_to_install:
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
            install_optional_accelerators(python_bin, node)

    for node in existing_accelerator_nodes:
        install_optional_accelerators(python_bin, node)

    if install_mode == "full" or any(
        node["name"] == "ComfyUI-EasyOCR" or node["folder"] == "ComfyUI-EasyOCR"
        for node in nodes_to_install
    ):
        apply_post_install_fixes()


if __name__ == "__main__":
    main()
