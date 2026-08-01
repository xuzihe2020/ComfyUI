#!/usr/bin/env python3
"""Install every ComfyUI dependency listed in custom_nodes.manifest.json.

Run from the repository root.

Windows PowerShell:

    cd "C:\\Users\\Tony Xu\\workspace\\comfyui"
    python .\\scripts\\install_custom_nodes.py

macOS/Linux:

    cd /path/to/comfyui
    python3 scripts/install_custom_nodes.py

The default command also installs the pinned comfyui-editor-bridge directly
under custom_nodes/, clones the pinned custom ComfyUI frontend directly under
the ComfyUI repository root, installs its JavaScript dependencies, and
produces its production dist build. run_comfyui.bat verifies and serves that
exact build; it never silently falls back to the packaged frontend.

Default behavior is diff mode: only dependencies whose installed state differs
from the manifest are processed. Existing nodes can get a dependency fix or
lightweight optional-accelerator check without running ComfyUI-Manager for every
node.

Diff mode is also SCOPED per repo: the stamp records every repo's HEAD, and
dependency work runs only for nodes whose own repo actually changed since the
last successful run. A ComfyUI-Manager-only update (routine — upstream moves
daily) refreshes Manager's requirements and nothing else. All cm-cli "fix"
calls for changed nodes are batched into ONE invocation, because every cm-cli
process re-fetches the ComfyRegistry (minutes) before doing any work.

The manifest's "tools" section lists standalone tool repos (e.g. video_sampler)
that are cloned into tools/ the same way: missing folders are cloned and their
requirements.txt installed; existing clones are left in place.

Manifest nodes with `"install_method": "git-clone"` bypass ComfyUI-Manager and
are cloned directly into custom_nodes/. This supports self-contained frontend
extensions whose upstream installation instructions require a direct clone.

Show help/options only:

    python .\\scripts\\install_custom_nodes.py --help
    python3 scripts/install_custom_nodes.py --help

Useful commands on Windows:

    python .\\scripts\\install_custom_nodes.py
    python .\\scripts\\install_custom_nodes.py --node "Comfy Canvas"
    python .\\scripts\\install_custom_nodes.py --no-deps
    python .\\scripts\\install_custom_nodes.py --full
    python .\\scripts\\install_custom_nodes.py --full --manager-fix-existing

Useful commands on macOS/Linux:

    python3 scripts/install_custom_nodes.py
    python3 scripts/install_custom_nodes.py --node "Comfy Canvas"
    python3 scripts/install_custom_nodes.py --no-deps
    python3 scripts/install_custom_nodes.py --full
    python3 scripts/install_custom_nodes.py --full --manager-fix-existing
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "custom_nodes.manifest.json"
DEFAULT_EXTRA_MODEL_PATHS = REPO_ROOT / "extra_model_paths.yaml"
CUSTOM_NODES_DIR = REPO_ROOT / "custom_nodes"
TOOLS_DIR = REPO_ROOT / "tools"
FRONTEND_ROOT = REPO_ROOT
FRONTEND_BUILD_MARKER = ".comfyui-frontend-build.json"
# Written after every successful run; diff mode exits immediately when the
# recomputed state (manifest hash + every repo's HEAD) matches the stamp, so a
# no-change run costs seconds instead of re-running dependency fixes and
# cm-cli (whose registry fetch alone is minutes per node).
INSTALL_STAMP = CUSTOM_NODES_DIR / ".install_state.json"
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
    "ComfyUI_essentials",
    "ComfyUI-qwenmultiangle",
    "Comfyui-LayerForge",
    "comfyui_face_parsing",
    "comfyui_controlnet_aux",
    # Acly's inpaint pack intentionally has no requirements.txt but imports
    # kornia at startup and loads LaMa through spandrel. Keep both dependencies
    # explicit so a node install never relies on ComfyUI startup to discover
    # or repair the environment.
    "comfyui-inpaint-nodes",
    # SeedVR2 pulls a sizable dependency set (omegaconf, einops, rotary
    # embeddings, etc.) that must be present before ComfyUI imports the node;
    # force its requirements.txt even under --no-deps.
    "ComfyUI-SeedVR2_VideoUpscaler",
    # GFPGAN imports native/runtime dependencies during ComfyUI startup.
    # Install everything up front and keep the gfpgan package's outdated
    # dependency chain disabled through the pinned post-install command below.
    "comfyui_gfpgan",
    # Fish S2's __init__.py auto-installs missing packages at ComfyUI startup;
    # force its requirements.txt (plus the --no-deps post-install fix below) so
    # that startup auto-installer never has anything to do.
    "ComfyUI-FishAudioS2",
    "ComfyUI-fish-audio-s2",
}
DIFF_MODE_FIX_EXISTING_DEPENDENCIES = {
    "ComfyUI_essentials",
}
SKIP_MANAGER_FIX_EXISTING = {
    # Fish S2 has its own startup dependency installer. Running Manager's
    # "fix" path for it can execute that same installer; use our pinned pip
    # path below instead.
    "ComfyUI-FishAudioS2",
    "ComfyUI-fish-audio-s2",
    "comfyui_gfpgan",
}
EXTRA_PIP_DEPENDENCIES = {
    "ComfyUI_essentials": [
        "rembg[cpu]",
        "rembg[gpu]",
    ],
    "ComfyUI-Watermark-Detection": [
        "ultralytics",
        "huggingface_hub",
    ],
    "comfyui-inpaint-nodes": [
        "kornia>=0.7.1",
        "spandrel",
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
# Shared linux GPU-accelerator chain. SageAttention 2.x is NOT on PyPI (the
# index stops at 1.0.6), so `sageattention==2.2.0` can never resolve — it must
# be compiled from the GitHub source (CUDA kernels; torch must be importable
# at build time, hence --no-build-isolation; ninja parallelizes the build).
# Pinned to the repo's v2.2.0 tag. TORCH_CUDA_ARCH_LIST covers our GPU pool
# (4090=8.9, 5090=12.0) so the ONE build persisted in the volume venv works on
# both pod types; extend the list before building if a new GPU family joins.
# One-time build cost on a pod: roughly 5-10 minutes.
LINUX_GPU_ACCELERATORS = [
    {"module": "triton", "pip_args": ["triton"]},
    {"module": "ninja", "pip_args": ["ninja"]},
    {
        "module": "sageattention",
        "pip_args": [
            "git+https://github.com/thu-ml/SageAttention.git@v2.2.0",
            "--no-build-isolation",
        ],
        "env": {"TORCH_CUDA_ARCH_LIST": "8.9;12.0"},
    },
    {"module": "flash_attn", "pip_args": ["flash-attn", "--no-build-isolation"]},
]

# Each platform-specific "pip_args" entry is passed verbatim after `pip install`.
OPTIONAL_ACCELERATORS = {
    "ComfyUI-SeedVR2_VideoUpscaler": {
        "platforms": ["linux", "windows"],
        "platform_pip_args": {
            "linux": LINUX_GPU_ACCELERATORS,
            "windows": [
                {"module": "triton", "pip_args": ["triton-windows"]},
                {"module": "sageattention", "pip_args": ["sageattention", "--no-build-isolation"]},
            ],
        },
    },
    # Fish S2 falls back to PyTorch sdpa when no accelerator is present, so
    # these stay best-effort. bitsandbytes (for the INT8/NF4 low-VRAM modes) is
    # already pinned in the node's requirements.txt and needs no entry here.
    "ComfyUI-FishAudioS2": {
        "platforms": ["linux", "windows"],
        "platform_pip_args": {
            "linux": LINUX_GPU_ACCELERATORS,
            "windows": [
                {"module": "triton", "pip_args": ["triton-windows"]},
                {"module": "sageattention", "pip_args": ["sageattention", "--no-build-isolation"]},
            ],
        },
    },
    "ComfyUI-fish-audio-s2": {
        "platforms": ["linux", "windows"],
        "platform_pip_args": {
            "linux": LINUX_GPU_ACCELERATORS,
            "windows": [
                {"module": "triton", "pip_args": ["triton-windows"]},
                {"module": "sageattention", "pip_args": ["sageattention", "--no-build-isolation"]},
            ],
        },
    },
}


def install_fish_audio_s2_dependencies(python_bin: str) -> None:
    """descript-audio-codec and descript-audiotools pin protobuf<5, which
    conflicts with other nodes in the shared environment, so they must be
    installed with --no-deps (the Fish S2 README warns that omitting the flag
    breaks other ComfyUI nodes). Their real runtime deps (flatten-dict, julius,
    ffmpy, argbind, etc.) are already pinned in the node's requirements.txt."""
    run([python_bin, "-m", "pip", "install", "descript-audio-codec", "--no-deps"])
    run([python_bin, "-m", "pip", "install", "descript-audiotools>=0.7.2", "--no-deps"])


def install_comfyui_gfpgan_dependencies(python_bin: str) -> None:
    """Install GFPGAN without its obsolete upstream BasicSR dependency.

    comfyui_gfpgan's requirements install basicsr-fixed, which is compatible
    with current ComfyUI/PyTorch. Installing gfpgan with dependencies enabled
    would replace it with the incompatible legacy basicsr package.
    """
    run([python_bin, "-m", "pip", "install", "gfpgan", "--no-deps"])


# Node-specific dependency installs that need pip flags EXTRA_PIP_DEPENDENCIES
# cannot express (it feeds one flat `pip install` command). Run whenever the
# node's dependencies are being fixed, same gating as EXTRA_PIP_DEPENDENCIES.
POST_INSTALL_DEPENDENCY_FIXES = {
    "ComfyUI-FishAudioS2": install_fish_audio_s2_dependencies,
    "ComfyUI-fish-audio-s2": install_fish_audio_s2_dependencies,
    "comfyui_gfpgan": install_comfyui_gfpgan_dependencies,
}

# Some custom nodes resolve model folders directly under folder_paths.models_dir
# and overwrite normal extra_model_paths registrations at import time. Keep
# their files external by creating narrow directory links from ComfyUI/models/
# to the configured external model root.
EXTERNAL_MODEL_DIRECTORY_LINKS = {
    "comfyui_gfpgan": {
        "face_detection": "face_detection",
        "face_restoration": "face_restoration",
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
    (e.g. CUDA-heavy nodes gated away from macOS). No key = install everywhere."""
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


def command_prefix(name: str) -> list[str] | None:
    """Return a subprocess-safe command prefix for native executables/scripts."""
    executable = shutil.which(name)
    if executable is None:
        return None
    if os.name == "nt" and Path(executable).suffix.lower() in {".bat", ".cmd"}:
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", executable]
    return [executable]


def pnpm_prefix(package_manager: str) -> list[str]:
    corepack = command_prefix("corepack")
    if corepack is not None:
        return [*corepack, package_manager]
    direct = command_prefix("pnpm")
    if direct is not None:
        expected_version = package_manager.partition("@")[2]
        result = subprocess.run(
            [*direct, "--version"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        actual_version = result.stdout.strip() if result.returncode == 0 else None
        if expected_version and actual_version != expected_version:
            raise SystemExit(
                f"The manifest requires {package_manager}, but pnpm "
                f"{actual_version or 'could not be executed'} is installed. "
                "Install/enable Corepack so the installer can run the pinned version."
            )
        return direct
    raise SystemExit(
        "The pinned ComfyUI frontend requires Node.js with pnpm or corepack. "
        "Install Node.js, then rerun scripts/install_custom_nodes.py."
    )


def validate_frontend_node_engine(frontend: dict) -> None:
    requirement = frontend.get("node_engine")
    if not requirement:
        return
    match = re.fullmatch(r">=(\d+)\s+<(\d+)", requirement)
    if match is None:
        raise SystemExit(
            f"Unsupported frontend node_engine requirement: {requirement!r}"
        )
    node = command_prefix("node")
    if node is None:
        raise SystemExit(
            f"The pinned ComfyUI frontend requires Node.js {requirement}. "
            "Install that Node.js version, then rerun scripts/install_custom_nodes.py."
        )
    result = subprocess.run(
        [*node, "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip()
    version_match = re.fullmatch(r"v?(\d+)(?:\.\d+){0,2}", version)
    minimum = int(match.group(1))
    maximum = int(match.group(2))
    if (
        result.returncode != 0
        or version_match is None
        or not minimum <= int(version_match.group(1)) < maximum
    ):
        raise SystemExit(
            f"The pinned ComfyUI frontend requires Node.js {requirement}, "
            f"but {version or 'Node.js could not be executed'} is active. "
            "Use the repository .nvmrc (nvm use), then rerun "
            "scripts/install_custom_nodes.py."
        )


def frontend_path(manifest: dict) -> Path:
    frontend = manifest.get("frontend")
    if not frontend:
        raise SystemExit("custom_nodes.manifest.json is missing the required frontend section.")
    return FRONTEND_ROOT / frontend["folder"]


def migrate_legacy_frontend_checkout(manifest: dict) -> None:
    """Move the short-lived tools/ install to the root-level managed path."""
    frontend = manifest["frontend"]
    target = frontend_path(manifest)
    legacy = TOOLS_DIR / frontend["folder"]
    if target.exists() or not legacy.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    legacy.rename(target)
    print(
        f"{frontend['name']}: migrated managed checkout from {legacy} to {target}.",
        flush=True,
    )


def frontend_build_marker_path(path: Path, dist: str = "dist") -> Path:
    return path / dist / FRONTEND_BUILD_MARKER


def frontend_build_marker(path: Path, dist: str = "dist") -> dict | None:
    try:
        with frontend_build_marker_path(path, dist).open(
            "r",
            encoding="utf-8",
        ) as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def frontend_package_version(path: Path) -> str | None:
    try:
        with (path / "package.json").open("r", encoding="utf-8") as f:
            package = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    value = package.get("version")
    return value if isinstance(value, str) and value else None


def editor_integration_errors(manifest: dict) -> list[str]:
    """Return actionable errors for the pinned bridge/frontend runtime."""
    errors: list[str] = []
    frontend = manifest.get("frontend")
    if not frontend:
        return ["manifest frontend section is missing"]

    frontend_dir = frontend_path(manifest)
    frontend_head = repo_head(frontend_dir) if frontend_dir.exists() else None
    expected_frontend_ref = frontend.get("ref")
    if frontend_head != expected_frontend_ref:
        errors.append(
            f"frontend checkout is {frontend_head or 'missing'}, "
            f"expected {expected_frontend_ref}"
        )
    package_version = frontend_package_version(frontend_dir)
    expected_version = frontend.get("version")
    if package_version != expected_version:
        errors.append(
            f"frontend package version is {package_version or 'missing'}, "
            f"expected {expected_version}"
        )

    dist_dir = frontend_dir / frontend.get("dist", "dist")
    for relative_path in frontend.get("required_dist_paths", ["index.html"]):
        if not (dist_dir / relative_path).is_file():
            errors.append(f"frontend build is missing {dist_dir / relative_path}")

    dist_name = frontend.get("dist", "dist")
    marker = frontend_build_marker(frontend_dir, dist_name)
    if marker is None:
        errors.append(
            "frontend build marker is missing: "
            f"{frontend_build_marker_path(frontend_dir, dist_name)}"
        )
    elif (
        marker.get("head") != frontend_head
        or marker.get("ref") != expected_frontend_ref
        or marker.get("version") != expected_version
    ):
        errors.append("frontend build marker does not match the pinned checkout")

    bridge = next(
        (
            node
            for node in manifest.get("nodes", [])
            if node.get("folder") == "comfyui-editor-bridge"
        ),
        None,
    )
    if bridge is None:
        errors.append("manifest comfyui-editor-bridge node is missing")
    else:
        bridge_dir = CUSTOM_NODES_DIR / bridge["folder"]
        bridge_head = repo_head(bridge_dir) if bridge_dir.exists() else None
        if bridge_head != bridge.get("ref"):
            errors.append(
                f"editor bridge checkout is {bridge_head or 'missing'}, "
                f"expected {bridge.get('ref')}"
            )
        for relative_path in bridge.get("required_paths", []):
            if not (bridge_dir / relative_path).is_file():
                errors.append(f"editor bridge is missing {bridge_dir / relative_path}")

    return errors


def check_editor_integration(manifest: dict) -> None:
    errors = editor_integration_errors(manifest)
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(
            "ComfyUI editor integration is not installed or is stale:\n"
            f"{details}\n"
            "Run scripts/install_custom_nodes.py, then start ComfyUI again."
        )
    print(
        "ComfyUI editor integration ready: pinned frontend build and editor bridge verified.",
        flush=True,
    )


def install_frontend(manifest: dict, *, install_mode: str) -> None:
    """Clone, dependency-install, and build the manifest-pinned frontend."""
    frontend = manifest["frontend"]
    migrate_legacy_frontend_checkout(manifest)
    target = frontend_path(manifest)
    before_head = repo_head(target) if target.exists() else None
    newly_cloned = not target.exists()
    clone_repo(frontend["repo"], target, frontend.get("ref"))
    after_head = repo_head(target)

    missing_source_paths = missing_required_paths(frontend, target)
    if missing_source_paths:
        raise SystemExit(
            f"{frontend['name']}: checkout is missing required paths: "
            + ", ".join(missing_source_paths)
        )
    package_version = frontend_package_version(target)
    if package_version != frontend.get("version"):
        raise SystemExit(
            f"{frontend['name']}: package version is "
            f"{package_version or 'missing'}, expected {frontend.get('version')}"
        )

    dist_name = frontend.get("dist", "dist")
    marker = frontend_build_marker(target, dist_name)
    dist_dir = target / dist_name
    missing_dist_paths = [
        relative_path
        for relative_path in frontend.get("required_dist_paths", ["index.html"])
        if not (dist_dir / relative_path).is_file()
    ]
    needs_build = (
        newly_cloned
        or install_mode == "full"
        or before_head != after_head
        or bool(missing_dist_paths)
        or marker is None
        or marker.get("head") != after_head
        or marker.get("ref") != frontend.get("ref")
        or marker.get("version") != frontend.get("version")
    )
    if not needs_build:
        # Only dist/ is a runtime artifact. Never retain the development
        # dependency tree on a persistent workstation or RunPod volume.
        shutil.rmtree(target / "node_modules", ignore_errors=True)
        shutil.rmtree(target / ".pnpm-build-store", ignore_errors=True)
        print(
            f"{frontend['name']}: pinned checkout and build are current; skipping build.",
            flush=True,
        )
        return

    package_manager = frontend.get("package_manager", "pnpm")
    if not package_manager.startswith("pnpm"):
        raise SystemExit(f"Unsupported frontend package manager: {package_manager}")
    validate_frontend_node_engine(frontend)
    pnpm = pnpm_prefix(package_manager)
    node_modules = target / "node_modules"
    build_store = target / ".pnpm-build-store"
    shutil.rmtree(node_modules, ignore_errors=True)
    shutil.rmtree(build_store, ignore_errors=True)
    try:
        run(
            [
                *pnpm,
                "install",
                "--frozen-lockfile",
                "--store-dir",
                str(build_store),
            ],
            cwd=target,
        )
        run([*pnpm, "run", frontend.get("build_script", "build")], cwd=target)
    finally:
        # The compiled dist is served directly by ComfyUI. Keeping the pnpm
        # virtual store/node_modules after the build wastes many GB and is not
        # required at runtime. Cleanup also runs after a failed build.
        shutil.rmtree(node_modules, ignore_errors=True)
        shutil.rmtree(build_store, ignore_errors=True)

    missing_dist_paths = [
        relative_path
        for relative_path in frontend.get("required_dist_paths", ["index.html"])
        if not (dist_dir / relative_path).is_file()
    ]
    if missing_dist_paths:
        raise SystemExit(
            f"{frontend['name']}: build completed but required outputs are missing: "
            + ", ".join(missing_dist_paths)
        )

    marker_value = {
        "head": after_head,
        "ref": frontend.get("ref"),
        "version": frontend.get("version"),
        "built_at": datetime.now().astimezone().isoformat(),
    }
    with frontend_build_marker_path(target, dist_name).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(marker_value, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"{frontend['name']}: build ready at {dist_dir}", flush=True)


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


def ensure_external_model_directory_links(manifest: dict) -> None:
    model_base = external_model_base()
    if model_base is None:
        return

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    eligible = {
        node["name"]
        for node in manifest.get("nodes", [])
        if node_allowed_here(node)
    }

    for node_name, links in EXTERNAL_MODEL_DIRECTORY_LINKS.items():
        if node_name not in eligible:
            continue
        for local_name, external_name in links.items():
            target = (model_base / external_name).resolve()
            link = models_dir / local_name
            target.mkdir(parents=True, exist_ok=True)

            if link.exists():
                if link.resolve() == target:
                    continue
                if link.is_dir() and not any(link.iterdir()):
                    link.rmdir()
                else:
                    raise SystemExit(
                        f"Cannot link {link} to {target}: the local path exists "
                        "and is not an empty directory or the expected link."
                    )

            if os.name == "nt":
                run(["cmd", "/c", "mklink", "/J", str(link), str(target)])
            else:
                link.symlink_to(target, target_is_directory=True)
            print(f"Linked external model directory: {link} -> {target}", flush=True)


def comfy_python() -> str:
    if os.name == "nt":
        candidate = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def clone_repo(repo: str, target: Path, ref: str | None = None) -> None:
    if target.exists():
        if not (target / ".git").exists():
            raise SystemExit(f"{target} exists but is not a git repository")
        if ref:
            run(["git", "fetch", "origin", ref], cwd=target)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target)
            print(f"{target}: checked out pinned ref {ref}")
        else:
            print(f"{target} already exists; leaving clone in place")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if ref:
        run(["git", "clone", "--no-checkout", repo, str(target)])
        run(["git", "fetch", "origin", ref], cwd=target)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target)
    else:
        run(["git", "clone", repo, str(target)])


def missing_required_paths(node: dict, target: Path) -> list[str]:
    """Return manifest-declared files missing from an installed checkout."""
    return [
        relative_path
        for relative_path in node.get("required_paths", [])
        if not (target / relative_path).exists()
    ]


def quarantine_incomplete_checkout(node: dict, target: Path) -> Path:
    """Move a partial direct clone aside so rerunning the installer can recover.

    Failed/interrupted `git clone` calls can leave a directory containing only
    `.git`. Merely checking `Path.exists()` permanently misclassifies that state
    as installed. Preserve the partial checkout for diagnosis, but move it out
    of ComfyUI's active node path before cloning again.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = target.with_name(f"{target.name}.incomplete-{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = target.with_name(
            f"{target.name}.incomplete-{timestamp}-{suffix}"
        )
        suffix += 1
    target.rename(candidate)
    print(
        f"{node['name']}: moved incomplete checkout to {candidate}",
        flush=True,
    )
    return candidate


def is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        return path.exists() and path.resolve() != path.absolute()
    except OSError:
        return False


def remove_link_or_junction(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    else:
        path.rmdir()


def install_git_clone_node(
    node: dict,
    python_bin: str,
    *,
    install_mode: str,
    no_deps: bool,
) -> None:
    """Install a self-contained manifest node without ComfyUI-Manager."""
    target = CUSTOM_NODES_DIR / node["folder"]
    if node.get("require_local_checkout") and is_link_or_junction(target):
        remove_link_or_junction(target)
        print(
            f"{node['name']}: removed legacy external link; "
            "installing the manifest-owned checkout.",
            flush=True,
        )
    newly_cloned = not target.exists()
    if target.exists():
        missing_paths = missing_required_paths(node, target)
        if missing_paths:
            print(
                f"{node['name']}: checkout is incomplete; missing "
                + ", ".join(missing_paths),
                flush=True,
            )
            quarantine_incomplete_checkout(node, target)
            newly_cloned = True
    clone_repo(node["repo"], target, node.get("ref"))

    missing_paths = missing_required_paths(node, target)
    if missing_paths:
        raise SystemExit(
            f"{node['name']}: clone completed but required paths are missing: "
            + ", ".join(missing_paths)
        )

    requirements = target / "requirements.txt"
    if (
        requirements.exists()
        and not no_deps
        and (newly_cloned or install_mode == "full")
    ):
        run([python_bin, "-m", "pip", "install", "-r", str(requirements)])


def install_tools(
    manifest: dict,
    python_bin: str,
    *,
    install_mode: str,
    no_deps: bool,
    changed_keys: set[str] | None = None,
) -> None:
    """Clone the standalone tool repos listed under "tools" in the manifest
    into tools/. Unlike custom nodes these are plain CLI repos that ComfyUI
    never imports, so there is no ComfyUI-Manager step: clone when the folder
    is missing, then install requirements.txt on first clone, in full mode,
    or when the tool's repo pulled new commits since the last stamp."""
    for tool in manifest.get("tools", []):
        if not node_allowed_here(tool):
            print(f"{tool['name']}: skipping on {current_os()} "
                  f"(platforms={tool['platforms']})", flush=True)
            continue
        target = TOOLS_DIR / tool["folder"]
        newly_cloned = not target.exists()
        clone_repo(tool["repo"], target, tool.get("ref"))
        requirements = target / "requirements.txt"
        tool_changed = changed_keys is not None and f"tools/{tool['folder']}" in changed_keys
        if requirements.exists() and not no_deps and (newly_cloned or install_mode == "full" or tool_changed):
            run([python_bin, "-m", "pip", "install", "-r", str(requirements)])


def install_manager(manifest: dict, python_bin: str, *, install_requirements: bool) -> Path:
    manager = manifest["manager"]
    manager_dir = CUSTOM_NODES_DIR / manager["folder"]
    clone_repo(manager["repo"], manager_dir, manager.get("ref"))

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
) -> str | None:
    """Install or dependency-fix one node. Returns the node name when it still
    needs a ComfyUI-Manager `fix` pass — the caller batches those into a single
    cm-cli call, because each cm-cli process re-fetches the ComfyRegistry."""
    folder = CUSTOM_NODES_DIR / node["folder"]
    repo = node["repo"]
    name = node["name"]
    always_fix_deps = name in ALWAYS_FIX_DEPENDENCIES or node["folder"] in ALWAYS_FIX_DEPENDENCIES
    extra_dependencies = EXTRA_PIP_DEPENDENCIES.get(name, []) + EXTRA_PIP_DEPENDENCIES.get(node["folder"], [])
    post_install_fix = POST_INSTALL_DEPENDENCY_FIXES.get(name) or POST_INSTALL_DEPENDENCY_FIXES.get(node["folder"])
    pinned_ref = node.get("ref")

    if pinned_ref:
        clone_repo(repo, folder, pinned_ref)
        requirements = folder / "requirements.txt"
        if requirements.exists() and (always_fix_deps or not no_deps):
            run([python_bin, "-m", "pip", "install", "-r", str(requirements)])
        if extra_dependencies and (always_fix_deps or not no_deps):
            run([python_bin, "-m", "pip", "install", *extra_dependencies])
        if post_install_fix and (always_fix_deps or not no_deps):
            post_install_fix(python_bin)
        return None

    base_cmd = [python_bin, str(manager_cli)]
    if folder.exists():
        print(f"{folder} already exists", flush=True)
        requirements = folder / "requirements.txt"
        if requirements.exists() and (always_fix_deps or not no_deps):
            run([python_bin, "-m", "pip", "install", "-r", str(requirements)])
        if extra_dependencies and (always_fix_deps or not no_deps):
            run([python_bin, "-m", "pip", "install", *extra_dependencies])
        if post_install_fix and (always_fix_deps or not no_deps):
            post_install_fix(python_bin)
        skip_manager_fix = name in SKIP_MANAGER_FIX_EXISTING or node["folder"] in SKIP_MANAGER_FIX_EXISTING
        if (manager_fix_existing or always_fix_deps) and not skip_manager_fix:
            return name
        if skip_manager_fix:
            print(f"{name}: skipping ComfyUI-Manager fix; dependencies were handled by pinned installer commands.", flush=True)
        return None

    cmd = base_cmd + ["install", repo, "--mode", "local", "--exit-on-fail"]
    if no_deps and not always_fix_deps:
        cmd.append("--no-deps")
    run(cmd, env=manager_env())
    if extra_dependencies and (always_fix_deps or not no_deps):
        run([python_bin, "-m", "pip", "install", *extra_dependencies])
    if post_install_fix and (always_fix_deps or not no_deps):
        post_install_fix(python_bin)
    return None


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
            env = None
            if isinstance(entry, dict) and entry.get("env"):
                env = {**os.environ, **entry["env"]}
            run([python_bin, "-m", "pip", "install", *pip_args], env=env)
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
            if (
                node.get("install_method") == "git-clone"
                and missing_required_paths(node, folder)
            ):
                print(
                    f"{folder} exists but its direct clone is incomplete; "
                    "scheduling repair",
                    flush=True,
                )
                missing.append(node)
                continue
            print(f"{folder} already exists; skipping in diff mode", flush=True)
            continue
        missing.append(node)
    return missing


def node_repo_changed(node: dict, changed_keys: set[str] | None) -> bool:
    """changed_keys=None means scoping is unavailable (first run, manifest
    edit, or full mode): treat every repo as changed, matching old behavior."""
    return changed_keys is None or f"custom_nodes/{node['folder']}" in changed_keys


def diff_mode_existing_dependency_nodes(
    manifest: dict, changed_keys: set[str] | None = None
) -> list[dict]:
    nodes = []
    for node in manifest_nodes_in_install_order(manifest):
        if not node_allowed_here(node):
            continue
        name = node["name"]
        folder_name = node["folder"]
        folder = CUSTOM_NODES_DIR / folder_name
        fix_existing_deps = name in DIFF_MODE_FIX_EXISTING_DEPENDENCIES or folder_name in DIFF_MODE_FIX_EXISTING_DEPENDENCIES
        always_fix_deps = name in ALWAYS_FIX_DEPENDENCIES or folder_name in ALWAYS_FIX_DEPENDENCIES
        pinned_ref = bool(node.get("ref"))
        if folder.exists() and (fix_existing_deps or always_fix_deps or pinned_ref):
            if not node_repo_changed(node, changed_keys):
                continue
            print(f"{folder} already exists; checking dependencies", flush=True)
            nodes.append(node)
    return nodes


def diff_mode_existing_accelerator_nodes(
    manifest: dict, changed_keys: set[str] | None = None
) -> list[dict]:
    nodes = []
    for node in manifest_nodes_in_install_order(manifest):
        if not node_allowed_here(node):
            continue
        name = node["name"]
        folder_name = node["folder"]
        folder = CUSTOM_NODES_DIR / folder_name
        has_optional_accelerators = name in OPTIONAL_ACCELERATORS or folder_name in OPTIONAL_ACCELERATORS
        if folder.exists() and has_optional_accelerators:
            if not node_repo_changed(node, changed_keys):
                continue
            print(f"{folder} already exists; checking optional accelerators", flush=True)
            nodes.append(node)
    return nodes


def repo_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def dependency_state(path: Path) -> str | None:
    """Track a real nested repo by HEAD; embedded nodes by dependency files.

    ``git -C embedded/path rev-parse HEAD`` walks up to ComfyUI's parent
    repository. That previously made every embedded node look changed after
    any unrelated ComfyUI commit and caused a large false pip refresh.
    """
    if not path.exists():
        return None
    if (path / ".git").exists():
        head = repo_head(path)
        return f"git:{head}" if head else None

    dependency_file_set = set(path.glob("requirements*.txt"))
    requirements_dir = path / "requirements"
    if requirements_dir.is_dir():
        dependency_file_set.update(requirements_dir.glob("*.txt"))
    dependency_file_set.update(
        candidate
        for candidate in (
            path / "pyproject.toml",
            path / "setup.py",
            path / "setup.cfg",
        )
        if candidate.is_file()
    )
    dependency_files = sorted(
        dependency_file_set,
        key=lambda item: str(item.relative_to(path)),
    )
    digest = hashlib.sha256()
    for dependency_file in dependency_files:
        digest.update(str(dependency_file.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(dependency_file.read_bytes())
        digest.update(b"\0")
    return f"deps:{digest.hexdigest()}"


def install_state_entry_changed(previous: str | None, current: str | None) -> bool:
    if previous == current:
        return False
    # Migrate legacy stamps that recorded the parent ComfyUI SHA for embedded
    # directories. The previous successful stamp already proves their deps
    # were installed; adopt the new dependency fingerprint without rerunning.
    if (
        isinstance(previous, str)
        and re.fullmatch(r"[0-9a-f]{40}", previous)
        and isinstance(current, str)
        and (
            current.startswith("deps:")
            or current == f"git:{previous}"
        )
    ):
        return False
    return True


def compute_install_state(manifest: dict, manifest_path: Path) -> dict:
    """Manifest hash + dependency revision of every eligible component."""
    repos: dict[str, str | None] = {}
    manager_folder = CUSTOM_NODES_DIR / manifest["manager"]["folder"]
    repos[f"custom_nodes/{manifest['manager']['folder']}"] = (
        dependency_state(manager_folder)
    )
    for node in manifest["nodes"]:
        if not node_allowed_here(node):
            continue
        folder = CUSTOM_NODES_DIR / node["folder"]
        repos[f"custom_nodes/{node['folder']}"] = dependency_state(folder)
    for tool in manifest.get("tools", []):
        if not node_allowed_here(tool):
            continue
        folder = TOOLS_DIR / tool["folder"]
        repos[f"tools/{tool['folder']}"] = dependency_state(folder)
    frontend = manifest.get("frontend")
    if frontend:
        folder = frontend_path(manifest)
        repos[f"frontend/{frontend['folder']}"] = (
            dependency_state(folder)
        )
    return {
        "manifest_md5": hashlib.md5(manifest_path.read_bytes()).hexdigest(),
        "repos": repos,
    }


def load_install_stamp() -> dict | None:
    try:
        with INSTALL_STAMP.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_install_stamp(state: dict) -> None:
    INSTALL_STAMP.parent.mkdir(parents=True, exist_ok=True)
    with INSTALL_STAMP.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


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
            "  python .\\scripts\\install_custom_nodes.py\n"
            "  python .\\scripts\\install_custom_nodes.py --node \"Comfy Canvas\"\n"
            "  python .\\scripts\\install_custom_nodes.py --no-deps\n"
            "  python .\\scripts\\install_custom_nodes.py --full\n"
            "  python .\\scripts\\install_custom_nodes.py --full --manager-fix-existing\n"
            "\n"
            "macOS/Linux examples:\n"
            "  python3 scripts/install_custom_nodes.py\n"
            "  python3 scripts/install_custom_nodes.py --node \"Comfy Canvas\"\n"
            "  python3 scripts/install_custom_nodes.py --no-deps\n"
            "  python3 scripts/install_custom_nodes.py --full\n"
            "  python3 scripts/install_custom_nodes.py --full --manager-fix-existing\n"
            "\n"
            "Show help/options only:\n"
            "  python .\\scripts\\install_custom_nodes.py --help\n"
            "  python3 scripts/install_custom_nodes.py --help\n"
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
            "and checks dependency fixes or optional accelerators for selected existing nodes "
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
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        metavar="NAME_OR_FOLDER",
        help=(
            "Install/check only the named manifest node (match by name or "
            "folder). Repeat to select multiple nodes. Scoped runs do not "
            "rewrite the global install-state stamp."
        ),
    )
    parser.add_argument(
        "--check-editor-integration",
        action="store_true",
        help="Verify the manifest-pinned frontend build and editor bridge, then exit.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.check_editor_integration:
        check_editor_integration(manifest)
        return
    scoped_node_run = bool(args.node)
    if scoped_node_run:
        requested = set(args.node)
        selected_nodes = [
            node
            for node in manifest["nodes"]
            if node["name"] in requested or node["folder"] in requested
        ]
        matched = {
            requested_value
            for requested_value in requested
            if any(
                node["name"] == requested_value
                or node["folder"] == requested_value
                for node in selected_nodes
            )
        }
        unknown = sorted(requested - matched)
        if unknown:
            raise SystemExit(
                "Unknown manifest node name/folder: " + ", ".join(unknown)
            )
        manifest = {
            **manifest,
            "nodes": selected_nodes,
            "tools": [],
        }
        print(
            "Scoped manifest install: "
            + ", ".join(node["name"] for node in selected_nodes),
            flush=True,
        )
    ensure_external_model_directory_links(manifest)
    install_mode = "full" if args.full else args.install_mode

    # None = no scoping possible (full mode, first run, or the manifest itself
    # changed): every repo is treated as changed. A set = only these repos get
    # dependency work; everything else is skipped outright.
    changed_keys: set[str] | None = None
    if install_mode == "diff" and not scoped_node_run:
        current_state = compute_install_state(manifest, args.manifest)
        stamp = load_install_stamp()
        if (current_state == stamp
                and None not in current_state["repos"].values()
                and not editor_integration_errors(manifest)):
            print(
                "Install state unchanged since last successful run; nothing to do. "
                f"(stamp: {INSTALL_STAMP}; use --full to force a re-check)",
                flush=True,
            )
            return
        if stamp and stamp.get("manifest_md5") == current_state["manifest_md5"]:
            stamped_repos = stamp.get("repos", {})
            changed_keys = {
                key for key, head in current_state["repos"].items()
                if head is None or install_state_entry_changed(
                    stamped_repos.get(key), head
                )
            }
            print(
                "Changed since last run: "
                + (", ".join(sorted(changed_keys)) if changed_keys else "(nothing)")
                + " — dependency work is scoped to these repos only.",
                flush=True,
            )

    if not scoped_node_run:
        install_frontend(manifest, install_mode=install_mode)
    install_tools(manifest, comfy_python(), install_mode=install_mode,
                  no_deps=args.no_deps, changed_keys=changed_keys)
    existing_dependency_nodes: list[dict] = []
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
        existing_dependency_nodes = diff_mode_existing_dependency_nodes(manifest, changed_keys)
        existing_accelerator_nodes = diff_mode_existing_accelerator_nodes(manifest, changed_keys)
        seen = {node["folder"] for node in nodes_to_install}
        existing_dependency_nodes = [
            node for node in existing_dependency_nodes
            if node["folder"] not in seen
        ]
        seen.update(node["folder"] for node in existing_dependency_nodes)
        existing_accelerator_nodes = [
            node for node in existing_accelerator_nodes
            if node["folder"] not in seen
        ]

    python_bin = comfy_python()
    manager_folder_name = manifest["manager"]["folder"]
    manager_dir = CUSTOM_NODES_DIR / manager_folder_name
    manager_changed = (changed_keys is not None
                       and f"custom_nodes/{manager_folder_name}" in changed_keys)
    git_clone_nodes = [
        node
        for node in nodes_to_install
        if node.get("install_method") == "git-clone"
    ]
    manager_nodes_to_install = [
        node
        for node in nodes_to_install
        if node.get("install_method", "manager") == "manager"
    ]
    unsupported_install_methods = [
        node
        for node in nodes_to_install
        if node.get("install_method", "manager") not in {"manager", "git-clone"}
    ]
    if unsupported_install_methods:
        details = ", ".join(
            f"{node['name']}={node.get('install_method')!r}"
            for node in unsupported_install_methods
        )
        raise SystemExit(f"Unsupported custom-node install method(s): {details}")

    for node in git_clone_nodes:
        install_git_clone_node(
            node,
            python_bin,
            install_mode=install_mode,
            no_deps=args.no_deps,
        )
        install_optional_accelerators(python_bin, node)

    nodes_requiring_manager = manager_nodes_to_install + existing_dependency_nodes

    if manager_changed and manager_dir.exists() and not nodes_requiring_manager:
        # Manager pulled new commits but no node needs work: refresh Manager's
        # own requirements (seconds) instead of the full per-node pass.
        requirements = manager_dir / "requirements.txt"
        if requirements.exists():
            run([python_bin, "-m", "pip", "install", "-r", str(requirements)])

    if not nodes_to_install and not existing_dependency_nodes and not existing_accelerator_nodes:
        print("No missing custom nodes, dependency fixes, or optional accelerator checks found in manifest; diff install is complete.", flush=True)
        if not scoped_node_run:
            check_editor_integration(manifest)
            write_install_stamp(compute_install_state(manifest, args.manifest))
        return

    if nodes_requiring_manager:
        manager_cli = install_manager(
            manifest,
            python_bin,
            install_requirements=(install_mode == "full" or manager_changed
                                  or not manager_dir.exists()),
        )

        fix_queue: list[str] = []
        for node in nodes_requiring_manager:
            fix_name = manager_install_node(
                python_bin=python_bin,
                manager_cli=manager_cli,
                node=node,
                no_deps=args.no_deps,
                manager_fix_existing=args.manager_fix_existing,
            )
            if fix_name:
                fix_queue.append(fix_name)
            install_optional_accelerators(python_bin, node)
        if fix_queue:
            # One batched call: every cm-cli process re-fetches the
            # ComfyRegistry before working, so N separate fixes = N fetches.
            run([python_bin, str(manager_cli), "fix", *fix_queue, "--mode", "local"],
                env=manager_env())

    for node in existing_accelerator_nodes:
        install_optional_accelerators(python_bin, node)

    if install_mode == "full" or any(
        node["name"] == "ComfyUI-EasyOCR" or node["folder"] == "ComfyUI-EasyOCR"
        for node in nodes_to_install
    ):
        apply_post_install_fixes()

    if not scoped_node_run:
        check_editor_integration(manifest)
        write_install_stamp(compute_install_state(manifest, args.manifest))


if __name__ == "__main__":
    main()
