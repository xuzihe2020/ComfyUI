#!/usr/bin/env python3
"""Sync ComfyUI models between machines and the RunPod network volume, and
probe volume usage.

Two modes, two homes:

  TRANSFER (runs anywhere — Mac/Windows/pod; S3 API, no pod required):
    python scripts/runpod/model_sync.py -u checkpoints/flux2 loras/anna
    python scripts/runpod/model_sync.py -d loras/anna_v1.safetensors vae

  PROBE (runs ON A POD ONLY — real filesystem, du-based):

    # whole volume: tree of folders AND files with sizes + file counts,
    # then TOTAL / quota / free. Walks the venvs' ~130k files: ~45s.
    python /workspace/ComfyUI/scripts/runpod/model_sync.py --probe

    # any subpath under /workspace — instant on model/data trees
    python /workspace/ComfyUI/scripts/runpod/model_sync.py --probe comfyui_models

    # --depth N controls nesting of the breakdown (folders and files).
    # Default: 2. Higher = deeper tree, same walk time.
    python /workspace/ComfyUI/scripts/runpod/model_sync.py --probe comfyui_models/text_encoders --depth 3

Probing deliberately does NOT work from a workstation: the S3 API has no
aggregate sizes, so any remote probe degenerates into listing every object
(minutes). On a pod, `du` walks the mounted filesystem in seconds. RunPod
exposes no volume-usage API (verified: their networkvolumes endpoint returns
only id/name/size/dataCenterId).

Transfer path layout is mirrored on both sides:

    local    <LOCAL_ROOT>/<path>
    volume   comfyui_models/<path>     (pods see /workspace/comfyui_models/<path>)

Configuration (repo .env, auto-loaded; only needed for transfers):
    RUNPOD_S3_ACCESS_KEY_ID / RUNPOD_S3_SECRET_ACCESS_KEY   required
    RUNPOD_S3_ENDPOINT / RUNPOD_S3_REGION / RUNPOD_VOLUME_ID  preset defaults
    RUNPOD_VOLUME_SIZE_GB   quota used by --probe free-space math
    COMFYUI_MODELS_LOCAL_ROOT  overrides the local root

LOCAL_ROOT default: Windows -> C:\\Users\\Tony Xu\\workspace\\comfyui_models,
macOS/Linux workstation -> ~/comfyui_models.

Transfers require the AWS CLI (`aws`) on PATH — the only S3 client verified
to sign correctly against RunPod's endpoint.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lib.envfile import env_value  # noqa: E402

VOLUME_ROOT = "comfyui_models"
POD_WORKSPACE = Path("/workspace")
TRANSFER_PROGRESS_ARGS = ["--progress-multiline", "--progress-frequency", "1"]

DEFAULT_ENDPOINT = "https://s3api-eu-ro-1.runpod.io"
DEFAULT_REGION = "eu-ro-1"
DEFAULT_VOLUME_ID = "1qwuc4cm14"
DEFAULT_VOLUME_SIZE_GB = 100.0

WINDOWS_LOCAL_ROOT = r"C:\Users\Tony Xu\workspace\comfyui_models"
WINDOWS_AWS_CLI_PATHS = (
    r"C:\Program Files\Amazon\AWSCLIV2\aws.exe",
    r"C:\Program Files\Amazon\AWSCLI\bin\aws.exe",
)


def on_pod() -> bool:
    return sys.platform.startswith("linux") and POD_WORKSPACE.is_dir()


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        self.access_key = env_value("RUNPOD_S3_ACCESS_KEY_ID")
        self.secret_key = env_value("RUNPOD_S3_SECRET_ACCESS_KEY")
        self.endpoint = env_value("RUNPOD_S3_ENDPOINT") or DEFAULT_ENDPOINT
        self.region = env_value("RUNPOD_S3_REGION") or DEFAULT_REGION
        self.volume_id = env_value("RUNPOD_VOLUME_ID") or DEFAULT_VOLUME_ID
        try:
            self.volume_size_gb = float(env_value("RUNPOD_VOLUME_SIZE_GB") or DEFAULT_VOLUME_SIZE_GB)
        except ValueError:
            self.volume_size_gb = DEFAULT_VOLUME_SIZE_GB

        local_root = args.local_root or env_value("COMFYUI_MODELS_LOCAL_ROOT")
        if not local_root:
            if on_pod():
                local_root = str(POD_WORKSPACE / VOLUME_ROOT)
            elif os.name == "nt":
                local_root = WINDOWS_LOCAL_ROOT
            else:
                local_root = str(Path.home() / "comfyui_models")
        self.local_root = Path(local_root)

    def require_credentials(self) -> None:
        if not self.access_key or not self.secret_key:
            raise SystemExit(
                "error: RUNPOD_S3_ACCESS_KEY_ID / RUNPOD_S3_SECRET_ACCESS_KEY not set.\n"
                "Put them in the repo .env (see the volume's S3 API key in the RunPod console)."
            )

    def aws_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = self.access_key or ""
        env["AWS_SECRET_ACCESS_KEY"] = self.secret_key or ""
        env["AWS_PAGER"] = ""
        return env

    def s3_url(self, rel: str) -> str:
        rel = rel.strip("/")
        return f"s3://{self.volume_id}/{VOLUME_ROOT}/{rel}" if rel else f"s3://{self.volume_id}/{VOLUME_ROOT}"

    def endpoint_args(self) -> list[str]:
        return ["--endpoint-url", self.endpoint, "--region", self.region]


def aws_cli() -> str:
    path = shutil.which("aws")
    if path:
        return path
    if os.name == "nt":
        for candidate in WINDOWS_AWS_CLI_PATHS:
            if Path(candidate).is_file():
                return candidate
    raise SystemExit(
        "error: AWS CLI not found on PATH. Install it first "
        "(macOS: brew install awscli; Windows: winget install Amazon.AWSCLI)."
    )


def run_aws(cfg: Config, args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [aws_cli(), *args, *cfg.endpoint_args()]
    return subprocess.run(
        cmd, env=cfg.aws_env(), text=True,
        capture_output=capture, check=False,
    )


def normalize_rel(cfg: Config, raw: str) -> str:
    """User paths are relative to the shared root; tolerate leading slashes,
    backslashes, and absolute paths under the local root."""
    rel = raw.replace("\\", "/").strip()
    p = Path(raw)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(cfg.local_root.resolve()).as_posix()
        except ValueError:
            raise SystemExit(
                f"error: absolute path {raw} is not under the local root {cfg.local_root}"
            )
    return rel.strip("/")


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:,.1f} TB"


# --------------------------------------------------------------------------- #
# Upload / download (anywhere, S3)
# --------------------------------------------------------------------------- #

def upload(cfg: Config, raw_paths: list[str]) -> int:
    failures = 0
    for raw in raw_paths:
        rel = normalize_rel(cfg, raw)
        local = cfg.local_root / rel
        target = cfg.s3_url(rel)
        if local.is_dir():
            print(f">> upload dir  {local}  ->  {target}")
            result = run_aws(cfg, ["s3", "sync", str(local), target, *TRANSFER_PROGRESS_ARGS])
        elif local.is_file():
            print(f">> upload file {local}  ->  {target}")
            result = run_aws(cfg, ["s3", "cp", str(local), target, *TRANSFER_PROGRESS_ARGS])
        else:
            print(f"[FAIL] {local} does not exist locally")
            failures += 1
            continue
        if result.returncode != 0:
            print(f"[FAIL] upload {rel}")
            failures += 1
        else:
            print(f"[OK]   {rel}")
    return failures


def remote_kind(cfg: Config, rel: str) -> str:
    """'dir' if objects exist under rel/, 'file' if rel is an exact object, else 'missing'."""
    probe_dir = run_aws(cfg, ["s3", "ls", cfg.s3_url(rel) + "/"], capture=True)
    if probe_dir.returncode == 0 and probe_dir.stdout.strip():
        return "dir"
    probe_file = run_aws(cfg, ["s3", "ls", cfg.s3_url(rel)], capture=True)
    if probe_file.returncode == 0 and probe_file.stdout.strip():
        return "file"
    return "missing"


def download(cfg: Config, raw_paths: list[str]) -> int:
    failures = 0
    for raw in raw_paths:
        rel = normalize_rel(cfg, raw)
        local = cfg.local_root / rel
        source = cfg.s3_url(rel)
        kind = remote_kind(cfg, rel)
        if kind == "dir":
            print(f">> download dir  {source}  ->  {local}")
            local.mkdir(parents=True, exist_ok=True)
            result = run_aws(cfg, ["s3", "sync", source, str(local), *TRANSFER_PROGRESS_ARGS])
        elif kind == "file":
            print(f">> download file {source}  ->  {local}")
            local.parent.mkdir(parents=True, exist_ok=True)
            result = run_aws(cfg, ["s3", "cp", source, str(local), *TRANSFER_PROGRESS_ARGS])
        else:
            print(f"[FAIL] {VOLUME_ROOT}/{rel} does not exist on the volume")
            failures += 1
            continue
        if result.returncode != 0:
            print(f"[FAIL] download {rel}")
            failures += 1
        else:
            print(f"[OK]   {rel}")
    return failures


# --------------------------------------------------------------------------- #
# Probe (POD ONLY: real filesystem, du-based)
# --------------------------------------------------------------------------- #

def run_du(args: list[str]) -> str:
    result = subprocess.run(["du", *args], text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):  # 1 = permission warnings, output still usable
        raise SystemExit(f"error: du failed:\n{result.stderr.strip()}")
    return result.stdout


def du_map(target: Path, depth: int, mode_flag: str) -> dict[str, int]:
    """One du walk → {abs_path: value} for every dir AND file down to depth.
    mode_flag: --block-size=1 (bytes) or --inodes (file/dir counts)."""
    out = run_du(["-a", mode_flag, f"--max-depth={depth}", str(target)])
    result: dict[str, int] = {}
    for line in out.strip().splitlines():
        value_s, _, path = line.partition("\t")
        try:
            result[path.rstrip("/") or "/"] = int(value_s)
        except ValueError:
            continue
    return result


def probe(cfg: Config, subpath: str, depth: int) -> int:
    if not on_pod():
        raise SystemExit(
            "error: --probe runs ON A POD only.\n"
            "The S3 API has no folder sizes — remote probing means listing every\n"
            "object (minutes). On any pod this answers in seconds:\n"
            "    python /workspace/ComfyUI/scripts/runpod/model_sync.py --probe\n"
            "Upload/download (-u/-d) work from anywhere."
        )
    target = POD_WORKSPACE / subpath.strip("/") if subpath.strip("/") else POD_WORKSPACE
    if not target.exists():
        raise SystemExit(f"error: {target} does not exist")
    depth = max(1, depth)

    print(f"volume usage via du   scope: {target}   depth: {depth}")
    sizes = du_map(target, depth, "--block-size=1")
    counts = du_map(target, depth, "--inodes")

    base = str(target).rstrip("/") or "/"
    children: dict[str, list[str]] = {}
    for path in sizes:
        if path == base:
            continue
        parent = path.rsplit("/", 1)[0] or "/"
        children.setdefault(parent, []).append(path)

    def render(parent: str, indent: int) -> None:
        for child in sorted(children.get(parent, []), key=lambda p: -sizes.get(p, 0)):
            is_dir = child in children or Path(child).is_dir()
            name = child[len(parent):].lstrip("/") + ("/" if is_dir else "")
            pad = max(6, 42 - 2 * indent)
            suffix = f"   ({counts.get(child, 0):,} files)" if is_dir else ""
            print(f"  {'  ' * indent}{name:<{pad}} {human(sizes.get(child, 0)):>12}{suffix}")
            render(child, indent + 1)

    render(base, 0)
    total_bytes = sizes.get(base, 0)
    print(f"  {'TOTAL':<42} {human(total_bytes):>12}   ({counts.get(base, 0):,} files)")

    if target == POD_WORKSPACE:
        quota = cfg.volume_size_gb * 1024**3
        free = max(0.0, quota - total_bytes)
        pct = 100.0 * total_bytes / quota if quota else 0.0
        print(f"  {'quota (RUNPOD_VOLUME_SIZE_GB=' + str(int(cfg.volume_size_gb)) + ')':<42} {human(quota):>12}")
        print(f"  {'free':<42} {human(free):>12}   ({pct:.0f}% used)")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

EXAMPLES = """
examples:

  transfers (any machine — Mac / Windows / pod; S3 API, no pod required):
    model_sync.py -u checkpoints/flux2 loras/anna     upload folders (recursive, incremental)
    model_sync.py -u loras/anna_v1.safetensors        upload a single file
    model_sync.py -d vae checkpoints/flux2            download the other direction
      paths are relative to BOTH roots at once:
        local   <LOCAL_ROOT>/<path>          (Windows default: C:\\Users\\Tony Xu\\workspace\\comfyui_models)
        volume  comfyui_models/<path>        (pods: /workspace/comfyui_models/<path>)

  probe (POD ONLY — walks the real filesystem with du):
    model_sync.py --probe                             whole volume: tree + TOTAL/quota/free (~45s)
    model_sync.py --probe comfyui_models              breakdown of one dir (instant on model trees)
    model_sync.py --probe comfyui_models --depth 3    deeper nesting; folders AND files shown
    model_sync.py --probe ComfyUI/models --depth 1    any path under /workspace works
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync ComfyUI models between machines and the RunPod network volume (S3, anywhere), "
                    "and probe volume usage (du, pod only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("-u", "--upload", nargs="+", metavar="PATH",
                      help=f"Upload local <root>/PATH... to volume {VOLUME_ROOT}/PATH... "
                           "Any machine. Multiple paths allowed; dirs are recursive + incremental.")
    mode.add_argument("-d", "--download", nargs="+", metavar="PATH",
                      help=f"Download volume {VOLUME_ROOT}/PATH... to local <root>/PATH... "
                           "Any machine. Multiple paths allowed; dirs are recursive + incremental.")
    mode.add_argument("--probe", nargs="?", const="", metavar="SUBPATH",
                      help="POD ONLY. Storage breakdown (folders AND files, sizes, file counts) of "
                           "/workspace, or /workspace/SUBPATH if given. Whole volume adds TOTAL, "
                           "quota and free space and takes ~45s; subpaths are near-instant.")
    p.add_argument("--depth", type=int, default=2, metavar="N",
                   help="Probe only: nest the breakdown N levels deep (default: 2). "
                        "Deeper costs no extra walk time.")
    p.add_argument("--local-root", default=None, metavar="DIR",
                   help="Transfers only: override the local models root "
                        "(default: COMFYUI_MODELS_LOCAL_ROOT from .env, else the platform default).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config(args)

    if args.probe is not None:
        return probe(cfg, args.probe, args.depth)

    cfg.require_credentials()
    aws_cli()
    print(f"local root:  {cfg.local_root}")
    print(f"volume root: {VOLUME_ROOT}/  (volume {cfg.volume_id})")
    failures = upload(cfg, args.upload) if args.upload else download(cfg, args.download)
    if failures:
        print(f"\n{failures} path(s) failed")
        return 1
    print("\nall paths synced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
