#!/usr/bin/env python3
"""Sync ComfyUI models between machines and the RunPod network volume, and
probe volume usage.

Three modes:

  TRANSFER (runs anywhere — Mac/Windows/pod; S3 API, no pod required):
    python scripts/runpod/model_sync.py -u checkpoints/flux2 loras/anna
    python scripts/runpod/model_sync.py -d loras/anna_v1.safetensors vae

    Every path is retried until VERIFIED or attempts run out:
      --retries N        attempts per path (default: 5), backoff 10/30/60/120s
      verification       files: remote size must equal local size (head-object)
                         dirs:  re-synced until a pass transfers nothing
    Large files upload as multipart (64MB parts, 4 streams, SDK-level part
    retries: AWS_RETRY_MODE=adaptive, AWS_MAX_ATTEMPTS=10). Partial parts left
    by THIS tool's own failed attempts are cleaned between its retries; the
    tool NEVER aborts upload sessions it did not create.

  PROBE (runs ON A POD ONLY — real filesystem, du-based):

    # whole volume: tree of folders AND files with sizes + file counts,
    # then TOTAL / quota / free. Env dirs (ComfyUI/, ai-toolkit/) are shown
    # as totals only (contents managed by git), cached against the installer
    # stamp — warm runs answer in <1s; ~20s only right after installer changes.
    python /workspace/ComfyUI/scripts/runpod/model_sync.py --probe

    # any subpath under /workspace — instant on model/data trees
    python /workspace/ComfyUI/scripts/runpod/model_sync.py --probe comfyui_models

    # --depth N: nesting of the breakdown (default: 2; deeper costs nothing)
    python /workspace/ComfyUI/scripts/runpod/model_sync.py --probe comfyui_models/text_encoders --depth 3

    # --quick: one size number only (plus quota/free at root), no tree
    python /workspace/ComfyUI/scripts/runpod/model_sync.py --probe --quick

  MAINTENANCE — DESTRUCTIVE, read before use (runs anywhere; S3 API):

    # DRY RUN (default): list incomplete multipart uploads with their ages.
    # Aborts NOTHING without --yes.
    python scripts/runpod/model_sync.py --abort-stale-uploads

    # abort sessions older than --stale-age hours (default: 6.0)
    python scripts/runpod/model_sync.py --abort-stale-uploads --stale-age 6 --yes

    WARNING: aborting a session that ANY machine is actively uploading KILLS
    that upload mid-flight (the uploader gets NoSuchUpload and loses all
    progress). Only run with --yes when you are certain no upload is running
    anywhere. Incident 2026-07-11: a live 92%-complete 17GB upload was
    destroyed exactly this way.

Probing deliberately does NOT work from a workstation: the S3 API has no
aggregate sizes, so any remote probe degenerates into listing every object
(minutes). On a pod, `du` walks the mounted filesystem in seconds. RunPod
exposes no volume-usage API (verified: their networkvolumes endpoint returns
only id/name/size/dataCenterId).

Transfer path layout is mirrored on both sides:

    local    <LOCAL_ROOT>/<path>
    volume   comfyui_models/<path>     (pods see /workspace/comfyui_models/<path>)

Configuration (repo .env, auto-loaded; only needed for transfers):
    S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY   required
    S3_ENDPOINT / S3_REGION / VOLUME_ID  preset defaults
    VOLUME_SIZE_GB   quota used by --probe free-space math
    COMFYUI_MODELS_LOCAL_ROOT  overrides the local root

LOCAL_ROOT default: Windows -> C:\\Users\\Tony Xu\\workspace\\comfyui_models,
macOS/Linux workstation -> ~/comfyui_models.

Transfers require the AWS CLI (`aws`) on PATH — the only S3 client verified
to sign correctly against RunPod's endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
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
        self.access_key = env_value("S3_ACCESS_KEY_ID")
        self.secret_key = env_value("S3_SECRET_ACCESS_KEY")
        self.endpoint = env_value("S3_ENDPOINT") or DEFAULT_ENDPOINT
        self.region = env_value("S3_REGION") or DEFAULT_REGION
        self.volume_id = env_value("VOLUME_ID") or DEFAULT_VOLUME_ID
        try:
            self.volume_size_gb = float(env_value("VOLUME_SIZE_GB") or DEFAULT_VOLUME_SIZE_GB)
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
                "error: S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY not set.\n"
                "Put them in the repo .env (see the volume's S3 API key in the RunPod console)."
            )

    def aws_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = self.access_key or ""
        env["AWS_SECRET_ACCESS_KEY"] = self.secret_key or ""
        env["AWS_PAGER"] = ""
        # Reliability for large multipart transfers (RunPod's own guidance):
        # retry dropped parts aggressively at the SDK level instead of failing
        # the whole 20 GB upload at completion time.
        env["AWS_RETRY_MODE"] = "adaptive"
        env["AWS_MAX_ATTEMPTS"] = "10"
        env["AWS_CONFIG_FILE"] = str(_aws_config_file())
        return env

    def s3_url(self, rel: str) -> str:
        rel = rel.strip("/")
        return f"s3://{self.volume_id}/{VOLUME_ROOT}/{rel}" if rel else f"s3://{self.volume_id}/{VOLUME_ROOT}"

    def endpoint_args(self) -> list[str]:
        return ["--endpoint-url", self.endpoint, "--region", self.region]


POD_AWS_CLI = POD_WORKSPACE / "bin" / "aws"  # volume-persistent install (bootstrap step 9)


def aws_cli() -> str:
    path = shutil.which("aws")
    if path:
        return path
    if on_pod() and POD_AWS_CLI.is_file():
        return str(POD_AWS_CLI)
    if os.name == "nt":
        for candidate in WINDOWS_AWS_CLI_PATHS:
            if Path(candidate).is_file():
                return candidate
    raise SystemExit(
        "error: AWS CLI not found.\n"
        "  macOS:   brew install awscli\n"
        "  Windows: winget install Amazon.AWSCLI\n"
        "  pod:     install once onto the VOLUME (nothing on the pod itself):\n"
        "           re-run scripts/runpod/bootstrap_pod.sh, or the aws-cli block "
        "from its step 9 — installs to /workspace/bin + /workspace/aws-cli."
    )


def run_aws(cfg: Config, args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [aws_cli(), *args, *cfg.endpoint_args()]
    return subprocess.run(
        cmd, env=cfg.aws_env(), text=True,
        capture_output=capture, check=False,
    )


def _aws_config_file() -> Path:
    """AWS CLI config with multipart tuning: 64MB parts (a 20GB file becomes
    ~320 parts instead of ~2500 — far fewer chances to lose one) and lower
    concurrency (kinder to home uplinks)."""
    cfg_dir = Path.home() / ".cache" / "model_sync"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "aws_config"
    content = (
        "[default]\n"
        "s3 =\n"
        "    multipart_threshold = 64MB\n"
        "    multipart_chunksize = 64MB\n"
        "    max_concurrent_requests = 4\n"
    )
    try:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
    except OSError:
        pass
    return path


def remote_file_size(cfg: Config, rel: str) -> int | None:
    result = run_aws(
        cfg,
        ["s3api", "head-object", "--bucket", cfg.volume_id,
         "--key", f"{VOLUME_ROOT}/{rel}", "--query", "ContentLength", "--output", "text"],
        capture=True,
    )
    try:
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def incomplete_uploads(cfg: Config, prefix: str | None = None) -> list[dict]:
    """List incomplete multipart upload sessions (the .s3compat_uploads
    entries). Read-only."""
    args = ["s3api", "list-multipart-uploads", "--bucket", cfg.volume_id, "--output", "json"]
    if prefix:
        args += ["--prefix", prefix]
    result = run_aws(cfg, args, capture=True)
    if result.returncode != 0:
        return []
    return (json.loads(result.stdout or "{}") or {}).get("Uploads") or []


def abort_stale_uploads(cfg: Config, prefix: str | None, min_age_hours: float,
                        quiet: bool = False) -> int:
    """DESTRUCTIVE: abort incomplete multipart uploads. Aborting a session
    another machine is actively feeding KILLS that upload (NoSuchUpload).
    Internal transfer retries call this ONLY on their own key after their own
    failed attempt; the standalone CLI mode requires an explicit --yes."""
    uploads = incomplete_uploads(cfg, prefix)
    now = datetime.now(timezone.utc)
    aborted = 0
    for up in uploads:
        if not prefix:
            try:
                initiated = datetime.fromisoformat(up["Initiated"].replace("Z", "+00:00"))
                if (now - initiated).total_seconds() < min_age_hours * 3600:
                    continue
            except (KeyError, ValueError):
                continue
        rc = run_aws(
            cfg,
            ["s3api", "abort-multipart-upload", "--bucket", cfg.volume_id,
             "--key", up["Key"], "--upload-id", up["UploadId"]],
            capture=True,
        )
        if rc.returncode == 0:
            aborted += 1
            if not quiet:
                print(f"[CLEANED] aborted stale upload: {up['Key']}")
    return aborted


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

RETRY_BACKOFF_SECONDS = (10, 30, 60, 120)


def _verify_upload(cfg: Config, rel: str, local: Path, target: str) -> bool:
    """File: remote size must equal local size. Dir: a second sync pass must
    transfer nothing (converged)."""
    if local.is_file():
        remote = remote_file_size(cfg, rel)
        if remote == local.stat().st_size:
            return True
        print(f"[VERIFY-FAIL] {rel}: remote size {remote} != local {local.stat().st_size}")
        return False
    check = run_aws(cfg, ["s3", "sync", str(local), target], capture=True)
    if check.returncode == 0 and not check.stdout.strip():
        return True
    print(f"[VERIFY-FAIL] {rel}: sync had not converged, re-syncing")
    return False


def upload(cfg: Config, raw_paths: list[str], attempts: int) -> int:
    failures = 0
    for raw in raw_paths:
        rel = normalize_rel(cfg, raw)
        local = cfg.local_root / rel
        target = cfg.s3_url(rel)
        if not local.exists():
            print(f"[FAIL] {local} does not exist locally")
            failures += 1
            continue
        kind = "dir" if local.is_dir() else "file"
        print(f">> upload {kind}  {local}  ->  {target}")
        # NOTE deliberately NO cleanup before the first attempt: another
        # machine may be mid-upload on this key, and aborting a session we
        # did not create kills that transfer (incident 2026-07-11). Cleanup
        # runs only after OUR OWN attempts fail.
        ok = False
        for attempt in range(1, attempts + 1):
            if local.is_dir():
                result = run_aws(cfg, ["s3", "sync", str(local), target, *TRANSFER_PROGRESS_ARGS])
            else:
                result = run_aws(cfg, ["s3", "cp", str(local), target, *TRANSFER_PROGRESS_ARGS])
            if result.returncode == 0 and _verify_upload(cfg, rel, local, target):
                ok = True
                break
            if attempt < attempts:
                delay = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
                print(f"[RETRY] {rel}: attempt {attempt}/{attempts} failed — "
                      f"cleaning partial parts, retrying in {delay}s")
                abort_stale_uploads(cfg, prefix=f"{VOLUME_ROOT}/{rel}", min_age_hours=0, quiet=True)
                time.sleep(delay)
        if ok:
            print(f"[OK]   {rel} (verified)")
        else:
            print(f"[FAIL] upload {rel} after {attempts} attempts")
            abort_stale_uploads(cfg, prefix=f"{VOLUME_ROOT}/{rel}", min_age_hours=0, quiet=True)
            failures += 1
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


def _verify_download(cfg: Config, rel: str, local: Path, source: str, kind: str) -> bool:
    if kind == "file":
        remote = remote_file_size(cfg, rel)
        if local.is_file() and remote == local.stat().st_size:
            return True
        print(f"[VERIFY-FAIL] {rel}: local size mismatch vs remote {remote}")
        return False
    check = run_aws(cfg, ["s3", "sync", source, str(local)], capture=True)
    if check.returncode == 0 and not check.stdout.strip():
        return True
    print(f"[VERIFY-FAIL] {rel}: sync had not converged, re-syncing")
    return False


def download(cfg: Config, raw_paths: list[str], attempts: int) -> int:
    failures = 0
    for raw in raw_paths:
        rel = normalize_rel(cfg, raw)
        local = cfg.local_root / rel
        source = cfg.s3_url(rel)
        kind = remote_kind(cfg, rel)
        if kind == "missing":
            print(f"[FAIL] {VOLUME_ROOT}/{rel} does not exist on the volume")
            failures += 1
            continue
        print(f">> download {kind}  {source}  ->  {local}")
        ok = False
        for attempt in range(1, attempts + 1):
            if kind == "dir":
                local.mkdir(parents=True, exist_ok=True)
                result = run_aws(cfg, ["s3", "sync", source, str(local), *TRANSFER_PROGRESS_ARGS])
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                result = run_aws(cfg, ["s3", "cp", source, str(local), *TRANSFER_PROGRESS_ARGS])
            if result.returncode == 0 and _verify_download(cfg, rel, local, source, kind):
                ok = True
                break
            if attempt < attempts:
                delay = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
                print(f"[RETRY] {rel}: attempt {attempt}/{attempts} failed — retrying in {delay}s")
                time.sleep(delay)
        if ok:
            print(f"[OK]   {rel} (verified)")
        else:
            print(f"[FAIL] download {rel} after {attempts} attempts")
            failures += 1
    return failures


# --------------------------------------------------------------------------- #
# Probe (POD ONLY: real filesystem, du-based)
# --------------------------------------------------------------------------- #

def run_du(args: list[str]) -> str:
    result = subprocess.run(["du", *args], text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):  # 1 = permission warnings, output still usable
        raise SystemExit(f"error: du failed:\n{result.stderr.strip()}")
    return result.stdout


def du_map(target: Path, depth: int, mode_flag: str,
           excludes: tuple[Path, ...] = ()) -> dict[str, int]:
    """One du walk → {abs_path: value} for every dir AND file down to depth.
    mode_flag: --block-size=1 (bytes) or --inodes (file/dir counts)."""
    args = ["-a", mode_flag, f"--max-depth={depth}"]
    args += [f"--exclude={p}" for p in excludes]
    out = run_du([*args, str(target)])
    result: dict[str, int] = {}
    for line in out.strip().splitlines():
        value_s, _, path = line.partition("\t")
        try:
            result[path.rstrip("/") or "/"] = int(value_s)
        except ValueError:
            continue
    return result


QUICK_CACHE = POD_WORKSPACE / ".quick_size_cache.json"
# The two venv-bearing monsters (~120k files each walk). Their contents only
# change when the manifest installer runs, and every installer run rewrites
# its state stamp — so their sizes are cached keyed to that stamp and reused
# while the stamp is unchanged. Everything else is walked live (few files).
ENV_DIRS = ("ComfyUI", "ai-toolkit")
INSTALL_STAMP = POD_WORKSPACE / "ComfyUI" / "custom_nodes" / ".install_state.json"


def du_total(path: Path) -> int:
    return int(run_du(["-s", "--block-size=1", str(path)]).split("\t", 1)[0])


def _stamp_key() -> str:
    import hashlib
    try:
        return hashlib.md5(INSTALL_STAMP.read_bytes()).hexdigest()
    except OSError:
        return "no-stamp"


def env_dir_sizes() -> tuple[dict[str, int], str]:
    """Totals for the env dirs, cached keyed to the installer stamp: their
    contents only change when the installer runs, so the cache is provably
    fresh while the stamp is unchanged."""
    key = _stamp_key()
    try:
        cache = json.loads(QUICK_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache = {}
    if cache.get("key") == key:
        return cache["sizes"], "cached; installer stamp unchanged"
    sizes = {d: du_total(POD_WORKSPACE / d) for d in ENV_DIRS if (POD_WORKSPACE / d).exists()}
    QUICK_CACHE.write_text(json.dumps({"key": key, "sizes": sizes}), encoding="utf-8")
    return sizes, "walked fresh; cache (re)seeded"


def quick(cfg: Config, subpath: str) -> int:
    """Size only, no counts, no tree. Subfolders: one live du (near-instant).
    Whole volume: live du on everything except the env dirs (stamp-cached)."""
    if not on_pod():
        raise SystemExit("error: --probe --quick runs ON A POD only (same reason as --probe).")
    target = POD_WORKSPACE / subpath.strip("/") if subpath.strip("/") else POD_WORKSPACE
    if not target.exists():
        raise SystemExit(f"error: {target} does not exist")

    if target != POD_WORKSPACE:
        print(f"{target}: {human(du_total(target))}")
        return 0

    total = 0
    for entry in os.scandir(POD_WORKSPACE):
        if entry.name in ENV_DIRS:
            continue
        total += entry.stat().st_size if entry.is_file() else du_total(Path(entry.path))
    env_sizes, env_note = env_dir_sizes()
    total += sum(env_sizes.values())

    quota = cfg.volume_size_gb * 1024**3
    free = max(0.0, quota - total)
    pct = 100.0 * total / quota if quota else 0.0
    print(f"/workspace: {human(total)}   (env dirs {env_note})")
    print(f"quota: {human(quota)}   free: {human(free)}   ({pct:.0f}% used)")
    return 0


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

    # Env dirs (ComfyUI/, ai-toolkit/) are total-only by design: their
    # contents are managed by git + installer, so the file-by-file walk is
    # skipped and their totals come from the stamp-keyed cache.
    excludes = tuple(POD_WORKSPACE / d for d in ENV_DIRS) if target == POD_WORKSPACE else ()

    print(f"volume usage via du   scope: {target}   depth: {depth}")
    sizes = du_map(target, depth, "--block-size=1", excludes)
    counts = du_map(target, depth, "--inodes", excludes)

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

    if target == POD_WORKSPACE:
        env_sizes, env_note = env_dir_sizes()
        for name, b in sorted(env_sizes.items(), key=lambda kv: -kv[1]):
            print(f"  {name + '/':<42} {human(b):>12}   (total only — contents managed by git; {env_note})")
        total_bytes += sum(env_sizes.values())
        print(f"  {'TOTAL':<42} {human(total_bytes):>12}   ({counts.get(base, 0):,} files excl. env dirs)")
        quota = cfg.volume_size_gb * 1024**3
        free = max(0.0, quota - total_bytes)
        pct = 100.0 * total_bytes / quota if quota else 0.0
        print(f"  {'quota (VOLUME_SIZE_GB=' + str(int(cfg.volume_size_gb)) + ')':<42} {human(quota):>12}")
        print(f"  {'free':<42} {human(free):>12}   ({pct:.0f}% used)")
    else:
        print(f"  {'TOTAL':<42} {human(total_bytes):>12}   ({counts.get(base, 0):,} files)")
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
    model_sync.py --probe --quick                     size + quota/free only, no tree
    model_sync.py --probe comfyui_models --quick      size of one folder (near-instant)

  reliability (large files / flaky networks):
    every transfer is verified (remote size == local size; dirs re-synced to
    convergence) and retried up to --retries times with backoff; the tool
    cleans only its OWN failed sessions between retries.

  maintenance (DESTRUCTIVE with --yes — can kill a live upload):
    model_sync.py --abort-stale-uploads               DRY RUN: list incomplete uploads + ages
    model_sync.py --abort-stale-uploads --yes         abort those older than --stale-age (6h)
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
    mode.add_argument("--abort-stale-uploads", action="store_true",
                      help="List incomplete multipart uploads (DRY RUN by default — aborts "
                           "nothing). With --yes: abort those older than --stale-age hours and "
                           "free their space. DESTRUCTIVE with --yes: kills any live upload it "
                           "catches. Any machine.")
    p.add_argument("--yes", action="store_true",
                   help="Required for --abort-stale-uploads to actually abort anything "
                        "(without it: dry-run listing only).")
    p.add_argument("--retries", type=int, default=5, metavar="N",
                   help="Transfers: attempts per path until verified (default: 5); backoff "
                        "10/30/60/120s. Cleanup between attempts touches only this tool's own "
                        "failed sessions.")
    p.add_argument("--stale-age", type=float, default=6.0, metavar="HOURS",
                   help="--abort-stale-uploads --yes: only abort uploads older than this "
                        "(default: 6.0h). Anything younger is treated as possibly live.")
    p.add_argument("--quick", action="store_true",
                   help="With --probe: size only — no counts, no tree. Single du pass: "
                        "root ~20s, subfolders near-instant. Root scope still shows quota/free.")
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

    if args.quick and args.probe is None:
        raise SystemExit("error: --quick is a modifier of --probe (use: --probe [SUBPATH] --quick)")
    if args.probe is not None:
        if args.quick:
            return quick(cfg, args.probe)
        return probe(cfg, args.probe, args.depth)

    cfg.require_credentials()
    aws_cli()
    if args.abort_stale_uploads:
        uploads = incomplete_uploads(cfg)
        if not uploads:
            print("no incomplete multipart uploads on the volume")
            return 0
        now = datetime.now(timezone.utc)
        for up in uploads:
            try:
                started = datetime.fromisoformat(up["Initiated"].replace("Z", "+00:00"))
                age = f"{(now - started).total_seconds() / 3600:.1f}h"
            except (KeyError, ValueError):
                age = "?"
            print(f"  {up['Key']}   started {up.get('Initiated', '?')}   age {age}")
        if not args.yes:
            print(f"\nDRY RUN — nothing aborted. A session may be a LIVE upload from another "
                  f"machine.\nAdd --yes to abort sessions older than {args.stale_age}h.")
            return 0
        n = abort_stale_uploads(cfg, prefix=None, min_age_hours=args.stale_age)
        print(f"{n} upload(s) aborted (older than {args.stale_age}h)")
        return 0
    print(f"local root:  {cfg.local_root}")
    print(f"volume root: {VOLUME_ROOT}/  (volume {cfg.volume_id})")
    attempts = max(1, args.retries)
    failures = (upload(cfg, args.upload, attempts) if args.upload
                else download(cfg, args.download, attempts))
    if failures:
        print(f"\n{failures} path(s) failed")
        return 1
    print("\nall paths synced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
