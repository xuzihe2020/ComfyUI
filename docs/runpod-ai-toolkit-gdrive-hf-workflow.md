# RunPod AI-Toolkit Workflow With Google Drive And Hugging Face

This runbook describes a repeatable workflow for training Flux LoRAs with
AI-Toolkit on RunPod while keeping the RunPod disk small. The intended storage
split is:

- RunPod: active working cache only.
- Google Drive: image/video inputs, training outputs, sample review, deletion.
- Hugging Face: base model references, final LoRAs, clean datasets, model cards.
- GitHub: AI-Toolkit fork plus reusable helper scripts/config templates.

The fork created for this workflow is:

```text
https://github.com/xuzihe2020/ai-toolkit
```

## 1. Architecture

```text
Local machine
  - browser for RunPod, Google Drive, Hugging Face
  - SSH client
  - optional local rclone config/bootstrap

RunPod pod
  /workspace/fluxlab/
    src/ai-toolkit/       cloned fork
    datasets/             current dataset only
    runs/                 current run outputs
    models/               current base model and LoRAs only
    configs/              training configs
    cache/                HF/cache/tmp
    scripts/              Swiss-knife tools
    .secrets/             tokens and rclone config; local runtime copy

Google Drive
  FluxLab/
    datasets/
    runs/
    review/
    inbox/
    bootstrap/
      secrets/             optional convenience backup for non-critical keys

Hugging Face
  xuzihe2020/<character>-lora
  xuzihe2020/<clean-dataset>
  model references and private repos as needed
```

The RunPod disk should not be treated as the archive. Keep only the dataset,
model, and outputs required for the current run.

## 2. Source Of Truth Rules

Use one source of truth per artifact class:

```text
Google Drive = images, videos, raw datasets, sample outputs for browser review
Hugging Face = model artifacts, final LoRAs, clean ML datasets
RunPod = temporary working cache
GitHub = code, scripts, docs, config templates
```

Use matching logical names across RunPod and Google Drive:

```text
datasets/<dataset_slug>
runs/<run_id>
review/<run_id>
```

Default strict mode avoids syncing these directories to Google Drive or Hugging
Face:

```text
.secrets/
cache/
venv/
node_modules/
__pycache__/
models/base/ unless explicitly intended
```

Convenience mode is also supported. If the API keys are low-risk for your use
case and easy to rotate, you can keep a backup copy in a private Google Drive
folder or a private GitHub repository and restore them onto the pod. In that
mode, sync secrets explicitly with `fluxlab pull-secrets` and
`fluxlab push-secrets` rather than including secrets in every run backup.

Prefer `rclone copy` for routine movement. Use `rclone sync` only after a
`--dry-run` and only when the destination should exactly match the source.

## 3. RunPod Mode

Use RunPod Pods, not Serverless.

Pods are the right fit because this workflow needs:

- SSH access.
- A browser WebUI on port 8675.
- Long-running training jobs.
- A persistent `/workspace`.
- Manual inspection and debugging.
- Easy use with `rclone`, `huggingface-cli`, and custom shell tools.

Serverless is better for packaged request/response inference or batch jobs. It
is not the right starting point for interactive AI-Toolkit training.

## 4. GPU Selection

For Flux.2 Klein 4B character LoRA:

```text
Best speed/value: RTX 4090 or RTX 5090
Best 48GB comfort: RTX A6000, A40, RTX 6000 Ada, L40S
Only when needed: A100 80GB
```

Suggested choices:

- Start on local RTX 4090 to validate dataset and config.
- Use RunPod RTX 5090 or RTX 4090 for normal 24GB/32GB runs.
- Use A6000/A40/RTX 6000 Ada 48GB if you hit VRAM limits.
- Use A100 80GB only for larger model variants or aggressive settings.

## 4.1. Starting Cheap And Upgrading GPUs Later

Do not assume a running Pod can hot-swap to a different GPU. Treat GPU changes
as a redeploy/migration operation.

RunPod lets you stop a Pod, which releases the GPU and preserves data stored in
`/workspace` on the Pod volume. The container/OS disk is cleared when stopped.
When using a network volume, `/workspace` data is preserved even if the Pod is
terminated.

For this workflow, the safest cheap-start strategy is:

```text
1. Create a network volume.
2. Deploy a cheap Pod with that network volume attached.
3. Install AI-Toolkit, rclone, Hugging Face auth, and helper scripts under /workspace.
4. Stop or terminate the cheap Pod when done.
5. Deploy a new Pod with a stronger GPU and attach the same network volume.
6. Continue from the same /workspace.
```

Important constraints:

- Network volumes must be selected when deploying the Pod. They cannot be
  attached to an already-created Pod without deleting/redeploying the Pod.
- Network volumes are only available for Pods in RunPod Secure Cloud.
- The available GPU choices depend on the network volume's data center.
- A stopped Pod releases its GPU. Restarting later may fail if that exact host
  no longer has capacity.
- If you use only the default Pod volume, data is easier to preserve across
  stop/start than across GPU changes or termination.

For maximum portability, keep the canonical data in Google Drive and Hugging
Face, and keep `/workspace` as a rebuildable working environment.

## 5. RunPod Pod Creation

In RunPod:

1. Go to `Pods` -> `Deploy a Pod`.
2. Choose a PyTorch/CUDA template or an AI-Toolkit template if available.
3. Select a GPU.
4. Enable SSH terminal access.
5. Add your public SSH key.
6. Expose HTTP port `8675` for the AI-Toolkit UI.
7. Use `/workspace` as the persistent working area.
8. Keep the disk small if desired, but understand that 80GB is only a working
   cache.

Recommended active cache estimate:

```text
AI-Toolkit repo + venv + node deps: 10-25GB
Current base model/cache: 15-45GB
Current 30-image dataset: usually small
Training outputs/checkpoints: 5-30GB depending save frequency
Safety margin: 10GB+
```

If disk space is tight, reduce checkpoint retention and push outputs frequently.

## 5.1. Using A RunPod Volume For Models

Yes, a RunPod network volume can hold models and datasets so they do not live on
the small container disk. When a network volume is attached to a Pod, it replaces
the normal volume disk at `/workspace`. Put model files under `/workspace`, for
example:

```text
/workspace/fluxlab/models/base/
/workspace/fluxlab/models/loras/
/workspace/fluxlab/datasets/
```

AI-Toolkit can load model paths directly from those directories. You do not need
to copy the models into the container/OS disk.

One network volume can be reused by different Pods in the same data center, so
this is a good fit for budget splitting: use an A100 Pod for LoRA training, an
RTX 5090 Pod for production generation, and a cheaper 4090/3090 Pod for small
tasks, all reading the same model store. Create the volume first, then deploy
each Pod with that volume attached. Avoid running multiple Pods that write into
the same output/cache directory at the same time; keep task-specific outputs
under separate folders such as `/workspace/fluxlab/runs/<run_id>` and
`/workspace/fluxlab/prod/output/<batch_id>`.

Speed expectations:

- Network volume reads are usually fast enough for model loading and training
  setup.
- RunPod documents standard network volumes as high-performance NVMe-backed
  storage with typical transfer speeds around 200-400 MB/s, with higher peak
  throughput.
- High-performance network volumes cost more and are intended for workloads
  where storage throughput/IOPS matters.
- GPU training itself usually runs from VRAM after model load. Dataset/model
  loading can still affect startup, caching, and sample/checkpoint IO.

For Flux LoRA, the practical layout is:

```text
/workspace/fluxlab/models/base/       persistent model cache on network volume
/workspace/fluxlab/models/loras/      LoRA outputs you want to reuse
/workspace/fluxlab/datasets/          current or frequently reused datasets
/workspace/fluxlab/runs/              active run outputs, synced to Google Drive
/tmp or container disk                throwaway temp only
```

Pricing from RunPod docs:

```text
Network volume, standard tier:
  First 1 TB: $0.07/GB/month
  Above 1 TB: $0.05/GB/month

Network volume, high-performance tier:
  $0.14/GB/month

Container disk:
  $0.10/GB/month while running
  not charged while stopped
  erased when Pod stops

Pod volume disk:
  $0.10/GB/month while running
  $0.20/GB/month while stopped
  retained until Pod is deleted
```

Approximate standard network volume monthly cost:

```text
100 GB:  $7/month
250 GB:  $17.50/month
500 GB:  $35/month
1 TB:    $70/month
2 TB:    about $120/month
```

RunPod warns that it is not intended as long-term cloud storage. Use the network
volume as a fast reusable workspace/cache, and keep important outputs backed up
to Google Drive or Hugging Face.

## 6. SSH Setup

Create or reuse an SSH key locally:

```bash
ssh-keygen -t ed25519 -C "runpod-ai-toolkit"
cat ~/.ssh/id_ed25519.pub
```

Paste the public key into RunPod when deploying the pod.

Optional local SSH config:

```sshconfig
Host runpod-ai-toolkit
  HostName <runpod-host>
  Port <runpod-ssh-port>
  User root
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 30
  ServerAliveCountMax 6
  LocalForward 8675 127.0.0.1:8675
```

Then connect:

```bash
ssh runpod-ai-toolkit
```

With the local forward, open the UI at:

```text
http://localhost:8675
```

If you expose port 8675 directly through RunPod, secure it with
`AI_TOOLKIT_AUTH`.

## 7. Base Directory Layout

On the pod:

```bash
mkdir -p /workspace/fluxlab/{src,datasets,runs,models/base,models/loras,configs,cache,scripts,.secrets}
chmod 700 /workspace/fluxlab/.secrets
```

Recommended environment file:

```bash
cat > /workspace/fluxlab/scripts/env.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

export FLUXLAB_ROOT="${FLUXLAB_ROOT:-/workspace/fluxlab}"
export SECRETS_DIR="${SECRETS_DIR:-$FLUXLAB_ROOT/.secrets}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-$SECRETS_DIR/rclone.conf}"
export GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive:FluxLab}"
export HF_HOME="${HF_HOME:-$FLUXLAB_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

if [ -f "$SECRETS_DIR/hf_token" ]; then
  export HF_TOKEN="$(cat "$SECRETS_DIR/hf_token")"
fi
EOF

chmod 700 /workspace/fluxlab/scripts/env.sh
```

## 8. Install System Tools

On the pod:

```bash
apt-get update
apt-get install -y git git-lfs rclone tmux htop nvtop unzip rsync curl ca-certificates
git lfs install
```

If Node.js is missing or too old:

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
node --version
npm --version
```

Install Hugging Face CLI inside the Python environment later, or globally:

```bash
python3 -m pip install -U huggingface_hub
```

## 9. Clone Your AI-Toolkit Fork

```bash
cd /workspace/fluxlab/src
git clone --recursive https://github.com/xuzihe2020/ai-toolkit.git
cd ai-toolkit
git remote -v
```

If you want to push changes from the pod, authenticate with GitHub:

```bash
gh auth login
```

If `gh` is not installed:

```bash
type -p curl >/dev/null || apt-get install -y curl
mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | tee /etc/apt/sources.list.d/github-cli.list >/dev/null
apt-get update
apt-get install -y gh
```

## 10. Install AI-Toolkit

From the fork:

```bash
cd /workspace/fluxlab/src/ai-toolkit
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the template does not already have a compatible PyTorch/CUDA build, install a
matching PyTorch wheel before `requirements.txt`. Check the template CUDA
version first:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

AI-Toolkit changes quickly, so prefer the install instructions in the upstream
README when they differ from this runbook.

## 11. Configure Hugging Face

Create a Hugging Face token with enough permission for private model/dataset
repos.

On the pod:

```bash
read -rsp "HF token: " HF_TOKEN_VALUE
printf "\n"
printf "%s" "$HF_TOKEN_VALUE" > /workspace/fluxlab/.secrets/hf_token
chmod 600 /workspace/fluxlab/.secrets/hf_token
unset HF_TOKEN_VALUE

source /workspace/fluxlab/scripts/env.sh
huggingface-cli login --token "$HF_TOKEN"
```

Use Hugging Face for:

- final LoRA `.safetensors`
- model cards
- clean datasets
- reusable base-model references when appropriate

Example download:

```bash
source /workspace/fluxlab/scripts/env.sh
huggingface-cli download black-forest-labs/FLUX.2-Klein-dev \
  --local-dir "$FLUXLAB_ROOT/models/base/flux2-klein-dev"
```

Example upload:

```bash
source /workspace/fluxlab/scripts/env.sh
huggingface-cli upload xuzihe2020/my-character-lora \
  "$FLUXLAB_ROOT/runs/my_character_flux2k_001/final" \
  --repo-type model
```

## 12. Configure Google Drive With rclone

Google Drive is the preferred review surface for images/videos because browser
preview and delete workflows are good.

Recommended Google Drive layout:

```text
My Drive/
  FluxLab/
    datasets/
      char_alice_v001/
        images/
        captions/
        metadata.json
    runs/
      char_alice_flux2k_001/
        samples/
        checkpoints/
        logs/
        eval/
    review/
    inbox/
```

### Option A: Configure rclone directly on the pod

```bash
rclone config
```

Create a remote named `gdrive` using the Google Drive backend. If rclone asks
for browser authorization, follow the URL from your local browser and paste the
result back into the SSH session.

Then move the generated config into the secret directory:

```bash
mkdir -p /workspace/fluxlab/.secrets
cp ~/.config/rclone/rclone.conf /workspace/fluxlab/.secrets/rclone.conf
chmod 600 /workspace/fluxlab/.secrets/rclone.conf
```

### Option B: Configure rclone locally and upload the config

On your local machine:

```bash
rclone config
rclone lsd gdrive:
scp ~/.config/rclone/rclone.conf runpod-ai-toolkit:/workspace/fluxlab/.secrets/rclone.conf
```

On the pod:

```bash
chmod 600 /workspace/fluxlab/.secrets/rclone.conf
source /workspace/fluxlab/scripts/env.sh
rclone lsd "$GDRIVE_REMOTE"
```

## 13. Swiss-Knife Script

Create a single command wrapper on the pod:

```bash
cat > /workspace/fluxlab/scripts/fluxlab <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

usage() {
  cat <<USAGE
Usage:
  fluxlab doctor
  fluxlab disk
  fluxlab pull-secrets
  fluxlab push-secrets
  fluxlab pull-dataset <dataset_slug>
  fluxlab push-run <run_id>
  fluxlab sync-samples <run_id>
  fluxlab backup-loop <run_id> [seconds]
  fluxlab pull-hf <repo_id> <local_subdir> [repo_type]
  fluxlab push-hf-model <repo_id> <local_dir>
  fluxlab start-ui
  fluxlab start-comfy
  fluxlab prod-pull <batch_id>
  fluxlab prod-run <manifest_jsonl> <batch_id>
  fluxlab prod-push <batch_id>
  fluxlab prod-run-and-stop <manifest_jsonl> <batch_id>
  fluxlab stop-pod
  fluxlab clean-cache

Examples:
  fluxlab pull-dataset char_alice_v001
  fluxlab push-run char_alice_flux2k_001
  fluxlab backup-loop char_alice_flux2k_001 300
  fluxlab pull-hf black-forest-labs/FLUX.2-Klein-dev models/base/flux2-klein-dev model
  fluxlab push-hf-model xuzihe2020/char-alice-lora runs/char_alice_flux2k_001/final
  fluxlab prod-run-and-stop /workspace/fluxlab/prod/batches/night_001/manifest.jsonl night_001
USAGE
}

need_arg() {
  local value="${1:-}"
  local name="$2"
  if [ -z "$value" ]; then
    echo "Missing argument: $name" >&2
    usage
    exit 2
  fi
}

cmd="${1:-}"
shift || true

case "$cmd" in
  doctor)
    echo "FLUXLAB_ROOT=$FLUXLAB_ROOT"
    echo "SECRETS_DIR=$SECRETS_DIR"
    echo "RCLONE_CONFIG=$RCLONE_CONFIG"
    echo "GDRIVE_REMOTE=$GDRIVE_REMOTE"
    echo "HF_HOME=$HF_HOME"
    echo
    command -v rclone
    command -v huggingface-cli
    command -v git
    command -v python3
    nvidia-smi || true
    echo
    rclone lsd "$GDRIVE_REMOTE" | head || true
    ;;

  disk)
    df -h "$FLUXLAB_ROOT" /workspace || true
    du -sh "$FLUXLAB_ROOT"/* 2>/dev/null || true
    ;;

  pull-secrets)
    mkdir -p "$SECRETS_DIR"
    rclone copy "$GDRIVE_REMOTE/bootstrap/secrets" "$SECRETS_DIR" -P
    chmod 700 "$SECRETS_DIR"
    chmod 600 "$SECRETS_DIR"/* 2>/dev/null || true
    ;;

  push-secrets)
    mkdir -p "$SECRETS_DIR"
    rclone copy "$SECRETS_DIR" "$GDRIVE_REMOTE/bootstrap/secrets" -P
    ;;

  pull-dataset)
    dataset="${1:-}"
    need_arg "$dataset" "dataset_slug"
    mkdir -p "$FLUXLAB_ROOT/datasets/$dataset"
    rclone copy "$GDRIVE_REMOTE/datasets/$dataset" "$FLUXLAB_ROOT/datasets/$dataset" -P
    ;;

  push-run)
    run_id="${1:-}"
    need_arg "$run_id" "run_id"
    rclone copy "$FLUXLAB_ROOT/runs/$run_id" "$GDRIVE_REMOTE/runs/$run_id" -P \
      --exclude ".secrets/**" \
      --exclude "cache/**" \
      --exclude "venv/**" \
      --exclude "node_modules/**" \
      --exclude "__pycache__/**"
    ;;

  sync-samples)
    run_id="${1:-}"
    need_arg "$run_id" "run_id"
    mkdir -p "$FLUXLAB_ROOT/runs/$run_id/samples"
    rclone copy "$FLUXLAB_ROOT/runs/$run_id/samples" "$GDRIVE_REMOTE/runs/$run_id/samples" -P
    ;;

  backup-loop)
    run_id="${1:-}"
    seconds="${2:-300}"
    need_arg "$run_id" "run_id"
    while true; do
      date
      "$0" push-run "$run_id"
      sleep "$seconds"
    done
    ;;

  pull-hf)
    repo_id="${1:-}"
    local_subdir="${2:-}"
    repo_type="${3:-model}"
    need_arg "$repo_id" "repo_id"
    need_arg "$local_subdir" "local_subdir"
    mkdir -p "$FLUXLAB_ROOT/$local_subdir"
    huggingface-cli download "$repo_id" \
      --repo-type "$repo_type" \
      --local-dir "$FLUXLAB_ROOT/$local_subdir"
    ;;

  push-hf-model)
    repo_id="${1:-}"
    local_dir="${2:-}"
    need_arg "$repo_id" "repo_id"
    need_arg "$local_dir" "local_dir"
    huggingface-cli upload "$repo_id" "$FLUXLAB_ROOT/$local_dir" --repo-type model
    ;;

  start-ui)
    cd "$FLUXLAB_ROOT/src/ai-toolkit/ui"
    if [ -z "${AI_TOOLKIT_AUTH:-}" ] && [ -f "$SECRETS_DIR/ai_toolkit_auth" ]; then
      export AI_TOOLKIT_AUTH="$(cat "$SECRETS_DIR/ai_toolkit_auth")"
    fi
    if [ -z "${AI_TOOLKIT_AUTH:-}" ]; then
      echo "AI_TOOLKIT_AUTH is not set. Create $SECRETS_DIR/ai_toolkit_auth or export it." >&2
      exit 1
    fi
    npm install
    AI_TOOLKIT_AUTH="$AI_TOOLKIT_AUTH" npm run build_and_start -- --host 0.0.0.0 --port 8675
    ;;

  start-comfy)
    mkdir -p "$FLUXLAB_ROOT/prod/input" "$FLUXLAB_ROOT/prod/output" "$FLUXLAB_ROOT/prod/temp"
    cd "$FLUXLAB_ROOT/src/ComfyUI"
    extra_args=()
    if [ -f "$FLUXLAB_ROOT/configs/extra_model_paths.yaml" ]; then
      extra_args+=(--extra-model-paths-config "$FLUXLAB_ROOT/configs/extra_model_paths.yaml")
    fi
    python3 main.py \
      --listen 0.0.0.0 \
      --port 8188 \
      --input-directory "$FLUXLAB_ROOT/prod/input" \
      --output-directory "$FLUXLAB_ROOT/prod/output" \
      --temp-directory "$FLUXLAB_ROOT/prod/temp" \
      --disable-auto-launch \
      "${extra_args[@]}"
    ;;

  prod-pull)
    batch_id="${1:-}"
    need_arg "$batch_id" "batch_id"
    mkdir -p "$FLUXLAB_ROOT/prod/batches/$batch_id"
    rclone copy "$GDRIVE_REMOTE/prod/batches/$batch_id" "$FLUXLAB_ROOT/prod/batches/$batch_id" -P
    ;;

  prod-run)
    manifest="${1:-}"
    batch_id="${2:-}"
    need_arg "$manifest" "manifest_jsonl"
    need_arg "$batch_id" "batch_id"
    mkdir -p "$FLUXLAB_ROOT/prod/batches/$batch_id/history" "$FLUXLAB_ROOT/prod/output/$batch_id"
    python3 "$FLUXLAB_ROOT/scripts/prodgen_runner.py" \
      --server "127.0.0.1:8188" \
      --workflow-root "$FLUXLAB_ROOT/prod/workflows" \
      --manifest "$manifest" \
      --batch-id "$batch_id" \
      --history-dir "$FLUXLAB_ROOT/prod/batches/$batch_id/history"
    ;;

  prod-push)
    batch_id="${1:-}"
    need_arg "$batch_id" "batch_id"
    if [ -d "$FLUXLAB_ROOT/prod/batches/$batch_id" ]; then
      rclone copy "$FLUXLAB_ROOT/prod/batches/$batch_id" "$GDRIVE_REMOTE/prod/batches/$batch_id" -P
    fi
    if [ -d "$FLUXLAB_ROOT/prod/output/$batch_id" ]; then
      rclone copy "$FLUXLAB_ROOT/prod/output/$batch_id" "$GDRIVE_REMOTE/prod/batches/$batch_id/output" -P
    fi
    ;;

  prod-run-and-stop)
    manifest="${1:-}"
    batch_id="${2:-}"
    need_arg "$manifest" "manifest_jsonl"
    need_arg "$batch_id" "batch_id"
    "$0" prod-run "$manifest" "$batch_id"
    "$0" prod-push "$batch_id"
    "$0" stop-pod
    ;;

  stop-pod)
    pod_id="${RUNPOD_POD_ID:-}"
    if [ -z "$pod_id" ] && [ -f "$SECRETS_DIR/runpod_pod_id" ]; then
      pod_id="$(cat "$SECRETS_DIR/runpod_pod_id")"
    fi
    if [ -z "$pod_id" ]; then
      echo "RUNPOD_POD_ID is not set and $SECRETS_DIR/runpod_pod_id is missing." >&2
      exit 1
    fi
    if command -v runpodctl >/dev/null 2>&1; then
      runpodctl pod stop "$pod_id"
    else
      api_key="${RUNPOD_API_KEY:-}"
      if [ -z "$api_key" ] && [ -f "$SECRETS_DIR/runpod_api_key" ]; then
        api_key="$(cat "$SECRETS_DIR/runpod_api_key")"
      fi
      if [ -z "$api_key" ]; then
        echo "runpodctl missing and RUNPOD_API_KEY is not available." >&2
        exit 1
      fi
      curl --request POST \
        --header 'content-type: application/json' \
        --url "https://api.runpod.io/graphql?api_key=${api_key}" \
        --data "{\"query\":\"mutation { podStop(input: {podId: \\\"${pod_id}\\\"}) { id desiredStatus } }\"}"
    fi
    ;;

  clean-cache)
    rm -rf "$FLUXLAB_ROOT/cache/tmp" "$FLUXLAB_ROOT"/src/ai-toolkit/ui/.next/cache 2>/dev/null || true
    rm -rf "$FLUXLAB_ROOT"/src/ComfyUI/temp 2>/dev/null || true
    find "$FLUXLAB_ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
    ;;

  ""|-h|--help|help)
    usage
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 2
    ;;
esac
EOF

chmod 700 /workspace/fluxlab/scripts/fluxlab
ln -sf /workspace/fluxlab/scripts/fluxlab /usr/local/bin/fluxlab
```

Validate:

```bash
fluxlab doctor
fluxlab disk
```

## 14. Secret Files

Recommended secret files:

```text
/workspace/fluxlab/.secrets/
  rclone.conf
  hf_token
  ai_toolkit_auth
  gcp-sa.json              optional, only if using GCS
```

Set permissions:

```bash
chmod 700 /workspace/fluxlab/.secrets
chmod 600 /workspace/fluxlab/.secrets/*
```

### Convenience Mode For Secrets

If convenience matters more than strict secret hygiene, keep a copy in:

```text
Google Drive: FluxLab/bootstrap/secrets/
```

Then restore onto a new pod:

```bash
fluxlab pull-secrets
```

After updating tokens on the pod, push them back:

```bash
fluxlab push-secrets
```

This is intentionally explicit. Regular run backups still exclude `.secrets/`
so a training output sync does not mix credentials into every run folder.

Private GitHub is also acceptable for this low-risk convenience setup. If you
do that, keep secrets in a private bootstrap repo, not in a public fork. Public
GitHub tokens are quickly harvested by bots, which can burn credits or delete
artifacts before you notice.

To create the AI-Toolkit UI password:

```bash
read -rsp "AI-Toolkit UI password: " AITK_AUTH
printf "\n"
printf "%s" "$AITK_AUTH" > /workspace/fluxlab/.secrets/ai_toolkit_auth
chmod 600 /workspace/fluxlab/.secrets/ai_toolkit_auth
unset AITK_AUTH
```

## 15. Start The AI-Toolkit UI

Use `tmux` so the UI survives SSH disconnects:

```bash
tmux new -s aitk
source /workspace/fluxlab/scripts/env.sh
fluxlab start-ui
```

Detach from tmux:

```text
Ctrl-b d
```

Reattach:

```bash
tmux attach -t aitk
```

Open the UI:

```text
http://localhost:8675
```

or the RunPod exposed HTTP URL if you are not using SSH forwarding.

## 16. Training Run Workflow

Pick stable names:

```text
dataset_slug=char_alice_v001
run_id=char_alice_flux2k_001
```

Pull the dataset from Google Drive:

```bash
fluxlab pull-dataset char_alice_v001
```

Pull or prepare the model:

```bash
fluxlab pull-hf black-forest-labs/FLUX.2-Klein-dev models/base/flux2-klein-dev model
```

Create the local run folder:

```bash
mkdir -p /workspace/fluxlab/runs/char_alice_flux2k_001/{samples,checkpoints,logs,eval,final}
```

In AI-Toolkit UI:

1. Create a training job.
2. Point the dataset path at:

   ```text
   /workspace/fluxlab/datasets/char_alice_v001
   ```

3. Point output/run paths under:

   ```text
   /workspace/fluxlab/runs/char_alice_flux2k_001
   ```

4. Configure periodic samples and saves.
5. Start training.

In another tmux session, push outputs to Google Drive every few minutes:

```bash
tmux new -s backup
fluxlab backup-loop char_alice_flux2k_001 300
```

## 17. Suggested AI-Toolkit Sample Evaluation Pattern

For visual LoRA evaluation, use repeated prompts at different LoRA strengths.
Keep seeds fixed.

Example sample block:

```yaml
sample:
  sampler: "flowmatch"
  sample_every: 250
  width: 1024
  height: 1024
  seed: 12345
  walk_seed: false
  guidance_scale: 4
  sample_steps: 20
  samples:
    - prompt: "photo of TOK character, close-up portrait, soft window light"
      network_multiplier: 0.6
    - prompt: "photo of TOK character, close-up portrait, soft window light"
      network_multiplier: 0.8
    - prompt: "photo of TOK character, close-up portrait, soft window light"
      network_multiplier: 1.0
    - prompt: "photo of TOK character, close-up portrait, soft window light"
      network_multiplier: 1.2
    - prompt: "photo of TOK character, full body, city street, natural light"
      network_multiplier: 1.0
    - prompt: "photo of TOK character, side profile, simple studio background"
      network_multiplier: 1.0
```

This gives a rough step-by-weight visual grid in the AI-Toolkit Samples tab and
in the synced Google Drive folder.

## 18. Reviewing Outputs In Google Drive

After sync:

```text
Google Drive -> FluxLab -> runs -> <run_id> -> samples
```

Review images/videos directly in the browser. Delete or move bad outputs in
Drive. If you manually curate a review folder, keep it separate:

```text
FluxLab/review/<run_id>/
```

Do not use Drive deletion as the authority for the raw RunPod run unless you
intend to mirror deletes back with `rclone sync`.

## 19. Upload Final LoRA To Hugging Face

Copy the final checkpoint(s) into:

```text
/workspace/fluxlab/runs/<run_id>/final/
```

Upload:

```bash
fluxlab push-hf-model xuzihe2020/char-alice-lora runs/char_alice_flux2k_001/final
```

For a private repo, create it first:

```bash
huggingface-cli repo create char-alice-lora --type model --private
```

Then upload again.

## 20. Production Image Generation With ComfyUI

This is the companion workflow for overnight production generation:

```text
Local machine:
  explore prompts and workflows interactively
  export ComfyUI workflow as API JSON
  prepare a batch manifest
  upload workflows, manifests, and inputs to Google Drive

RunPod:
  pull the batch from Google Drive
  start ComfyUI headless/API mode
  run the manifest through the ComfyUI API
  save outputs under /workspace/fluxlab/prod/output/<batch_id>
  upload outputs/history/logs back to Google Drive
  stop the pod when done
```

This is possible because ComfyUI exposes:

```text
POST /prompt
GET  /history/{prompt_id}
GET  /queue
```

The local ComfyUI repo also includes examples under `script_examples/` showing
how to queue exported API workflows.

### Production Directory Layout

On RunPod:

```text
/workspace/fluxlab/prod/
  workflows/
    flux_portrait_api.json
    flux_controlnet_pose_api.json
  batches/
    night_001/
      manifest.jsonl
      inputs/
      history/
      logs/
  input/
  output/
    night_001/
  temp/
```

In Google Drive:

```text
FluxLab/
  prod/
    workflows/
    batches/
      night_001/
        manifest.jsonl
        inputs/
        output/
        history/
        logs/
```

### Export Workflows From Local ComfyUI

On your local ComfyUI desktop:

1. Build and test the workflow interactively.
2. Use `File -> Export (API)`.
3. Save the exported JSON as:

   ```text
   FluxLab/prod/workflows/<workflow_slug>_api.json
   ```

4. Identify the node IDs and fields you want to patch, for example:

   ```text
   6.inputs.text                         positive prompt
   7.inputs.text                         negative prompt
   3.inputs.seed                         sampler seed
   3.inputs.steps                        sampler steps
   12.inputs.lora_name                   LoRA file name
   12.inputs.strength_model              LoRA strength
   20.inputs.image                       LoadImage file name
   9.inputs.filename_prefix              SaveImage output prefix
   ```

The exact node IDs come from the exported API JSON. They are workflow-specific.

### Batch Manifest Format

Use JSONL: one JSON object per generation job. Each line can point to a workflow
and patch any fields by dotted JSON path.

Example:

```jsonl
{"name":"hero_001","workflow":"flux_portrait_api.json","save_node":"9","updates":{"6.inputs.text":"photo of TOK character, cinematic close-up, soft window light","7.inputs.text":"blurry, distorted hands","3.inputs.seed":101,"12.inputs.lora_name":"char_alice.safetensors","12.inputs.strength_model":0.8}}
{"name":"hero_002","workflow":"flux_portrait_api.json","save_node":"9","updates":{"6.inputs.text":"photo of TOK character, full body, city street at night","7.inputs.text":"blurry, distorted hands","3.inputs.seed":102,"12.inputs.lora_name":"char_alice.safetensors","12.inputs.strength_model":1.0}}
{"name":"pose_001","workflow":"flux_controlnet_pose_api.json","save_node":"31","updates":{"6.inputs.text":"photo of TOK character, fashion editorial lighting","3.inputs.seed":201,"20.inputs.image":"night_001/inputs/pose_001.png","12.inputs.lora_name":"char_alice.safetensors","12.inputs.strength_model":0.9}}
```

If `save_node` is present, the runner sets that node's `filename_prefix` to:

```text
<batch_id>/<job_name>
```

That makes ComfyUI save under:

```text
/workspace/fluxlab/prod/output/<batch_id>/
```

You can still override `filename_prefix` manually in `updates`.

### Create The Batch Runner

Create this script on the pod:

```bash
cat > /workspace/fluxlab/scripts/prodgen_runner.py <<'EOF'
#!/usr/bin/env python3
import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_manifest(path: Path):
    jobs = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_num}: invalid JSONL: {exc}") from exc
    return jobs


def set_path(obj, dotted_path, value):
    parts = dotted_path.split(".")
    cur = obj
    for part in parts[:-1]:
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(server, timeout_s):
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            get_json(f"http://{server}/system_stats")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise SystemExit(f"ComfyUI did not become ready at {server}: {last_error}")


def queue_prompt(server, prompt, prompt_id):
    return post_json(f"http://{server}/prompt", {"prompt": prompt, "prompt_id": prompt_id})


def wait_for_history(server, prompt_id, timeout_s, poll_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        history = get_json(f"http://{server}/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id}")


def main():
    parser = argparse.ArgumentParser(description="Run a ComfyUI production batch from JSONL.")
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--workflow-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--poll-s", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    workflow_root = Path(args.workflow_root)
    manifest_path = Path(args.manifest)
    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    wait_for_server(args.server, timeout_s=180)
    jobs = read_manifest(manifest_path)
    results_path = history_dir / "results.jsonl"

    with results_path.open("a", encoding="utf-8") as results:
        for idx, job in enumerate(jobs, start=1):
            name = job.get("name") or f"job_{idx:04d}"
            workflow_name = job.get("workflow")
            if not workflow_name:
                raise SystemExit(f"Job {name} is missing workflow")

            workflow_path = Path(workflow_name)
            if not workflow_path.is_absolute():
                workflow_path = workflow_root / workflow_path

            print(f"[{idx}/{len(jobs)}] {name}: {workflow_path}", flush=True)
            prompt = copy.deepcopy(read_json(workflow_path))

            save_node = job.get("save_node")
            if save_node:
                set_path(prompt, f"{save_node}.inputs.filename_prefix", f"{args.batch_id}/{name}")

            for dotted_path, value in job.get("updates", {}).items():
                set_path(prompt, dotted_path, value)

            prompt_id = str(uuid.uuid4())
            record = {
                "name": name,
                "workflow": str(workflow_path),
                "prompt_id": prompt_id,
                "status": "queued",
                "started_at": time.time(),
            }

            try:
                queue_response = queue_prompt(args.server, prompt, prompt_id)
                record["queue_response"] = queue_response
                history = wait_for_history(args.server, prompt_id, args.timeout_s, args.poll_s)
                comfy_status = history.get("status", {})
                record["comfy_status"] = comfy_status
                status_str = comfy_status.get("status_str", "unknown")
                completed = comfy_status.get("completed", False)
                record["status"] = "done" if completed and status_str == "success" else f"comfy_{status_str}"
                record["finished_at"] = time.time()

                history_path = history_dir / f"{name}_{prompt_id}.json"
                with history_path.open("w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
                record["history_path"] = str(history_path)
                if record["status"] != "done" and not args.continue_on_error:
                    raise RuntimeError(f"ComfyUI job {name} finished with status {record['status']}")
            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
                record["finished_at"] = time.time()
                print(f"ERROR: {name}: {exc}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    results.write(json.dumps(record) + "\n")
                    raise

            results.write(json.dumps(record) + "\n")
            results.flush()


if __name__ == "__main__":
    main()
EOF

chmod 700 /workspace/fluxlab/scripts/prodgen_runner.py
```

### Install ComfyUI On The Pod

Clone ComfyUI into the same workspace:

```bash
cd /workspace/fluxlab/src
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If you need custom nodes, install them into:

```text
/workspace/fluxlab/src/ComfyUI/custom_nodes/
```

Keep this in GitHub notes or a bootstrap script so the pod can be rebuilt.

### Model Paths For ComfyUI

Use `extra_model_paths.yaml` so ComfyUI sees models on the network volume.
Do not use the local Windows config unchanged on RunPod: the checked-in
`extra_model_paths.yaml` points at `C:/Users/Tony Xu/workspace/comfyui_models`,
which does not exist in the pod.

For this repo's model layout, keep the model store beside the ComfyUI checkout
on the RunPod volume:

```text
/workspace/comfyui/          ComfyUI checkout
/workspace/comfyui_models/   persistent model store on the RunPod volume
```

Then write the RunPod model-path config:

```bash
cat > /workspace/fluxlab/configs/extra_model_paths.yaml <<'EOF'
external_comfyui_models:
  base_path: /workspace/comfyui_models
  is_default: true

  checkpoints: checkpoints
  diffusion_models: checkpoints
  loras: lora
  text_encoders: text_encoders
  vae: vae
  download_model_base: .
  sams: sams
  ultralytics: ultralytics
  ultralytics_bbox: ultralytics/bbox
  ultralytics_segm: ultralytics/segm
  yolo: yolo
  seedvr2: SEEDVR2
EOF
```

If you launch this repo's ComfyUI checkout directly from its root, update that
checkout's `extra_model_paths.yaml` with the same RunPod `base_path` before
starting ComfyUI:

```bash
cp /workspace/fluxlab/configs/extra_model_paths.yaml /workspace/comfyui/extra_model_paths.yaml
```

If your RunPod volume uses a different mount or folder name, replace only
`base_path`; keep the model category keys aligned with the repo's workflows.
For example, SeedVR2 expects the `seedvr2` model type and the physical
`SEEDVR2` folder.

### Run An Overnight Batch

Start ComfyUI in one tmux session:

```bash
tmux new -s comfy
source /workspace/fluxlab/scripts/env.sh
fluxlab start-comfy
```

Detach with `Ctrl-b d`.

Pull the batch from Google Drive:

```bash
fluxlab prod-pull night_001
```

Run the batch, upload outputs, then stop the pod:

```bash
tmux new -s prod
source /workspace/fluxlab/scripts/env.sh
fluxlab prod-run-and-stop /workspace/fluxlab/prod/batches/night_001/manifest.jsonl night_001
```

If you want to inspect before stopping:

```bash
fluxlab prod-run /workspace/fluxlab/prod/batches/night_001/manifest.jsonl night_001
fluxlab prod-push night_001
```

Then stop manually:

```bash
fluxlab stop-pod
```

### Failure Handling

For unattended batches, consider:

- Use a small test manifest with 1-2 jobs before a long run.
- Keep each job output prefix unique.
- Use fixed seeds for reproducibility.
- Sync periodically for very large batches by running `fluxlab prod-push
  <batch_id>` from another tmux session.
- Set RunPod's built-in stop-after timer as a hard cost ceiling.
- Do not use `prod-run-and-stop` until you trust the manifest and sync path.

The pod stop happens only after `prod-run` and `prod-push` return successfully.
If generation fails, the command exits before stopping so you can inspect logs,
unless you intentionally add more forgiving behavior.

## 21. Manual Setup Checklist

Use this checklist for a fresh pod:

```text
[ ] Deploy RunPod Pod, not Serverless.
[ ] Add SSH public key.
[ ] Expose or forward port 8675.
[ ] SSH into pod.
[ ] Create /workspace/fluxlab layout.
[ ] Install git, git-lfs, rclone, tmux, node, huggingface_hub.
[ ] Clone https://github.com/xuzihe2020/ai-toolkit.
[ ] Create /workspace/fluxlab/.secrets.
[ ] Install rclone.conf manually, or restore it with fluxlab pull-secrets.
[ ] Install hf_token manually, or restore it with fluxlab pull-secrets.
[ ] Install ai_toolkit_auth manually, or restore it with fluxlab pull-secrets.
[ ] Create scripts/env.sh.
[ ] Create scripts/fluxlab.
[ ] Run fluxlab doctor.
[ ] Install AI-Toolkit Python deps.
[ ] Start AI-Toolkit UI in tmux.
[ ] Pull dataset from Google Drive.
[ ] Pull model from Hugging Face.
[ ] Start training job.
[ ] Start backup-loop for run outputs.
[ ] Review samples in Google Drive.
[ ] Upload final LoRA to Hugging Face.
[ ] Optional: clone/install ComfyUI for production generation.
[ ] Optional: rewrite ComfyUI extra_model_paths.yaml for the RunPod volume path.
[ ] Optional: create prodgen_runner.py.
[ ] Optional: test fluxlab start-comfy and a tiny production manifest.
```

## 22. Codex SSH Handoff

When asking Codex to work on the pod, prepare the pod first:

1. The pod should be reachable by SSH.
2. Secrets should already exist in `/workspace/fluxlab/.secrets`, or be
   restorable with `fluxlab pull-secrets`.
3. The `fluxlab` command should be installed.
4. AI-Toolkit fork should be cloned under `/workspace/fluxlab/src/ai-toolkit`.
5. Long-running commands should run inside `tmux`.

Tell Codex the SSH alias, for example:

```text
Use SSH host runpod-ai-toolkit. Do not print secret values.
Use /workspace/fluxlab as the root.
Use fluxlab doctor first.
```

Codex can then:

- inspect disk/GPU state with `fluxlab doctor` and `fluxlab disk`
- pull named datasets
- start or restart the UI
- edit config files
- monitor logs
- push outputs to Google Drive
- run ComfyUI production batches
- stop the pod after output sync
- upload final LoRAs to Hugging Face
- modify your AI-Toolkit fork and push changes if GitHub auth is configured

Codex should not need to see or paste raw tokens in chat. The scripts read
secrets from the secret directory.

## 23. Disk Management

Check disk:

```bash
fluxlab disk
```

Common cleanup:

```bash
fluxlab clean-cache
rm -rf /workspace/fluxlab/runs/<old_run_id>
rm -rf /workspace/fluxlab/datasets/<old_dataset_slug>
rm -rf /workspace/fluxlab/models/base/<unused_model>
```

Before deleting a run, push it:

```bash
fluxlab push-run <run_id>
```

## 24. Provider-Specific Notes

RunPod persistent storage is still billed while the pod is stopped. The GPU
compute cost stops when the pod is stopped, but the persistent disk or network
volume continues to cost money.

The 80GB disk should be enough only if you keep a narrow working set. If you
need multiple base models and many runs available at the same time, increase
the network volume or move completed artifacts out quickly.

## 25. References

- AI-Toolkit: https://github.com/ostris/ai-toolkit
- Your AI-Toolkit fork: https://github.com/xuzihe2020/ai-toolkit
- ComfyUI API examples: https://github.com/comfyanonymous/ComfyUI/tree/master/script_examples
- RunPod GraphQL Pod management: https://docs.runpod.io/sdks/graphql/manage-pods
- RunPod environment variables: https://docs.runpod.io/pods/templates/environment-variables
- RunPod Pods: https://docs.runpod.io/pods
- RunPod SSH: https://docs.runpod.io/pods/configuration/use-ssh
- RunPod exposed ports: https://docs.runpod.io/pods/configuration/expose-ports
- RunPod storage: https://docs.runpod.io/pods/storage/types
- rclone Google Drive: https://rclone.org/drive/
- Hugging Face Hub CLI: https://huggingface.co/docs/huggingface_hub/guides/cli
- Hugging Face upload guide: https://huggingface.co/docs/huggingface_hub/guides/upload
