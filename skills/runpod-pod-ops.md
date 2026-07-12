# Skill: RunPod Pod Operations

Agent-agnostic procedure for working with our RunPod setup. Read in full
before any work involving RunPod pods, network volumes, pod SSH, or cloud
training/generation runs. Established 2026-07-10.

The broader (older) reference for gdrive/HF sync and ai-toolkit details is
`docs/runpod-ai-toolkit-gdrive-hf-workflow.md`; where the two disagree, THIS
file wins.

## Current assets

| Asset | Value |
|---|---|
| Network volume | `frankie_the_pug_volume`, 100 GB, **EU-RO-1**, ID `1qwuc4cm14`, S3-compatible |
| Pod template | `Runpod Pytorch 2.8.0` (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`) — torch 2.8.0+cu128 |
| Training GPU | RTX 5090 ($0.99/hr secure). Inference fallback: RTX 4090 ($0.69/hr), fp8 weights |
| SSH key (per device) | `~/.ssh/runpod_ed25519` (+ `.pub`). Public keys registered in RunPod → Settings → SSH keys (multi-line box, one key per line) |
| Models | Synced workstation↔volume with `scripts/runpod/model_sync.py` (`-u`/`-d`, S3 API, works anywhere, no pod needed; volume root `comfyui_models/` mirrors the local root layout). Large-file uploads are RESUMABLE: a failed/killed `-u` keeps its uploaded parts — rerun the same command and it continues from the first missing part (RunPod's Cloudflare front 524s cost seconds, not the file). Volume usage/stats: `--probe` — POD ONLY (du on the mounted filesystem, seconds; S3 has no folder sizes and RunPod exposes no usage API). Images/outputs ↔ Google Drive via rclone, pinned to `Flux Prod` only — see `skills/runpod-gdrive-rclone.md`. NO Hugging Face — Tony has no HF token. Training base model source: TBD |
| Agent access | see "Agent access" below |

## The pod model: disposable, terminate-only

**Never "Stop" a pod — always Terminate.** Stopping releases the GPU but keeps
billing the disk, and restart requires the same GPU type free on the same host
(with scarce 5090s this usually fails → locked pod, migration dialog).
Everything durable lives on the network volume; the pod is a throwaway GPU
clipped onto it.

- End of session: sync/verify outputs on `/workspace`, **Terminate**. Cost while off: $7/mo volume only.
- Start of session: deploy a fresh pod onto the volume (~2 min). New IP/port each time.
- The volume attaches at deploy time only, one pod at a time (don't count on simultaneous multi-pod attach).
- Volume can be grown, never shrunk. Track usage with `du -sh /workspace/*` (NOT `df` — it shows the whole MooseFS cluster).
- Data in/out with no pod running: the volume's S3-compatible API (key via Storage → Create S3 API key).

## Pod env vars & RunPod Secrets

API keys reach pods via RunPod Secrets referenced from the pod template —
never via `.env` files on pods (real env vars outrank `.env` in
`lib/envfile.py`, so repo scripts just work).

- Secrets live in console → Settings → Secrets. NAMING RULE: a secret name
  CANNOT start with `RUNPOD` (reserved prefix; the console rejects it).
- A template env row maps env var ← secret:
  `ENV_VAR_NAME={{ RUNPOD_SECRET_<secret name> }}` — the reference is always
  `RUNPOD_SECRET_` + the secret's name, and the left side is whatever the
  tools expect, independent of the secret's name.
- Current mapping (secret name → env var):
  `XAI_API_KEY→XAI_API_KEY`, `FLUX_API_KEY→FLUX_API_KEY`,
  `OPENAI_API_KEY→OPENAI_API_KEY`, `GEMINI_API_KEY→GEMINI_API_KEY`,
  `S3_ACCESS_KEY_ID→S3_ACCESS_KEY_ID`,
  `S3_SECRET_ACCESS_KEY→S3_SECRET_ACCESS_KEY`
  (S3 keys are needed on pods for `model_sync.py --abort-stale-uploads`;
  transfers/probe don't need them there). Plain template vars:
  `HF_HOME=/workspace/hf-cache`, `VOLUME_SIZE_GB=<quota>`.
- Env vars bind at DEPLOY: template edits affect new pods only; a running
  pod needs manual `export`s or a redeploy.
- NON-INTERACTIVE SSH GOTCHA: RunPod injects template env via
  `/etc/rp_environment`, sourced only by interactive shells. Agent SSH
  commands (`ssh host 'cmd'`) MUST prefix `. /etc/rp_environment;` or they
  see none of the template env vars — interactive shells see them fine, so
  a "missing env" seen only over agent SSH is this, not a broken template.

## Deploy checklist (console)

1. Pods → Deploy → Secure Cloud → RTX 5090 → attach `frankie_the_pug_volume` (Region row must show it; mounts at `/workspace`).
2. Template: `Runpod Pytorch 2.8.0`. Container disk: 80 GB (it's ephemeral scratch).
3. Set overrides → expose HTTP ports **8188** (ComfyUI) and **8675** (ai-toolkit UI); env vars `HF_TOKEN=<token>`, `HF_HOME=/workspace/hf-cache`. Verify they actually stick (check `env | grep HF_` on the pod).
4. SSH keys come from account settings automatically. On-demand, not spot, for interactive sessions.
5. Grab the "SSH over exposed TCP" line from the Connect tab: `ssh root@<ip> -p <port> -i ~/.ssh/runpod_ed25519` (the console prints `id_ed25519` — our key name differs).

## First session on a fresh volume + every spin-up: see the setup skill

The full software process — fork clone, in-repo `.venv`s, manifest-driven
custom-node install, Google Drive model sync, and the MANDATORY every-spin-up
sync (pull fork, pull every node, run `scripts/install_custom_nodes.py`) — is
defined in **`skills/runpod-comfyui-aitk-setup.md`**. Follow it exactly; the
summary here is not a substitute.

```bash
ssh root@<ip> -p <port> -i ~/.ssh/runpod_ed25519
source /workspace/scripts/session_start.sh    # runs the mandatory sync
# ComfyUI:
python /workspace/ComfyUI/main.py --listen 0.0.0.0 --port 8188
# training (headless):
source /workspace/ai-toolkit/.venv/bin/activate
cd /workspace/ai-toolkit && python run.py /workspace/configs/<job>.yaml
```

Web UIs: RunPod proxy links in the Connect tab (ports 8188/8675), or SSH
tunnel from any machine: `ssh -L 8188:localhost:8188 -L 8675:localhost:8675 root@<ip> -p <port> -i ~/.ssh/runpod_ed25519`.

## SSH keys on a new workstation

One keypair per device, never copy private keys between machines:

```bash
ssh-keygen -t ed25519 -C "<device>-runpod" -f ~/.ssh/runpod_ed25519 -N ""
```

Add the `.pub` content as a new line in RunPod → Settings → SSH keys. Pods
deployed afterwards accept every registered key. (Windows: same commands in
PowerShell; config lives at `C:\Users\<you>\.ssh\`.)

## Agent access (any coding agent, any workstation)

Agents drive the pod over SSH but must NOT grant themselves permissions —
a human authorizes SSH access once per workstation, in the agent's own
permission mechanism. For Claude Code, that's these allow rules in
`.claude/settings.local.json` (per machine, gitignored) or `.claude/settings.json`
(committed, all machines):

```json
{
  "permissions": {
    "allow": [
      "Bash(ssh -i ~/.ssh/runpod_ed25519 *)",
      "Bash(scp -i ~/.ssh/runpod_ed25519 *)"
    ]
  }
}
```

For other agents (Codex etc.), grant the equivalent shell approval for
`ssh -i ~/.ssh/runpod_ed25519 ...` / `scp -i ~/.ssh/runpod_ed25519 ...`.

Starting a new agent thread: have it read this file, then paste the current
pod's Connect line (`root@<ip> -p <port>`) — the IP/port change on every
deploy, so they are deliberately NOT recorded here.

Secrets stay human-handled: agents must not receive HF/RunPod API tokens in
chat; export them in the pod terminal or put them in the repo `.env`
(`RUNPOD_API_KEY`, for the planned pod_up/pod_down automation).

## Cost quick sheet

- 5090 pod $0.99/hr (billed per ms, only while pod exists) + container disk ~$0.004/hr
- Volume $0.07/GB/mo → 100 GB = $7/mo, always
- Training run (2k steps, Klein 9B): ~$0.6–1.2. Batch inference: ~$0.005/image
- Smoke-run budget for a new setup: ~$2–4 total
