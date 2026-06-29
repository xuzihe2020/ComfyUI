# FLUX.2-dev (BF16) + Alibaba ControlNet on RunPod — Setup Runbook

A one-time setup so you can spin a pod up/down and jump straight to experiments.
Everything heavy lives on a **persistent network volume at `/workspace`**, so models
and the environment survive pod restarts — daily startup is one command.

**Data flow (by design):**
- **Models** → HuggingFace (this doc)
- **Code / nodes / requirements** → GitHub (cloned by `setup.sh`)
- **Reference images** → Google Drive (section 9, wired up later)

---

## Deployment order at a glance

Run top to bottom; details are in the numbered sections below.

| # | Where | Do | Section |
|---|---|---|---|
| A1 | Browser · HF | Accept the FLUX.2-dev license + create a **READ** token | §1 |
| A2 | Browser · RunPod | Add your SSH **public key** to Settings | §1 |
| B1 | Browser · RunPod | **Create** the 200 GB network volume in an A100-80GB region | §2a |
| B2 | Browser · RunPod | **Deploy** the A100 pod *with that volume selected* → copy SSH cmd | §2b–2c |
| C | Claude · SSH | Paste Claude the SSH cmd + HF token → it runs setup → download → start → test | §3–8 |

> 🔑 **Two things that must happen in order:** accept the HF license **before**
> downloading (else 403), and select the volume **during** deploy (it can’t be
> attached afterward).

---

## 0. What gets installed

| Component | Source | Lands at | Size |
|---|---|---|---|
| FLUX.2-dev transformer (**BF16**) | `black-forest-labs/FLUX.2-dev` *(gated)* | `models/diffusion_models/flux2-dev.safetensors` | 64.4 GB |
| Mistral-3 text encoder (**BF16**) | `Comfy-Org/flux2-dev` | `models/text_encoders/mistral_3_small_flux2_bf16.safetensors` | 35.6 GB |
| FLUX.2 VAE | `Comfy-Org/flux2-dev` | `models/vae/flux2-vae.safetensors` | 0.3 GB |
| Fun ControlNet Union | `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` | `models/controlnet/FLUX.2-dev-Fun-Controlnet-Union.safetensors` | 8.3 GB |
| ComfyUI + all custom nodes (via `script/install_custom_nodes.py` + manifest) | GitHub `xuzihe2020/ComfyUI` | the repo + `custom_nodes/` | — |

**Total model download ≈ 108 GB** → size the network volume at **200 GB**.

---

## 1. Prerequisites (do once, in a browser)

1. **Accept the gated license.** Logged into HuggingFace, open
   <https://huggingface.co/black-forest-labs/FLUX.2-dev> and click **“Agree and
   access repository.”** (Without this, the 64 GB transformer download 403s.)
   If `Comfy-Org/flux2-dev` also prompts, accept it too.
2. **Create an HF token** (READ scope): <https://huggingface.co/settings/tokens>.
3. **Add your SSH public key** to RunPod → **Settings → SSH Public Keys**:
   ```
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAAsGUB9jkh4NCFktBYkxWKbAN1aUm48VxQiQt/f/n6R
   ```

---

## 2. Create the network volume, then deploy the pod

> ⚠️ **Order matters.** A network volume can **only be attached while deploying a
> pod** — RunPod cannot add or detach one afterward (*"they cannot be attached or
> detached later without deleting the Pod"*). The 80 GB container disk can’t hold the
> 64 GB transformer, so if you’ve already started deploying the A100 **without** a
> volume selected: **don’t deploy** — or terminate that pod and redeploy per 2b.

### 2a. Create the volume (do this first)

1. Console → **Storage** (<https://console.runpod.io/user/storage>) → **New Network Volume**.
2. **Data center:** pick one that has **A100 80GB** in stock — this *locks the volume to
   that region*, and the pod must be deployed in the same region.
3. **Name:** e.g. `flux2-models`. **Size: 200 GB** (≈ **$14/mo** at $0.07/GB/mo, billed
   continuously, including while the pod is stopped). **Tier:** Standard.
4. Click **Create Network Volume.**

### 2b. Deploy the pod with the volume bound

1. Console → **Pods** → **Deploy**.
2. Under **Network Volume**, **select `flux2-models`.** This binds it and filters the
   GPU list to that volume’s region.
3. **GPU:** A100 80GB PCIe. **Template:** `runpod/pytorch` (2.8.0 is fine — bare PyTorch;
   we install ComfyUI ourselves).
4. The volume **auto-mounts at `/workspace`** (one volume per pod; it replaces the default
   volume disk). Container disk can stay small — all big files live on `/workspace`.
5. SSH port 22 is exposed by default. Click **Deploy On-Demand**.

### 2c. Copy the SSH command

**Connect → “SSH over exposed TCP”** and copy it. Use this one — **not** the
`ssh.runpod.io` proxy (it can’t port-forward the UI):
```
ssh root@69.48.x.x -p 41953 -i ~/.ssh/id_ed25519
```

(Spinning the pod down later and reusing the same volume is covered in §10.)

---

## 3. Connect + SSH config (so both you and Claude use a short alias)

Add this to your Mac’s `~/.ssh/config` (fill IP/port from step 2 — they change
each time the pod is recreated). The `LocalForward` line tunnels the ComfyUI UI:

```sshconfig
Host runpod
    HostName 69.48.x.x          # <-- from RunPod Connect panel
    Port 41953                  # <-- from RunPod Connect panel
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
    LocalForward 8188 localhost:8188
    ServerAliveInterval 30
```

Now `ssh runpod` opens a shell **and** forwards port 8188. Test: `ssh runpod nvidia-smi`.

---

## 4. Clone this repo onto the pod

`ssh runpod`, then clone your ComfyUI fork onto the volume and enter it. The repo
carries ComfyUI itself, the custom-node **manifest + installer**, the `runpod/`
scripts, and your workflows:
```bash
git clone https://github.com/xuzihe2020/ComfyUI /workspace/ComfyUI
cd /workspace/ComfyUI
```
> ⚠️ **Commit + push first.** The pose-ControlNet workflow, the two new
> `custom_nodes.manifest.json` entries, and the `install_custom_nodes.py` change must
> be committed and pushed to `xuzihe2020/ComfyUI`, or the clone won't include them.

---

## 5. Install (≈ 10–15 min) — `runpod/setup.sh`

```bash
cp runpod/.env.example runpod/.env
nano runpod/.env                 # paste HF_TOKEN
bash runpod/setup.sh
```
This refreshes the clone, creates **`.venv` at the repo root** (where the installer
expects it), installs PyTorch (cu124) + ComfyUI requirements, then runs the repo's own
**`script/install_custom_nodes.py`** — your manifest installer, which pulls every custom
node including `comfyui-flux2fun-controlnet` and `comfyui_controlnet_aux` (its native
deps — OpenCV/onnxruntime — are force-installed via `ALWAYS_FIX_DEPENDENCIES`). Finally
it **verifies `nvidia-smi` + `torch.cuda`** and aborts if the GPU isn't visible.

---

## 6. Download models (≈ 108 GB) — `runpod/download_models.sh`

```bash
bash runpod/download_models.sh   # uses the same runpod/.env (HF_TOKEN) from §5
```
Downloads into `/workspace/ComfyUI/models/{diffusion_models,text_encoders,vae,controlnet}`
(on the volume, gitignored). Resumable (uses `hf_transfer`); re-running skips completed
files. At the end it prints `ls -lh` of every model dir + `df -h /workspace`. The script
sets `HF_HOME=/workspace/.hf_cache` so the download cache lands on the 200 GB volume,
**not** the ~80 GB container disk (otherwise the 64 GB transformer can fill `/`).

---

## 7. Launch ComfyUI + open it in your browser — `start.sh`

On the pod:
```bash
bash runpod/start.sh     # runs ComfyUI in tmux session 'comfy', logs to /workspace/comfyui.log
```
On your Mac (separate terminal):
```bash
ssh runpod               # the config's LocalForward tunnels 8188 automatically
```
Open **<http://localhost:8188>**. (If you prefer, expose 8188 as an HTTP port in
RunPod and use the proxy URL instead — but the SSH tunnel is simpler and private.)

---

## 8. How Claude debugs over SSH

Once `~/.ssh/config` has the `runpod` alias, Claude (from your Mac) can run, e.g.:
```bash
ssh runpod 'nvidia-smi'                                  # GPU / VRAM
ssh runpod 'tail -n 80 /workspace/comfyui.log'           # server log
ssh runpod 'tmux capture-pane -pt comfy | tail -n 40'    # live console
curl -s localhost:8188/system_stats                      # via the tunnel: server health
curl -s localhost:8188/object_info/Flux2FunControlNetApply | head  # node registered?
```
Just paste Claude the RunPod SSH command and it will fill in the config and take over
verification + the first test generation.

---

## 9. Reference images from Google Drive (wire up later)

When you share the Drive folder, install `rclone` (or `gdown`) on the pod and sync
into `ComfyUI/input/`:
```bash
# placeholder — fill in once access is granted
# pip install gdown && gdown --folder <folder-url> -O /workspace/ComfyUI/input/refs
# or: rclone copy gdrive:refs /workspace/ComfyUI/input/refs
```

---

## 10. Daily / restart flow (the “minimal time” path)

- **Pod was only stopped (same pod):** `ssh runpod` → `cd /workspace/ComfyUI` → `bash runpod/start.sh`. Done.
- **New pod, same network volume:** `ssh runpod` → `cd /workspace/ComfyUI` → `bash runpod/setup.sh`
  (fast — clone/venv/models already on the volume; it just `git pull`s, restores apt libs,
  and re-checks deps), then `bash runpod/start.sh`. Nothing re-downloads.
- Update the `HostName`/`Port` in `~/.ssh/config` to the new pod’s values.
- *Optional:* set the pod’s **Container Start Command** to
  `bash /workspace/ComfyUI/runpod/start.sh` so ComfyUI auto-launches on boot.

---

## 11. Using the ControlNet (quick reference)

Nodes (from `comfyui-flux2fun-controlnet`): **Load Flux2 Fun ControlNet** →
**Apply Flux2 Fun ControlNet** (inputs: conditioning, controlnet, vae, strength,
control_image). One checkpoint covers **Pose / Canny / Depth / HED / MLSD / Tile** —
it auto-detects the mode from the control image. Generate the control image with a
`comfyui_controlnet_aux` preprocessor (DWPose/OpenPose, Canny, Depth-Anything, …).

Recommended: **strength 0.65–0.80, 25–50 steps, CFG 3.5–4.5.**

> Your existing `vn_outfit_reference_draw_then_face_refine_flux2.json` also uses
> FaceParsing / Impact / essentials nodes. Load it in the UI and use
> **ComfyUI-Manager → Install Missing Custom Nodes** to pull the rest in one click.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | Wrong wheel for the driver. Check `nvidia-smi` CUDA version; set `TORCH_CUDA` in `.env` (cu124 for A100/Ada, cu128 for Blackwell) and re-run `setup.sh`. |
| 403 on `flux2-dev.safetensors` | License not accepted, or token lacks read scope (step 1). |
| `No space left on device` during download | HF cache landed on the container disk. Confirm `HF_HOME=/workspace/.hf_cache` is set (in `.env`), `df -h /` vs `/workspace`, then re-run. |
| ControlNet node “Module not found” | Restart ComfyUI (`bash start.sh`) after install. |
| ControlNet has no effect / black output | strength > 0, VAE connected, control image actually loaded. |
| OOM during sampling | Lower resolution; confirm you’re not passing `--highvram`. |
| Models don’t show in dropdowns | They must be under `models/<type>/`; re-check `download_models.sh` output, then refresh the UI. |
