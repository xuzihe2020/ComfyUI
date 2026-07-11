# RunPod rclone Google Drive Workflow

Use `rclone` on the RunPod network volume to move image datasets and generated
outputs between Google Drive and `/workspace`.

This is for images and run outputs, not ComfyUI model sync. Model sync uses
`scripts/runpod/model_sync.py` and the RunPod volume S3 API.

## Storage Layout

Persistent files live under `/workspace` on the RunPod network volume:

```text
/workspace/bin/rclone              # persistent rclone binary
/workspace/rclone/rclone.conf      # Google Drive token/config; secret
/workspace/datasets/               # pulled training/reference images
/workspace/output/                 # ComfyUI outputs
/workspace/prod/                   # optional production batches
```

## Scope: Pinned To "Flux Prod" — HARD RULE

The `gdrive` remote is pinned with `root_folder_id` to the Drive folder
`Flux Prod` (ID `1tuMvBkNNREzBBCHVY-DH6ViiQEiveWEi`). `gdrive:` RESOLVES TO
that folder — the rest of the Drive is not addressable through this remote.

Agents and scripts MUST NOT unset or change `root_folder_id`, MUST NOT create
additional Drive remotes, and MUST NOT attempt to reach anything outside
`Flux Prod`. The pin was set with:

```bash
rclone config update gdrive root_folder_id 1tuMvBkNNREzBBCHVY-DH6ViiQEiveWEi --non-interactive
```

(Note: OAuth scope itself is full-Drive — Google offers no folder-level
consent — so the pin is the enforced boundary at the tool level and the
config file remains a secret.)

Actual Drive layout under the pin (as of 2026-07-11):

```text
Flux Prod/          <- gdrive: root
  Inputs/           <- datasets, reference images
  Outputs/          <- generated results, run outputs
  Tests/
  screenshots/
```

TODO: rclone's shared Google client_id is being retired during 2026 — create
our own client_id per https://rclone.org/drive/#making-your-own-client-id and
update the remote before it stops working.

## One-Time Persistent rclone Install

Do not rely on `apt-get install rclone` as the durable setup. The pod container
is disposable, so apt-installed binaries disappear when the pod is terminated.

Install the rclone binary onto the network volume instead:

```bash
mkdir -p /workspace/bin /workspace/rclone /workspace/tmp
cd /workspace/tmp

curl -L https://downloads.rclone.org/rclone-current-linux-amd64.zip -o rclone.zip
unzip -o rclone.zip
cp rclone-*-linux-amd64/rclone /workspace/bin/rclone
chmod +x /workspace/bin/rclone
chmod 700 /workspace/rclone
```

For every pod session:

```bash
export PATH=/workspace/bin:$PATH
export RCLONE_CONFIG=/workspace/rclone/rclone.conf
```

## Get `rclone.conf`

`rclone.conf` stores the Google OAuth refresh token. Treat it like a secret.
Never commit it and do not paste its contents into chat.

### Option A: Configure Directly On The Pod

Use this if you are SSH'd into the pod and can open the Google auth URL in your
local browser:

```bash
export PATH=/workspace/bin:$PATH
export RCLONE_CONFIG=/workspace/rclone/rclone.conf
rclone config
```

Create a remote named:

```text
gdrive
```

Choose the Google Drive backend. For scope, use full Drive read/write access:

```text
drive
```

This is the practical choice for Tony's workflow because the pod must read
datasets uploaded manually through the Google Drive web UI and write generated
outputs back to Drive. More limited scopes such as `drive.file` can be awkward
because they may only see files created by rclone.

If rclone prints an auth URL, open it locally, approve access, and paste the
result back into the SSH session.

Then secure and test:

```bash
chmod 600 /workspace/rclone/rclone.conf
rclone lsd gdrive:
rclone mkdir gdrive:FluxLab
```

## Windows Setup (PowerShell) — step-by-step reference

Run these in a Windows terminal, in order:

```powershell
# 1. install rclone (once)
winget install Rclone.Rclone
```

Close and reopen the terminal so `rclone` is on PATH.

```powershell
# 2. authorize — a Google consent page opens in the browser; approve Drive access
rclone config create gdrive drive scope=drive

# 3. THE PIN — mandatory, do not skip: scopes access to the Flux Prod folder only
rclone config update gdrive root_folder_id 1tuMvBkNNREzBBCHVY-DH6ViiQEiveWEi --non-interactive

# 4. verify — must list ONLY: Inputs, Outputs, Tests, screenshots
rclone lsd gdrive:
```

If step 4 shows your whole Drive instead of those four folders, step 3 didn't
apply — rerun it.

Config file location on Windows: `%APPDATA%\rclone\rclone.conf` — it now
holds your refresh token; secret rules apply (never in git, never in chat).

Daily usage:

```powershell
# upload images to Drive
rclone copy "C:\some\folder\images" gdrive:Inputs/character_01 -P

# download results from Drive
rclone copy gdrive:Outputs/prod_batch_001 "C:\pulls\prod_batch_001" -P
```

Persistence: the grant works forever until you revoke it explicitly
(myaccount.google.com → Security → connections → rclone). No periodic
re-consent.

### Option B: Configure Locally, Then Copy To The Pod

Use this if local browser auth is easier on Windows or macOS.

On your local machine:

```powershell
rclone config
rclone lsd gdrive:
```

Then copy the config to the current pod. Replace the SSH target with the current
RunPod Connect line values:

```powershell
scp -P <port> -i ~/.ssh/runpod_ed25519 `
  "$env:APPDATA\rclone\rclone.conf" `
  root@<ip>:/workspace/rclone/rclone.conf
```

On the pod:

```bash
chmod 600 /workspace/rclone/rclone.conf
export PATH=/workspace/bin:$PATH
export RCLONE_CONFIG=/workspace/rclone/rclone.conf
rclone lsd gdrive:
```

## Pull Images From Google Drive

Use `copy` for routine movement. It copies new/changed files and does not
mirror deletions.

Pull a training dataset:

```bash
export PATH=/workspace/bin:$PATH
export RCLONE_CONFIG=/workspace/rclone/rclone.conf

mkdir -p /workspace/datasets/character_01
rclone copy gdrive:Inputs/character_01 /workspace/datasets/character_01 -P
```

Pull reference images:

```bash
mkdir -p /workspace/datasets/character_01_references
rclone copy gdrive:Inputs/character_01_references /workspace/datasets/character_01_references -P
```

## Push Outputs To Google Drive

Push production images:

```bash
rclone copy /workspace/output/prod_batch_001 gdrive:Outputs/prod_batch_001 -P
```

Push a training run folder:

```bash
rclone copy /workspace/output/runs/character_01_flux2_001 gdrive:Outputs/runs/character_01_flux2_001 -P
```

Push only samples during a long run:

```bash
rclone copy /workspace/output/runs/character_01_flux2_001/samples \
  gdrive:Outputs/runs/character_01_flux2_001/samples -P
```

## Use From Bash Scripts

At the top of any RunPod shell script that uses Google Drive:

```bash
#!/usr/bin/env bash
set -euo pipefail

export PATH=/workspace/bin:$PATH
export RCLONE_CONFIG=/workspace/rclone/rclone.conf

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found. Install it to /workspace/bin/rclone first." >&2
  exit 1
fi

if [ ! -f "$RCLONE_CONFIG" ]; then
  echo "rclone config missing: $RCLONE_CONFIG" >&2
  exit 1
fi
```

Then call `rclone copy ... -P`.

## Safety Rules

- The `gdrive` remote is pinned to `Flux Prod` (see Scope section). Never
  unset `root_folder_id`, never add unpinned Drive remotes, never touch
  folders outside `Flux Prod`.
- Use `rclone copy` by default.
- Use `rclone sync` only after `--dry-run` and only when you want deletions on
  one side to be mirrored to the other side.
- Keep `/workspace/rclone/rclone.conf` private.
- If the config leaks, revoke rclone access from the Google Account security
  page and create a new config.
- Do not store `rclone.conf` in GitHub.
