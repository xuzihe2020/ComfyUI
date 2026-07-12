# Skill: Google Drive ↔ RunPod Volume via rclone

Move image datasets and generated outputs between Google Drive and
`/workspace` on pods. This is for images and run outputs — NOT ComfyUI model
sync (models use `scripts/runpod/model_sync.py` + the volume S3 API).

## SCOPE — PINNED TO "Flux Prod" — HARD RULE

The `gdrive` remote is pinned via `root_folder_id` to the Drive folder
`Flux Prod` (ID `1tuMvBkNNREzBBCHVY-DH6ViiQEiveWEi`). `gdrive:` RESOLVES TO
that folder; the rest of Tony's Drive is not addressable through it.

Agents and scripts MUST NOT unset or change `root_folder_id`, MUST NOT create
additional Drive remotes, and MUST NOT attempt to reach anything outside
`Flux Prod`. The pin was set with:

```bash
rclone config update gdrive root_folder_id 1tuMvBkNNREzBBCHVY-DH6ViiQEiveWEi --non-interactive
```

The OAuth token itself is full-Drive (Google offers no folder-level consent
usable for this workflow), so the pin is the enforced boundary and
`rclone.conf` is a secret: chmod 600, volume + Tony's workstations only,
never in git, never pasted into chat. If it leaks: revoke at
myaccount.google.com → Security → third-party access, re-auth, re-pin.

## Layout

```text
Flux Prod/          <- gdrive: root (pinned)
  Inputs/           <- datasets, reference images (uploaded via Drive web UI)
  Outputs/          <- generated results, run outputs pushed from pods
  Tests/
  screenshots/

/workspace/bin/rclone              # persistent rclone binary (on the volume)
/workspace/rclone/rclone.conf      # Drive token/config; SECRET; chmod 600
/workspace/gdrive/                 # 1:1 mirror of Flux Prod (Inputs/, Outputs/, ...)
/workspace/output/                 # ComfyUI / training outputs
```

`/workspace/gdrive/` mirrors the pinned Drive root with identical structure —
datasets land at `/workspace/gdrive/Inputs/<name>`. CHECK IT FIRST
(`ls /workspace/gdrive`) before inventing destinations or mkdir-ing.

`session_start.sh` exports `PATH=/workspace/bin:$PATH` and
`RCLONE_CONFIG=/workspace/rclone/rclone.conf` — after the session sync,
`rclone` works in any pod shell with no setup.

## First-time install (already done on frankie_the_pug_volume; repeat only on a new volume)

Never `apt-get install rclone` as the durable path (container is disposable).
Binary goes on the volume:

```bash
mkdir -p /workspace/bin /workspace/rclone /workspace/tmp
cd /workspace/tmp
curl -L https://downloads.rclone.org/rclone-current-linux-amd64.zip -o rclone.zip
unzip -o rclone.zip
cp rclone-*-linux-amd64/rclone /workspace/bin/rclone
chmod +x /workspace/bin/rclone
chmod 700 /workspace/rclone
```

Auth (browser required → done on a workstation, never by an agent):

```bash
# on the workstation (Mac/Windows):
rclone config create gdrive drive scope=drive     # Google consent in browser
rclone config update gdrive root_folder_id 1tuMvBkNNREzBBCHVY-DH6ViiQEiveWEi --non-interactive
rclone lsd gdrive:                                # must list ONLY Flux Prod contents
# copy to the volume (agents may run this; conf content stays out of chat):
scp -P <port> -i <key> ~/.config/rclone/rclone.conf root@<ip>:/workspace/rclone/rclone.conf
# on the pod:
chmod 600 /workspace/rclone/rclone.conf
rclone lsd gdrive:                                # connection test
```

OAuth client: our OWN client_id (`...376fdh.apps.googleusercontent.com`,
GCP project `comfyui-models`, Desktop app "rclone", consent screen published
to Production — done 2026-07-11; rclone's shared client retirement no longer
affects us). Production status is what makes refresh tokens indefinite —
NEVER flip the consent screen back to "Testing" (Testing tokens die every
7 days; Tony's hard requirement is zero re-consent, forever). The client
credentials JSON lives outside git on Tony's machines; the working conf
(client + token + Flux Prod pin) is the file at /workspace/rclone/rclone.conf
and on each workstation. New machine setup: copy an existing conf, or fresh
auth MUST pass our client_id/client_secret explicitly (a bare
`rclone config create gdrive drive` would silently use rclone's dying shared
client).

## Copy cookbook (same commands on pod and workstation)

Core semantics — read before copying anything:

- `rclone copy SRC DST` copies the CONTENTS of SRC into DST. To upload folder
  `foo` AS `foo`, the destination must end in `/foo`.
- EVERY `rclone copy` is incremental by default: files identical on both sides
  (size + modtime) are skipped; only new/changed files transfer. "Sync updated
  only" = just re-run the same copy command.
- `copy` never deletes. `rclone sync` deletes destination files to mirror the
  source — never use it for routine transfers (see Rules).
- `-P` = live progress. `--transfers 8` helps with many small files.
  `--dry-run` previews. `--update` skips dest files newer than source.

### Upload one file → Drive

```bash
rclone copy /path/to/photo.png gdrive:Inputs/ -P                 # keeps filename
rclone copyto /path/to/photo.png gdrive:Inputs/renamed.png -P    # exact path/rename
```

### Upload one folder → Drive

```bash
rclone copy /path/to/dataset_01 gdrive:Inputs/dataset_01 -P
```

### Upload many folders/files → Drive

```bash
rclone copy /path/to/parent gdrive:Inputs -P --transfers 8        # whole parent, structure kept
rclone copy /path/to/parent gdrive:Inputs -P --include "run_*/**" # selected items only
```

### Download / resync Drive → local (pod or workstation)

```bash
rclone copy gdrive: /workspace/gdrive -P --transfers 8                        # whole Flux Prod mirror
rclone copy gdrive:Inputs/dataset_01 /workspace/gdrive/Inputs/dataset_01 -P   # one dataset
```

### Push outputs (pod → Drive)

```bash
rclone copy /workspace/output/prod_batch_001 gdrive:Outputs/prod_batch_001 -P
# samples only, during a long training run:
rclone copy /workspace/output/runs/<run>/samples gdrive:Outputs/runs/<run>/samples -P
```

## Rules

- `rclone copy` by default — it never deletes. `rclone sync` only after
  `--dry-run`, only when mirroring deletions is the explicit intent.
- Respect the Flux Prod pin (see SCOPE above) — no exceptions.
- In scripts, guard first:

```bash
export PATH=/workspace/bin:$PATH
export RCLONE_CONFIG=/workspace/rclone/rclone.conf
command -v rclone >/dev/null || { echo "rclone missing from /workspace/bin" >&2; exit 1; }
[ -f "$RCLONE_CONFIG" ] || { echo "rclone config missing" >&2; exit 1; }
```
