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

## Copy-Paste Cookbook

`/workspace/gdrive/` on the volume is a 1:1 mirror of the `Flux Prod` Drive
folder — same structure on both sides (`Inputs/`, `Outputs/`, ...). Work
inside `/workspace/gdrive`, then run the matching command below as-is.

Two rules that explain every command:

- `rclone copy SRC DST` copies the CONTENTS of SRC into DST — to transfer a
  folder under its own name, the destination must end with that folder name.
- Every `rclone copy` is incremental: files identical on both sides
  (size + modtime) are skipped automatically. "Sync only updated files" =
  re-run the same command. `copy` never deletes anything.

Upload ONE file to Drive:

```bash
rclone copy /workspace/gdrive/Inputs/photo.png gdrive:Inputs/ -P
```

Upload ONE folder to Drive:

```bash
rclone copy /workspace/gdrive/Inputs/dataset_01 gdrive:Inputs/dataset_01 -P
```

Upload EVERYTHING (all new/changed files and folders, volume → Drive):

```bash
rclone copy /workspace/gdrive gdrive: -P --transfers 8
```

Download / resync EVERYTHING (all new/changed files and folders, Drive → volume):

```bash
rclone copy gdrive: /workspace/gdrive -P --transfers 8
```

Download ONE folder from Drive:

```bash
rclone copy gdrive:Inputs/dataset_01 /workspace/gdrive/Inputs/dataset_01 -P
```

Preview what a command would transfer without touching anything: add
`--dry-run`.

Push ComfyUI/training outputs that live outside the mirror:
