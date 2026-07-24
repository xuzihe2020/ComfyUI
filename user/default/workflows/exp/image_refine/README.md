# Old Photo Image Refine Workflows

All five workflows accept one image through `LoadImage`, preview the original
and processed result, and save the processed image under `output/image_refine/`.

| Workflow | Purpose |
|---|---|
| `01_denoise_cleanup.json` | Real-ESRGAN cleanup followed by 0.25× Lanczos resize, preserving the input dimensions |
| `02_seedvr2_restore.json` | SeedVR2 detail reconstruction, deblurring, LAB color correction, and upscale to a 2048 px short edge |
| `03_face_restore.json` | GFPGAN face-only restoration |
| `04_final_polish.json` | Mild sharpening |
| `05_old_photo_restoration_e2e.json` | Runs all four stages in order and previews every intermediate result |

## Required custom nodes

- `ComfyUI-SeedVR2_VideoUpscaler`
- `comfyui_gfpgan`

Both are managed in `custom_nodes.manifest.json`. Run the repository installer
yourself before opening the workflows:

```powershell
python .\scripts\install_custom_nodes.py
```

## Required external models

Models remain under `C:\Users\Tony Xu\workspace\comfyui_models`:

- `upscale_models\RealESRGAN_x4plus.pth`
- `SEEDVR2\seedvr2_ema_7b_sharp_fp16.safetensors`
- `SEEDVR2\ema_vae_fp16.safetensors`
- `face_restoration\GFPGANv1.4.pth`
- `face_detection\detection_Resnet50_Final.pth`
- `face_detection\parsing_parsenet.pth`

The SeedVR2 files are already present. The GFPGAN node can download its three
weights on first use; the installer links its model directories to the
external model root so those downloads do not enter the repository.

Stage 2 defaults are conservative for still images: batch size 1, fixed seed
42, LAB color correction, input noise 0.05, tiled VAE encode/decode, and CPU
offloading with 18 swapped DiT blocks.
