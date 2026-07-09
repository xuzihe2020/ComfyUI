# Face Identity Consistency & FLUX.2 LoRA Training — Learning Notes

Research thread summary, 2026-07-08. Topic: getting strong face/identity consistency in
FLUX.2 workflows — how SDXL-era tools (FaceDetailer, IPAdapter-FaceID, PuLID) work, why
FLUX.2 needed a different approach, and the full plan for a face-specialist character
LoRA plus a personal style checkpoint.

Related docs: `detail-refinement-and-style-conversion-techniques.md`,
`runpod-ai-toolkit-gdrive-hf-workflow.md`, `vn-image-generation-passes.md`.

---

## 1. The identity-strength spectrum

All identity techniques sit on one spectrum, ordered by strength and cost:

1. **Reference tokens (FLUX.2 `ReferenceLatent`)** — zero training, appearance copying,
   no enforcement knob. What our current workflows use. Weakest.
2. **Identity adapters (IPAdapter-FaceID / InstantID / PuLID)** — pretrained ArcFace-based
   injection with a strength dial. Zero per-character cost. Middle.
3. **Character LoRA** — identity baked into the weights; changes the model's prior itself.
   Per-character training cost (~1 evening, ~$1 of GPU). Strongest. **The chosen endgame.**

Key principle discovered along the way: **more information ≠ stronger constraint.**
ReferenceLatent hands the model thousands of raw appearance tokens but nothing forces it
to use them; FaceID hands it 4 tokens of distilled, pose-invariant identity WITH an
enforcement knob — and wins on identity.

---

## 2. SDXL FaceDetailer vs FLUX.2 masked reference-inpaint

### FaceDetailer (Impact Pack) mechanics

Detect → crop → zoom → repaint → paste loop:

1. YOLO face detection (`UltralyticsDetectorProvider`) + optional SAM mask refine → SEGS
2. Crop each face with `crop_factor` ~3 (context for lighting/blend)
3. **Zoom**: upscale crop so the face fills `guide_size` (512–768) — repaint at the
   resolution where the model is actually good at faces
4. Masked img2img at **denoise 0.3–0.5** — identity survives from the input pixels
5. Decode, downscale, feathered paste-back

Exists because of an SDXL weakness: 4-channel VAE at 8× downsample → a 100px face is
~12 latent px; the UNet can't render small faces in-frame.

### Our FLUX.2 workflows (dev/vn_face_*) invert every choice

- Detection only makes the **mask** ("detection decides WHERE, FLUX.2 decides WHAT")
- **Full-frame** sampling with `SetLatentNoiseMask` — no crop/zoom/paste
- **Denoise 0.8** — input pixels intentionally do NOT carry identity
- Identity from 3–5 chained `ReferenceLatent` → `FluxKontextMultiReferenceLatentMethod`

### Why the architectures force different designs

| | SDXL | FLUX.2 |
|---|---|---|
| Denoise semantics | eps-prediction: 0.4 ≈ "regenerate texture, keep structure" | flow matching, shifted sigmas: <0.7 barely changes, real repaint needs 0.8+ |
| Identity mechanism | none native → IPAdapter/InstantID ecosystem | reference latents are native sequence tokens |
| Small faces | terrible (4-ch VAE) → FaceDetailer exists | better (fat latents) but token budget still caps identity |
| Guidance | real CFG 5–8 + negative prompts | distilled, cfg ~1.2, no real negatives |

The SD trick "polish face at denoise 0.45, identity survives free" **does not transfer**
to FLUX — at 0.45 FLUX barely changes; at 0.8 pixel identity is gone. Hence FLUX moved
identity out of the pixels and into reference tokens.

FaceDetailer itself is model-agnostic — reference workflows run it with FLUX pipes at
~0.45 as a *subtle polish* pass (same number ≠ same effect across model families).

---

## 3. How IPAdapter-FaceID really works (SDXL)

Three components, frozen base model:

1. **Image encoder**: CLIP ViT → 4 tokens (base) / 16 tokens (Plus, via perceiver
   resampler). The entire reference is compressed to a tiny semantic summary.
2. **Decoupled cross-attention** — the actual invention. Every cross-attn layer gets a
   second parallel K/V projection pair trained for image tokens:
   `out = Attn(Q, K_text, V_text) + weight × Attn(Q, K_img, V_img)`
   Naive concat of image tokens into text context was ablated and lost. The `weight`
   scalar is the strength dial; only ~22M adapter params train.
3. **FaceID variants swap the encoder** — the key insight: CLIP embeddings are NOT
   identity-discriminative. ArcFace (InsightFace) embeddings are metric-learned to be
   pose/lighting/expression-invariant for identity. **Identity strength came from the
   encoder, not the injection mechanism.**
   - FaceID-PlusV2: ArcFace + CLIP blend, companion LoRA
   - InstantID: ArcFace + keypoint ControlNet (spatial face pose lock)
   - PuLID: + contrastive alignment training → identity injection without polluting
     style/prompt-following (the anti-pollution property is the TRAINING RECIPE, not
     the architecture)

### vs FLUX.2 ReferenceLatent

| | SDXL + FaceID | FLUX.2 ReferenceLatent |
|---|---|---|
| Pathway | parallel cross-attn (one-way) | joint bidirectional self-attention |
| Information | 4–16 tokens, ArcFace-distilled | thousands of raw VAE tokens |
| Invariance | precomputed (pose/lighting factored out) | entangled — model must factor it per-sample |
| Strength control | weight + start/end scheduling | **none** |
| Training locus | small adapter, works across finetunes | baked into base pretraining, fixed |

Why our consistency was weak: denoise 0.8 (pixels carry nothing) + face gets ~150 tokens
of a 1024 frame + references are pose/lighting-entangled + no enforcement dial.

---

## 4. ComfyUI-PuLID-Flux2 (community port) assessment

Repo: <https://github.com/iFayens/ComfyUI-PuLID-Flux2> — first PuLID for FLUX.2,
Klein 4B/9B native-trained weights (v1/v2 on HF). Read the source (581-line single file).

**Same recipe as SDXL FaceID**: InsightFace antelopev2 (ArcFace 512-d) + EVA-CLIP face
crop → mini-IDFormer → 4 ID tokens → per-block cross-attn correction added to the
residual stream with a strength knob.

**Red flags found in code** (calibrate expectations before investing):
- Corrections are L2-normalized per token then scaled by hardcoded depth factors
  (8.0→3.0 double, 6.5→1.8 single) — hand-tuned magnitudes, every token (background
  included) gets an equal-norm nudge
- PuLID's anti-pollution training recipe likely NOT reproduced → behaves more like a
  rough FaceID than true PuLID
- Dev-32B path is fake: dimension mismatch → randomly initialized projection + fresh
  random injector at runtime. Only Klein-native (dim 4096) weights are real
- `strict=False` loading, `__create_new__` returns an untrained adapter that "works"
- Single-block patch adds correction to text tokens too (official FLUX.1 PuLID slices
  image tokens only)
- One reference image only, no scheduling, no attention mask

Verdict: mechanism is sound, engineering is young. Worth a cheap trial vs the
5-reference baseline; composes with masked-inpaint flow. If adopted: fork + manifest per
repo custom-node policy (needs insightface, open_clip, antelopev2 + EVA-CLIP + weights).

---

## 5. Interim pipelines (pre-LoRA)

### Hybrid: FLUX.2 base draw + SDXL FaceID face repaint

Proven community pattern (ran all through the FLUX.1 era). Pixel-space handoff decouples
the models completely; SDXL's weaknesses (hands, poses) are excluded by construction —
only a 768–1024px face crop is repainted with the mature FaceID stack.

- Face pass: YOLO+SAM SEGS → FaceDetailer with SDXL pipe + IPAdapter FaceID-PlusV2
  (weight 0.8–1.0), guide_size 768–1024, denoise 0.5–0.6, crop_factor ~3
- **Main risk = texture seam** (different render character). Mitigations in order:
  crop context, mask feathering, **final low-denoise (0.15–0.25) FLUX.2 unify pass over
  the full frame** (re-renders texture through one model without touching identity —
  FLUX barely changes below 0.8), matched-aesthetic SDXL finetune
- **Doubles as the LoRA dataset factory**: mass-produce identity-consistent images
  across poses/lighting → training data for the FLUX.2 character LoRA

### Stacking rules: LoRA + FaceID together?

Usually redundant — a good LoRA saturates identity; two slightly-different identity
targets fight (cousin face, waxy skin, frozen expressions — ArcFace embeds near-neutral
identity and fights expression prompts). Three cases where stacking earns its place:

1. Undertrained/dataset-limited LoRA (likely early versions) — FaceID as corrector
2. Out-of-distribution angles/lighting (matters less for controlled repaint crops)
3. Inverse trick: drop LoRA weight to shed baked style, FaceID holds likeness

Rule: **FaceID as low-weight corrector (0.3–0.5), never co-driver.** Needing >0.6 with
the LoRA active = retrain the LoRA. On FLUX.2, prefer LoRA + ReferenceLatent stacking
first (both native, zero pollution); PuLID-Flux2 port is last resort.

**Decide by measurement**: ArcFace cosine similarity of outputs vs canonical reference
set. Also the dataset QA tool (filter training candidates) and checkpoint-selection
metric.

---

## 6. Face-specialist character LoRA — the plan

Use case: masked face repaint only (base image has similar face shape via references but
wrong identity). Don't need body/hair/generalization → train on face close-ups only,
slight overfit acceptable ("same face every time" is the success criterion here).
This is deliberate **distribution matching**: train ≈ deploy.

### The four conditions

1. **Scale match (structural).** LoRAs are weight-delta adjustments fit to the
   activations the training data produced; networks are NOT scale-invariant. Close-up
   training (~3,000 tokens/face) + full-frame inference (200px face ≈ 12×12 ≈ 150
   tokens) = LoRA fires weakly AND the canvas can't express identity (ArcFace needs
   ~112px; identity = mid-frequency geometry). **Fix: crop-and-zoom repaint** — detect →
   crop ×2.5–3 context → upscale to ~training res → masked repaint → feathered stitch
   (FaceDetailer with the LoRA pipe, or InpaintCrop/InpaintStitch around the custom
   chain). Repaint AFTER any 2K/4K upscale stage. Specialist LoRA is only safe when
   deployment is guaranteed in-distribution — crop-zoom is that guarantee.
2. **Boundary learning.** Train head-and-shoulders framing (~20% of set), not
   nose-to-chin crops — repaint must blend at jawline/hairline/ears. Handle "don't care
   about hair" via variation + captioning, not cropping: **whatever varies and is
   captioned stays promptable; whatever is constant and uncaptioned fuses into the
   identity.** Same for expressions.
3. **Integration (the real overfit cost).** Overfitting bakes rendering conditions
   (lighting/grade/texture) → pasted-on faces. Vary lighting deliberately (caption it);
   keep the low-denoise unify pass. Photocopy effect: curate generated data hard —
   pairwise ArcFace across the training set, cull outliers BEFORE training.
4. **Stacking**: keep 2–3 ReferenceLatent face refs active during repaint as a free
   native corrector.

### Dataset numbers

- **40–60 curated images** (range 30–80). Coverage grid, sampled not factorial:
  ~5 head angles × ~4 expressions × ~3 lighting setups (+hair variants)
- <25 = axes fuse into identity; >100 generated = photocopy risk, not information
- **Resolution**: train at the highest resolution sources are NATIVELY sharp at
  (800 native beats upscaled 1024 — upscalers bake their texture signature).
  800 vs 1024 ≈ marginal (identity is mid-frequency); can match repaint res to training
- **Mixed aspect ratios fine** — trainers bucket by aspect at a target area (~1MP);
  don't force squares; consistent AREA and consistent face-fraction matter
- **Backgrounds: DO NOT segment to white.** Constant background fuses in → white-void
  bias, cutout halos, flat lighting, poor scene integration (fatal for inpaint use).
  Want varied natural backgrounds, different per image, soft DoF, one caption clause.
  Cull images with other people/faces
- **Rank 16–32** (identity is low-dimensional; higher rank = overfit surface)

### Training run parameters (Klein 9B, ai-toolkit)

- `black-forest-labs/FLUX.2-klein-base-9B`, **`timestep_type: linear`** (biggest
  likeness lever per the 50-run study), lr 1e-4, adamw8bit, batch 1, rank 32,
  **1,800–2,000 steps** (30–60 epochs on 50 imgs = the intended slight overfit)
- Save every 250 steps; pick winner by ArcFace + integration on fixed prompts,
  NOT training loss (loss ≈ uninformative for likeness)
- Iterate: train v1 → measure → fix dataset gaps → retrain. 2–3 loops beat one
  perfectly-planned run (each loop = an evening, ~$1)

---

## 7. Training infrastructure

### Base model (critical)

**`black-forest-labs/FLUX.2-klein-base-9B`** — the undistilled training checkpoint
("preserving complete training signal... ideal for LoRA training").

- **Gated repo**: accept FLUX Non-Commercial license in browser BEFORE pod time; the
  pod's hf_token must belong to the accepting account (else 401 mid-download, on meter)
- ~35–40 GB download (9B transformer bf16 + Qwen3-8B TE + VAE) — pull once to the
  network volume
- NOT: `FLUX.2-klein-9B` (distilled inference), ComfyUI fp8 single-files, mixes, Dev 32B
- ⚠️ Runbook §16 references `FLUX.2-Klein-dev` — wrong; needs updating

### Why train on base, not on moody_desire (despite deploying there)

Distilled models are bad training substrates: standard flow-matching gradients are
noisy/inconsistent against distillation-shaped weights, and training pushes them off the
distilled manifold (breaks few-step behavior). FLUX.1 precedent: schnell training failed,
community built "de-distilled" checkpoints. BFL shipping a separate base = the vendor's
answer. Also: base-trained LoRA is portable across all mixes; all recipes validated on
base. Capture deployment-match safely via (a) the post-hoc sweep on moody, (b) letting
moody generate part of the dataset (style harmonization through data, not substrate).

### Base ≠ deployment: LoRA transfer shift

A LoRA is a weight delta optimized against base weights. Applied to a mix
(base + distill + finetune drift), the same delta expresses differently: **best strength
shifts** (0.8 or 1.2 instead of 1.0) and **best checkpoint can shift** (overfit ckpt may
fight the mix's own face prior; lighter ckpt may integrate better). Same reason civitai
LoRAs carry per-checkpoint strength notes.
→ Read *trends* from in-training samples (rendered on base); make *decisions* from a
post-hoc checkpoint × strength sweep on the deployment model.

### Trainer: ai-toolkit (fork `xuzihe2020/ai-toolkit`)

Landscape (2026-07): ai-toolkit = first-class Klein support + official RunPod template
(**keep**). OneTrainer = faster/stronger likeness but Klein 9B in unmerged fork.
musubi-tuner = solid CLI alternative. SimpleTuner = best W&B logging. Fizgig = LoRA
post-ops (profile/repair/extract). fal.ai = hosted, no custom eval.

**Monitoring architecture: sidecar, not fork surgery.** Trainer emits fixed-seed sample
grids (`sample_every: 250`, `walk_seed: false`); a standalone `fluxlab eval-run` script
walks `samples/`, computes ArcFace cosine vs canonical refs, writes scoreboard CSV +
step-vs-likeness curves. Zero coupling to trainer internals; frozen prompts/seeds/refs
make runs cross-comparable.

**Web UI over SSH** (LocalForward 8675): good enough as live monitor — dataset browser
w/ caption editing, config GUI, queue, logs, loss chart (ignore), samples view,
mid-run checkpoint download. The samples view IS the steps × weights grid: rows =
sample events, columns = sample entries; duplicate the same prompt at
`network_multiplier` 0.6/0.8/1.0/1.2 (runbook §17). Caveats: sampling pauses training
(~1–2 min/event on 9B — budget ~10–15 min per run), grid is forward-only and fixed at
config time, no cross-run compare, no metrics overlay.
**Post-hoc sweep** = prodgen_runner manifest sweeping lora_name × strength_model, fixed
seed/prompt, on the deployment mix, scored by the sidecar.

### Hardware & cost

- **VRAM**: 9B LoRA fits <24 GB (4090 = community sweet spot); 5090 32 GB comfortable
- **32 GB buys** (ranked): disable gradient checkpointing (~1.3–1.4× per-step speedup —
  the real lever), bf16 base without fp8 quantization (cleanest signal; removes a debug
  variable), higher-res bucket headroom, batch 2–4 as a bonus. Batch is NOT a 4×
  speedup: batch-1 already saturates the card; big batches don't help 50-image likeness
  convergence (gradient noise isn't the bottleneck)
- **Speed**: ~1–2 s/it at 1024² → 2,000 steps ≈ 35–70 min (800² ≈ 0.6×). "Under an
  hour" per BFL
- **RunPod** (2026-07): 4090 $0.34 community / $0.69 secure; 5090 $0.99 secure.
  5090 vs secure-4090: 1.43× price for 1.6–2× work → faster AND cheaper per run.
  All runs <$1 — **overhead dominates** (model download, deps, caching ≈ 30–60 min
  billed, identical on both). → Buy a **persistent network volume** (secure cloud only;
  attach at deploy; ~$7/mo per 100 GB); pick 5090 for iteration speed + clean config

### Pre-flight (validate on the Mac, before any pod bills)

1. Eval sidecar end-to-end (InsightFace runs on CPU/Apple Silicon) against existing refs
2. Dataset + captions in final ai-toolkit form (images + sibling .txt) + ArcFace
   pre-filter of the set itself
3. Complete job config YAML in `configs/` (repo id, linear timesteps, rank/lr/steps,
   sample block, save cadence)
4. Sync + pin the ai-toolkit fork; no `git pull` on the pod
5. Accept the HF gate; verify token account
6. First pod session = 100–200-step smoke run (~$0.10) proving download → cache →
   sample → sync → eval, then queue the real run in the same session

---

## 8. Own checkpoint: Tier-2 style LoRA + bake  ← chosen path

How community checkpoints are really made: **Tier 1** true full finetunes (rare; 10k+
imgs, 48–80 GB VRAM or multi-GPU, days, $100s–1000s). **Tier 2** big style LoRA/LoKr
merged permanently into weights (solo-creator standard; looks like a finetune, costs an
evening). **Tier 3** pure merges of existing models + LoRAs (most "mixes";
moody_desire_**mix**_v**30** = merge recipe, 30th iteration; ComfyUI ModelMerge nodes,
CPU, minutes; the skill is evaluation discipline).

### Style dataset = character rules inverted

- **Style constant, everything else varies** (subjects, compositions, genres, scenes).
  #1 failure: the "style" is actually 3–4 styles → LoRA learns the average. Audit first
- Whatever doesn't vary fuses in (portraits-only set → style refuses landscapes)
- **Captions describe content, never style** — strip aesthetic adjectives from VLM
  captions (VLMs love describing aesthetics; that's exactly what must be absorbed)
- **No trigger word** — destiny is an always-on baked checkpoint
- 200–2,000 images, rank 64–128 (LoKr = v2 experiment; plain LoRA first for tool
  compatibility), 5k–20k steps (start ~8k, save every 500), batch 2–4 now useful
  (large diverse set = gradient noise matters), same base-9B + runbook
- Eval = fixed style battery (10–15 prompts across genres, fixed seeds), not ArcFace.
  Overfit signs: compositional sameness, subject bleed, prompt control loss

### The bake

`W_new = W_mix + s·(B·A)` per touched layer — permanent, CPU, minutes.

1. Find the marriage at inference first (style LoRA × mixes × strength ladder in Comfy)
2. Bake via ComfyUI (mix → LoRA at s → CheckpointSave) or safetensors script.
   ⚠️ **fused-QKV loader gotcha applies with teeth**: a mis-applied merge is permanent —
   use the loader path known to apply Klein LoRAs correctly
3. **Verify**: fixed-seed A/B baked-checkpoint vs mix+runtime-LoRA → near-identical
4. **Record the recipe** (mix version, LoRA file, strength, date) next to the artifact

**Lineage rule for merging**: checkpoint↔checkpoint merges stay within the same weight
neighborhood (distilled mixes with distilled mixes). Never checkpoint-merge base-9B with
a distilled mix (different manifolds → breaks few-step sampling). **LoRA deltas are the
bridge**: train on base, apply/bake into distilled-land. Character LoRA stays runtime
(swaps per project); style gets baked (always-on foundation).

---

## 9. Master sequencing

1. **Now (Mac, free)**: eval sidecar; dataset generation + ArcFace curation; job config;
   fork pin; HF gate; runbook §16 fix. Optional cheap trials: PuLID-Flux2, SDXL hybrid
   face pass (also = dataset factory)
2. **Character LoRA v1** (RunPod 5090, ~1 evening): smoke run → 2k-step run → sweep on
   moody → measure → iterate v2/v3
3. **Deploy**: crop-zoom masked repaint with LoRA (+ReferenceLatent corrector), after
   upscale stage; FaceID/PuLID only as low-weight corrector if measurement demands
4. **Style LoRA** (dataset curation is the long pole — start collecting now)
5. **Bake** personal checkpoint; character LoRA rides on top at runtime

---

## 10. Sources

- PuLID-Flux2 port: <https://github.com/iFayens/ComfyUI-PuLID-Flux2>
- ai-toolkit: <https://github.com/ostris/ai-toolkit> (fork: xuzihe2020/ai-toolkit)
- Base model: <https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B>
- BFL Klein training docs: <https://docs.bfl.ml/flux_2/flux2_klein_training>
- BFL "LoRA under 60 minutes": <https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora>
- Klein 9B high-likeness settings: <https://www.runcomfy.com/trainer/ai-toolkit/ai-toolkit-flux-klein-9b-high-likeness-character-training>
- 50-run parameter study: <https://medium.com/@calvinherbst/50-flux-2-klein-lora-training-runs-dev-and-klein-to-see-what-config-parameters-actually-matter-3196e4f64fd5>
- Klein 16GB VRAM limits: <https://www.runcomfy.com/trainer/ai-toolkit/flux-2-klein-16gb-vram-training>
- SimpleTuner FLUX.2 quickstart: <https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md>
- musubi-tuner Klein guide: <https://bitgamma.github.io/ai-blog/blog/musubi-tuner/>
- ai-toolkit UI walkthrough: <https://medium.com/diffusion-doodles/how-to-train-a-lora-ostris-ai-toolkit-44216331056e>
- PuLID vs InstantID vs FaceID: <https://myaiforce.com/pulid-vs-instantid-vs-faceid/>
- RunPod pricing: <https://www.runpod.io/pricing>
- Fizgig (LoRA post-ops): <https://github.com/shootthesound/Fizgig>
