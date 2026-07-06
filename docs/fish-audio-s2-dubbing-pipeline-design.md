# Fish Audio S2 Dubbing Pipeline Design

This document designs a local-first audio dubbing pipeline for short video
clips, using ComfyUI infrastructure wherever it provides useful orchestration,
preview, model loading, and output handling.

The initial target is not a fully automated production dubbing system. The first
target is a controlled test loop:

```text
short source clip
  -> source audio / reference voice
  -> transcript and translation inputs
  -> Fish Audio S2 voice-cloned target speech
  -> preview and saved audio
  -> optional mux back onto the clip
```

Once voice quality is proven on real clips, the pipeline can grow into:

```text
video clip
  -> audio extraction
  -> ASR with timestamps
  -> speaker segmentation
  -> translation and line adaptation
  -> emotion/style tagging
  -> segment-level Fish S2 synthesis
  -> duration alignment
  -> mixdown
  -> lip sync or audio mux
```

## Goals

- Generate high-quality Chinese or English dubbed speech from Japanese source
  dialogue.
- Preserve speaker timbre using short reference audio from the source clip.
- Preserve or approximate emotional delivery using Fish S2 inline control tags.
- Keep ComfyUI as the preview and model execution surface where practical.
- Keep brittle production logic, such as timeline assembly and validation, in
  repo-controlled scripts.
- Reuse this repo's existing patterns:
  - custom nodes are declared in `custom_nodes.manifest.json`,
  - installation behavior lives in `scripts/install_custom_nodes.py`,
  - saved UI workflows stay read-only,
  - API runners convert UI workflows to API prompts at runtime,
  - workflow patching is done by stable node names, not numeric node IDs.

## Non-Goals For The First Milestone

- Fully automatic speaker diarization.
- Fully automatic translation quality control.
- Full lip-sync video regeneration.
- Long movie or episode processing.
- Training or fine-tuning Fish S2.
- Editing installed custom node source under `custom_nodes/`.

These can be added after the initial Fish S2 voice quality and workflow
stability are proven.

## Main Model Choice

Use Fish Audio S2 through the ComfyUI custom node:

```text
https://github.com/Saganaki22/ComfyUI-FishAudioS2
```

This node is preferred over older Fish Speech wrappers because it is built for
Fish Audio S2-Pro and exposes the features needed for dubbing:

- `Fish S2 TTS`
- `Fish S2 Voice Clone TTS`
- `Fish S2 Multi-Speaker TTS`
- `Fish S2 Multi-Speaker Split TTS`
- native ComfyUI `AUDIO` inputs and outputs
- voice cloning from 10-30 second reference audio
- inline emotion/prosody tags
- multi-speaker conversation generation
- per-speaker audio isolation for later lip-sync workflows
- S2-Pro and S2-Pro-FP8 model options

Useful references:

- Fish S2 ComfyUI node:
  `https://github.com/Saganaki22/ComfyUI-FishAudioS2`
- Fish Speech upstream:
  `https://github.com/fishaudio/fish-speech`
- Fish S2 technical report:
  `https://arxiv.org/abs/2603.08823`

## Repository Integration

### Custom Node Manifest

Add the Fish S2 custom node to `custom_nodes.manifest.json`:

```json
{
  "name": "ComfyUI-FishAudioS2",
  "folder": "ComfyUI-FishAudioS2",
  "repo": "https://github.com/Saganaki22/ComfyUI-FishAudioS2",
  "reason": "Provides Fish Audio S2-Pro TTS, voice cloning, multi-speaker dialogue, inline emotion/prosody tags, and per-speaker audio outputs for local dubbing workflows.",
  "platforms": ["linux", "windows"]
}
```

Do not install or modify this custom node manually as the durable fix path. The
manifest is the source of truth.

### Installer Changes

Update `scripts/install_custom_nodes.py` so Fish S2 dependencies are installed
before ComfyUI startup. This is important because the repo policy says custom
node dependencies should be present before ComfyUI imports custom nodes.

Recommended installer additions:

```python
ALWAYS_FIX_DEPENDENCIES = {
    ...
    "ComfyUI-FishAudioS2",
}
```

Fish S2 dependency notes:

- The node documentation says `descript-audio-codec` and
  `descript-audiotools` should be installed with `--no-deps` to avoid forcing
  incompatible `protobuf` constraints into the shared ComfyUI environment.
- The node may auto-install these at startup, but this repo should prefer
  installer-controlled dependency setup.
- Optional accelerators should be best-effort, never fatal.

Suggested extra dependency handling:

```python
EXTRA_PIP_DEPENDENCIES = {
    ...
    "ComfyUI-FishAudioS2": [
        "descript-audio-codec --no-deps",
        "descript-audiotools>=0.7.2 --no-deps",
        "flatten-dict",
        "importlib-resources",
        "julius",
        "randomname",
        "ffmpy",
        "argbind",
    ],
}
```

The current installer represents pip packages as simple strings, so package
entries with flags need either:

- a new installer structure for per-package `pip_args`, or
- a small Fish-specific post-install function.

Prefer the second option if the change should stay narrow:

```text
install_fish_audio_s2_dependencies(python_bin)
```

That helper can run:

```bash
python -m pip install descript-audio-codec --no-deps
python -m pip install "descript-audiotools>=0.7.2" --no-deps
python -m pip install flatten-dict importlib-resources julius randomname ffmpy argbind
```

Optional accelerator handling can mirror the SeedVR2 pattern:

- Windows:
  - `triton-windows`
  - `sageattention` best-effort
  - `bitsandbytes` best-effort for 16-18 GB quantized modes
- Linux:
  - `triton`
  - `sageattention --no-build-isolation`
  - `flash-attn --no-build-isolation` best-effort
  - `bitsandbytes`

Do not make accelerator installation fatal. Fish S2 can fall back to SDPA.

### Model Paths

The user's external model base is:

```text
C:\Users\Tony Xu\workspace\comfyui_models
```

Fish S2 models should not be assumed to live under the repo-local
`models/` directory.

Add a Fish S2 model key to `extra_model_paths.yaml` if the node uses ComfyUI
folder path registration for `fishaudioS2`:

```yaml
  fishaudioS2: fishaudioS2
```

Preferred external layout:

```text
C:\Users\Tony Xu\workspace\comfyui_models\fishaudioS2\s2-pro
C:\Users\Tony Xu\workspace\comfyui_models\fishaudioS2\s2-pro-fp8
```

If the custom node does not honor `extra_model_paths.yaml`, use a repo-controlled
installer or setup note to place the model where the node expects it. Do not
hard-code local absolute paths into workflow JSON.

## Pipeline Architecture

### Phase 0: Clip Preparation

Use the existing video sampler in clip mode.

Example:

```powershell
.\.venv\Scripts\python.exe tools\video_sampler\main.py "path\to\source.mp4" `
  -o tmp\dubbing_clips `
  --mode clip `
  --video-format mp4 `
  --clips "0:01:00-0:01:20,0:02:10-0:02:30"
```

This produces short clips that still contain audio. Short clips are easier to
inspect, retry, and debug than full episodes.

### Phase 1: Manual Fish S2 Voice Clone Test

Purpose:

- Prove Fish S2 quality on real source audio.
- Prove ComfyUI can load reference audio and save synthesized output.
- Avoid translation, ASR, and alignment complexity at first.

Inputs:

- short source clip
- a clean 10-30 second reference audio sample for one speaker
- reference transcript in the original language
- target text in Chinese or English
- optional Fish S2 inline tags

ComfyUI workflow:

```text
LoadAudio(reference speaker audio)
  -> Fish S2 Voice Clone TTS
       model_path
       text
       reference_text
       language
       precision
       attention
       temperature
       top_p
       repetition_penalty
       seed
       keep_model_loaded
  -> PreviewAudio
  -> SaveAudioAdvanced
```

Saved workflow path:

```text
user/default/workflows/prod/audio/fish_s2_voice_clone_test.json
```

Important node naming convention:

- Use exact registered backend node names in `type`.
- Use exact registered backend node names in `properties["Node name for S&R"]`
  when present.
- Do not add custom `title` values unless there is a deliberate runner target.

If runner patching needs stable names, use a small number of explicit titles:

- `Load Reference Audio`
- `Fish S2 Voice Clone TTS`
- `Preview Generated Audio`
- `Save Generated Audio`

### Phase 2: Scripted Single-Line Runner

Create:

```text
scripts/workflows/run_fish_s2_voice_clone.py
```

The runner should follow the pattern already used by:

```text
scripts/workflows/run_img2img_refine.py
scripts/workflows/run_flux2_max_lora_references.py
```

Responsibilities:

- Load saved UI workflow JSON read-only.
- Audit graph structure before converting.
- Convert UI workflow to API prompt.
- Patch values by stable saved node names, not node IDs.
- Queue prompt through ComfyUI API.
- Wait for `/history/{prompt_id}` unless `--no-wait` is set.
- Write debug API prompt JSON when requested.
- Write per-job log JSON under `logs/fish_s2_dubbing/`.

The runner should not:

- edit the workflow JSON,
- write into `custom_nodes/`,
- assume models are under repo-local `models/`,
- invoke a custom node installer.

Suggested CLI:

```powershell
.\.venv\Scripts\python.exe scripts\workflows\run_fish_s2_voice_clone.py `
  --input-json data\dubbing\fish_s2_jobs.json `
  --workflow user\default\workflows\prod\audio\fish_s2_voice_clone_test.json `
  --server http://127.0.0.1:8188 `
  --output-prefix fish_s2_dubbing/test01 `
  --prompt-out-dir tmp\fish_s2_api_prompts
```

Suggested job JSON:

```json
[
  {
    "id": "clip_001_speaker_a_zh",
    "reference_audio": "tmp/dubbing_clips/clip_001_speaker_a.wav",
    "reference_text": "Original reference transcript here.",
    "target_language": "zh",
    "target_text": "[softly] Translated target line here. [short pause] Continue.",
    "model_path": "s2-pro-fp8",
    "seed": 12345,
    "temperature": 0.7,
    "top_p": 0.7,
    "repetition_penalty": 1.2,
    "output_prefix": "fish_s2_dubbing/test01/clip_001_speaker_a_zh"
  }
]
```

Required fields:

- `reference_audio`
- `target_text`

Recommended fields:

- `id`
- `reference_text`
- `target_language`
- `output_prefix`
- `seed`

Optional model fields:

- `model_path`
- `precision`
- `attention`
- `temperature`
- `top_p`
- `repetition_penalty`
- `max_new_tokens`
- `keep_model_loaded`

### Phase 3: Multi-Speaker Preview Workflow

Purpose:

- Generate short dialogue with multiple cloned voices in one ComfyUI run.
- Quickly judge whether Fish S2 speaker consistency is usable.
- Produce combined and isolated speaker tracks for future lip-sync tests.

ComfyUI workflow:

```text
LoadAudio(speaker 1 reference)
LoadAudio(speaker 2 reference)
...
  -> Fish S2 Multi-Speaker Split TTS
       [speaker_1]: ...
       [speaker_2]: ...
  -> PreviewAudio(combined)
  -> SaveAudioAdvanced(combined)
  -> SaveAudioAdvanced(speaker_1_audio)
  -> SaveAudioAdvanced(speaker_2_audio)
```

Saved workflow path:

```text
user/default/workflows/prod/audio/fish_s2_multispeaker_test.json
```

This workflow is useful for creative preview, but segment-level synthesis is
still preferred for production timing.

### Phase 4: Segment-Level Production Pipeline

For video dubbing, generate one audio segment per spoken line or phrase.

Canonical data model:

```json
{
  "clip_id": "clip_001",
  "source_video": "tmp/dubbing_clips/clip_001.mp4",
  "segments": [
    {
      "id": "seg_0001",
      "speaker": "speaker_a",
      "start": 1.2,
      "end": 3.8,
      "source_language": "ja",
      "source_text": "Original Japanese text.",
      "target_language": "zh",
      "target_text": "[angry but controlled] Target translated line.",
      "reference_audio": "data/dubbing/voices/speaker_a_ref.wav",
      "reference_text": "Transcript matching the reference audio.",
      "emotion": {
        "label": "angry",
        "intensity": 0.65,
        "tags": ["angry but controlled", "low voice"]
      }
    }
  ]
}
```

Segment synthesis loop:

```text
for each segment:
  build Fish S2 target text with inline tags
  queue Fish S2 voice clone workflow
  save segment wav/flac
  measure generated duration
  compare to source segment duration
  retry or mark for alignment
```

Mixdown loop:

```text
create silent base track with source clip duration
place generated segment audio at segment.start
apply optional gain normalization
apply optional time-stretch for small duration mismatches
export dubbed_track.wav
```

Mux loop:

```text
source video + dubbed_track.wav -> dubbed video
```

This assembly logic should be a script, not a ComfyUI workflow. It needs
repeatable timeline math, duration checks, retry policy, and logging.

## ASR And Translation Strategy

### Initial Manual Mode

Start with manually written or manually corrected transcripts and translations.
This avoids debugging ASR, translation, and TTS at the same time.

Manual job authoring is enough for:

- voice quality tests,
- language quality tests,
- emotion tag tests,
- reference audio selection tests.

### ASR Mode

Candidate custom node:

```text
https://github.com/yuvraj108c/ComfyUI-Whisper
```

Useful nodes/features:

- Whisper transcription
- word and segment timestamps
- SRT export
- subtitle preview

Alternative:

- repo script using `faster-whisper`

Prefer the script route for production because the transcript becomes structured
pipeline data, not only a ComfyUI node output.

Suggested later file:

```text
scripts/audio_dubbing/transcribe_clips.py
```

Outputs:

```text
data/dubbing/<clip_id>/transcript.json
data/dubbing/<clip_id>/transcript.srt
```

### Translation And Adaptation

Translation should be a repo-controlled script or external LLM call at first,
not a ComfyUI graph.

Reasons:

- Must preserve speaker labels.
- Must preserve timestamps.
- Must shorten or expand lines for target duration.
- Must emit Fish S2 style tags.
- Must keep a diffable JSON artifact.

Suggested later file:

```text
scripts/audio_dubbing/adapt_transcript.py
```

Input:

```text
transcript.json
target_language
style policy
```

Output:

```text
dubbing_plan.json
```

The adaptation prompt should ask for:

- faithful meaning,
- natural target-language dialogue,
- no overlong lines,
- emotion tags only where they help,
- explicit preservation of segment IDs and speaker IDs.

## Emotion And Prosody Control

Fish S2 supports inline tags. The pipeline should treat these as data, not as
random prose appended by hand.

Useful tag categories:

- emotion:
  - `[excited]`
  - `[sad]`
  - `[angry]`
  - `[surprised]`
  - `[delight]`
- volume:
  - `[whisper]`
  - `[low voice]`
  - `[volume up]`
  - `[shouting]`
- pacing:
  - `[pause]`
  - `[short pause]`
  - `[inhale]`
  - `[exhale]`
  - `[sigh]`
- vocalization:
  - `[laugh]`
  - `[chuckle]`
  - `[tsk]`
  - `[clearing throat]`
- tone:
  - `[professional broadcast tone]`
  - `[sarcastic tone]`
  - `[speaking slowly and clearly]`
  - `[pitch up]`
  - `[pitch down]`

Recommended style tag policy:

```text
source emotion label + intensity
  -> one leading style tag
  -> optional mid-line tag only where the original performance changes
```

Example:

```text
[angry but controlled] I told you not to follow me. [short pause] Why are you here?
```

Avoid over-tagging. Too many tags can make output unstable or theatrical.

## Audio Reference Selection

Reference quality is critical.

Preferred reference clip:

- 10-30 seconds.
- Single speaker.
- Minimal background music.
- Minimal overlapping dialogue.
- Similar emotional range to the desired output.
- Contains natural speech, not only shouting or whispering.

For a recurring character, keep curated reference assets:

```text
data/dubbing/voices/
  speaker_a/
    neutral.wav
    neutral.txt
    angry.wav
    angry.txt
    soft.wav
    soft.txt
  speaker_b/
    neutral.wav
    neutral.txt
```

The `*.txt` transcript should match the reference audio. Fish S2 can work
without reference text, but matching text generally improves cloning stability.

## Output Layout

Suggested local output layout:

```text
data/dubbing/
  clips/
    clip_001.mp4
    clip_002.mp4
  voices/
    speaker_a/
      neutral.wav
      neutral.txt
  plans/
    clip_001.dubbing_plan.json
  generated/
    clip_001/
      seg_0001.wav
      seg_0002.wav
      dubbed_track.wav
      clip_001.dubbed.mp4
  reports/
    clip_001.quality_report.json
```

ComfyUI output layout:

```text
output/
  fish_s2_dubbing/
    test01/
      clip_001_speaker_a_zh_00001.flac
```

Debug output:

```text
logs/
  fish_s2_dubbing/
    run_YYYYMMDD_HHMMSS/
      clip_001_speaker_a_zh.request.json
      clip_001_speaker_a_zh.history.json
```

## Workflow Runner Design

The runner should include local copies or a shared helper for:

- `audit_workflow_graph(workflow)`
- `convert_ui_workflow_to_api_prompt(workflow)`
- `unique_named_node(workflow, name, role)`
- `api_node(prompt, ui_node)`
- `post_json(server, path, payload)`
- `wait_for_history(server, prompt_id, timeout_s)`

The existing `scripts/workflows/run_img2img_refine.py` already contains most of
this logic. If this starts to duplicate too much code, create a shared helper:

```text
scripts/workflows/comfy_api_utils.py
```

Do that only when the second audio runner or another image runner needs it.

### Saved Node Name Contract

For a voice clone workflow runner:

```python
WORKFLOW_NODE_NAME_LOAD_REFERENCE_AUDIO = "Load Reference Audio"
WORKFLOW_NODE_NAME_FISH_S2_VOICE_CLONE = "Fish S2 Voice Clone TTS"
WORKFLOW_NODE_NAME_SAVE_AUDIO = "Save Generated Audio"
```

Patchable input names will depend on the custom node's registered sockets. The
expected names from the Fish S2 node documentation are:

- `model_path`
- `text`
- `language`
- `device`
- `precision`
- `attention`
- `max_new_tokens`
- `chunk_length`
- `temperature`
- `top_p`
- `repetition_penalty`
- `seed`
- `keep_model_loaded`
- `compile_model`
- `reference_audio`
- `reference_text`

The runner should validate every patched input exists before queueing.

### Dry Run Behavior

`--dry-run` should:

- load and audit the workflow,
- build all API prompts,
- write debug API prompt JSON if requested,
- print planned jobs,
- not call ComfyUI `/prompt`.

This mirrors the existing Flux runners and saves GPU time.

## Duration Alignment

Fish S2 is not guaranteed to match the original segment duration exactly.

Use three alignment levels:

### Level 1: Translation Length Control

During translation/adaptation, prefer lines that fit the original time span.
This is the best quality control because it changes the text naturally.

### Level 2: Retry With Tags

If generated audio is too long or too short, retry with tags:

- too long:
  - `[speaking quickly]`
  - shorter translation
  - fewer pauses
- too short:
  - `[speaking slowly and clearly]`
  - add a natural pause

### Level 3: Time-Stretch

Apply signal processing only for small corrections.

Suggested policy:

- within 5 percent: leave as-is or lightly stretch
- 5-12 percent: time-stretch
- above 12 percent: regenerate or rewrite line

Avoid aggressive time-stretching. It damages naturalness.

## Final Video Output

For first tests, mux generated audio onto the original clip without lip-sync:

```text
source clip video stream + dubbed audio track -> test output mp4
```

Later, add visual lip-sync:

- MuseTalk for fast talking-face/lip-sync tests.
- LatentSync for heavier diffusion-based lip-sync experiments.
- Per-speaker Fish S2 split outputs can feed multi-character lip-sync workflows.

## Quality Evaluation

Every test clip should be scored manually at first.

Suggested scorecard:

```json
{
  "clip_id": "clip_001",
  "model": "s2-pro-fp8",
  "speaker_similarity": 4,
  "naturalness": 4,
  "target_language_pronunciation": 5,
  "emotion_match": 3,
  "timing_fit": 3,
  "noise_or_artifacts": 2,
  "notes": "Good timbre, angry line too theatrical, segment 3 too long."
}
```

Scale:

- 5: excellent
- 4: usable with minor edits
- 3: promising but needs retry
- 2: weak
- 1: reject

Do not judge only by one clip. Use at least:

- one calm dialogue,
- one emotional line,
- one fast exchange,
- one whisper or low-volume line,
- one noisy/reference-challenging line.

## Implementation Milestones

### Milestone 1: Fish S2 Installed And Loadable

Changes:

- Add `ComfyUI-FishAudioS2` to `custom_nodes.manifest.json`.
- Add installer dependency handling.
- Add optional accelerator handling.
- Add external model path support if needed.

Acceptance:

- User runs `scripts/install_custom_nodes.py`.
- ComfyUI starts with Fish S2 nodes visible.
- No dependency install happens during ComfyUI import/startup.

### Milestone 2: Manual ComfyUI Workflow

Changes:

- Create `user/default/workflows/prod/audio/fish_s2_voice_clone_test.json`.

Acceptance:

- Load reference audio.
- Type target line.
- Preview generated audio.
- Save generated audio.

### Milestone 3: Scripted Voice Clone Runner

Changes:

- Add `scripts/workflows/run_fish_s2_voice_clone.py`.
- Add sample job JSON under a docs or examples path.

Acceptance:

- `--dry-run` writes API prompt JSON.
- Normal run queues ComfyUI and saves audio.
- Runner patches workflow by saved node name.

### Milestone 4: Segment Dubbing Plan

Changes:

- Add `scripts/audio_dubbing/` package.
- Add `dubbing_plan.json` schema.
- Add segment-level synthesis orchestration.

Acceptance:

- Multiple segments generate into separate audio files.
- Segment outputs include duration metadata.

### Milestone 5: Mix And Mux

Changes:

- Add mixdown script.
- Add mux script or integrate mux into the same CLI.

Acceptance:

- Generated segment audio becomes a single dubbed track.
- Dubbed track is muxed into the source clip.

### Milestone 6: ASR And Translation

Changes:

- Add Faster-Whisper or ComfyUI-Whisper transcription path.
- Add translation/adaptation script.

Acceptance:

- Source clip produces timestamped transcript JSON.
- Transcript becomes target-language dubbing plan.

### Milestone 7: Lip Sync

Changes:

- Evaluate MuseTalk and LatentSync.
- Add one lip-sync workflow or external script.

Acceptance:

- Dubbed audio can drive mouth movement on a short test clip.

## Risk Register

### Dependency Conflicts

Risk:

- Fish S2 dependencies may conflict with ComfyUI or other custom nodes.

Mitigation:

- Install fragile dependencies in repo installer.
- Use `--no-deps` for packages known to pull incompatible protobuf versions.
- Keep Fish S2 install gated to Windows/Linux.
- Avoid manual local patches under `custom_nodes/`.

### VRAM Requirements

Risk:

- Full S2-Pro needs about 24 GB VRAM.

Mitigation:

- Use `s2-pro-fp8` on 20 GB+ Ada/Blackwell GPUs.
- Use BNB INT8/NF4 when VRAM is tighter.
- Keep model loaded only for batches where it helps.

### Reference Audio Quality

Risk:

- Bad reference audio causes poor cloning.

Mitigation:

- Curate references per speaker.
- Store reference transcripts.
- Avoid overlapping dialogue and background music.

### Timing Drift

Risk:

- Generated speech duration does not fit original video timing.

Mitigation:

- Segment-level generation.
- Translation length control.
- Retry policy.
- Light time-stretch only for small differences.

### Over-Tagged Speech

Risk:

- Too many style tags make speech unstable or unnatural.

Mitigation:

- One leading tag per segment by default.
- Add mid-line tags only for meaningful changes.
- Track tags in structured data.

### Workflow Drift

Risk:

- Saved ComfyUI workflow nodes change and runner patching breaks.

Mitigation:

- Patch by stable saved node name.
- Validate patched inputs exist.
- Audit graph before conversion.
- Keep API prompt debug output.

## Suggested First Test

1. Clip a 15-30 second Japanese conversation sample.
2. Extract or create one clean 10-30 second reference audio per speaker.
3. Manually transcribe the reference audio.
4. Manually write one Chinese and one English target line.
5. Run the Fish S2 voice clone workflow.
6. Try three tag variants:
   - no tag,
   - one emotion tag,
   - one emotion tag plus one pacing tag.
7. Score speaker similarity, naturalness, pronunciation, emotion, and timing.

Only after that should the pipeline add ASR and translation automation.

