#!/usr/bin/env python3
"""Proof-of-concept: Fish Audio S2 voice cloning from a tagged transcript.

Second key PoC from docs/fish-audio-s2-dubbing-pipeline-design.md (the first
is poc_asr_ser_translate.py). Given a Chinese transcript with inline Fish S2
tags and a few ~20 s reference audio pieces of one person, synthesize every
line in the cloned voice.

No ComfyUI server and no workflow JSON are involved. The ComfyUI-FishAudioS2
custom node is a thin wrapper around a bundled, ComfyUI-independent
fish_speech package (every `import comfy` / `folder_paths` in the node is
inside try/except with a standalone fallback). This script imports that
engine directly and wires the same call chain the node's
FishS2VoiceCloneTTS.generate() uses:

    launch_thread_safe_queue(...)      # text2semantic LLaMA worker thread
    load_decoder_model(...)            # modded-DAC vocoder
    TTSInferenceEngine(...).inference(ServeTTSRequest(...))

The ComfyUI workflow route (Milestones 2-3 in the design doc) stays the plan
for interactive preview; this script is the batch/PoC path and keeps the
model hot across all segments in one process.

Engine source resolution (first hit wins):
 1. --fish-src PATH  (node repo root, its fish_speech_src/, or any dir that
    contains a `fish_speech` package)
 2. custom_nodes/ComfyUI-FishAudioS2/fish_speech_src   (installed node)
 3. tmp/ComfyUI-FishAudioS2/fish_speech_src            (scratch clone)

For a Mac/PoC run without the node installed:
    git clone https://github.com/Saganaki22/ComfyUI-FishAudioS2 tmp/ComfyUI-FishAudioS2

Model weights: --model-dir must hold the s2-pro checkpoint (config.json +
weights) with the DAC decoder (codec.pth or firefly-gan-*.pth) in the same
folder or its parent. Pass --download to fetch --hf-repo (default
fishaudio/s2-pro, ~10 GB) into --model-dir first.

Dependencies: repo requirements.txt covers the plain ones ("repo-specific:
audio dubbing pipeline" section). Two packages must be installed manually
with --no-deps (their protobuf<5 pin conflicts with the shared venv):
    .venv/bin/python -m pip install descript-audio-codec --no-deps
    .venv/bin/python -m pip install "descript-audiotools>=0.7.2" --no-deps

Device notes: CUDA bfloat16 is the real target (~24 GB VRAM for full
s2-pro). MPS (float16) and CPU work but are experimental and slow
(seconds per token) — fine for a one-line smoke test on the Mac.

Examples:
    # all segments of an ASR PoC transcript, voice refs from a folder
    .venv/bin/python scripts/audio/poc_fish_s2_clone.py \
        data/dubbing/clip_001/transcript.json \
        --voice-dir data/dubbing/voices/speaker_b

    # single ad-hoc line, quick smoke test, first 20 tokens of quality
    .venv/bin/python scripts/audio/poc_fish_s2_clone.py \
        --text "[angry] 我不是让你别跟着我吗？ [short pause] 你为什么还在这里？" \
        --ref-audio data/dubbing/voices/speaker_b/neutral.wav \
        --ref-text-file data/dubbing/voices/speaker_b/neutral.txt \
        --limit 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DECODER_CANDIDATES = (
    "codec.pth",
    "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    "decoder.pth",
    "vocoder.pth",
)
REFERENCE_AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
REFERENCE_SWEET_SPOT_SECONDS = 30.0
OVERLAP_TOLERANCE_SECONDS = 0.25
LANGUAGES = ("auto", "zh", "en", "ja", "ko", "es", "pt", "fr", "de", "ru")


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Engine source + model resolution
# ---------------------------------------------------------------------------

def locate_fish_src(explicit: Path | None) -> Path:
    """Return the directory that contains the `fish_speech` package."""
    candidates: list[Path] = []
    if explicit:
        explicit = explicit.resolve()
        candidates += [explicit, explicit / "fish_speech_src"]
    candidates += [
        REPO_ROOT / "custom_nodes" / "ComfyUI-FishAudioS2" / "fish_speech_src",
        REPO_ROOT / "tmp" / "ComfyUI-FishAudioS2" / "fish_speech_src",
    ]
    for candidate in candidates:
        if (candidate / "fish_speech" / "__init__.py").is_file():
            return candidate
    raise SystemExit(
        "fish_speech engine source not found. Either install the "
        "ComfyUI-FishAudioS2 custom node (custom_nodes.manifest.json + "
        "scripts/install_custom_nodes.py, see the design doc), or clone it "
        "for the PoC:\n"
        "  git clone https://github.com/Saganaki22/ComfyUI-FishAudioS2 "
        "tmp/ComfyUI-FishAudioS2\n"
        "or pass --fish-src /path/to/ComfyUI-FishAudioS2"
    )


def find_decoder(model_dir: Path) -> Path:
    for search in (model_dir, model_dir.parent):
        for name in DECODER_CANDIDATES:
            path = search / name
            if path.is_file():
                return path
    raise SystemExit(
        f"DAC decoder checkpoint not found near {model_dir} "
        f"(expected one of: {', '.join(DECODER_CANDIDATES)}). "
        "It ships with the model download."
    )


def ensure_model(model_dir: Path, hf_repo: str, download: bool) -> Path:
    has_weights = model_dir.is_dir() and (model_dir / "config.json").is_file()
    if not has_weights:
        if not download:
            raise SystemExit(
                f"model not found: {model_dir}\n"
                f"Download it with --download (HF repo: {hf_repo}), or point "
                "--model-dir at an existing s2-pro checkpoint folder."
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("huggingface_hub is required for --download.")
        log(f"[model] downloading {hf_repo} -> {model_dir}")
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(model_dir),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.h5"],
        )
    find_decoder(model_dir)  # fail early if the vocoder is missing
    return model_dir


def resolve_device(requested: str) -> tuple[str, "object"]:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    dtype = {
        "cuda": torch.bfloat16,
        "mps": torch.float16,
        "cpu": torch.float32,
    }[requested]
    return requested, dtype


def resolve_precision(choice: str, device_dtype):
    import torch

    if choice == "auto":
        return device_dtype
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[choice]


def load_engine(model_dir: Path, device: str, dtype, compile_model: bool):
    """Standalone version of the node loader: queue + decoder + engine."""
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

    log(f"[engine] loading LLaMA from {model_dir} on {device} ({dtype})")
    started = time.perf_counter()
    result = launch_thread_safe_queue(
        checkpoint_path=str(model_dir),
        device=device,
        precision=dtype,
        compile=compile_model,
    )
    llama_queue, llama_thread = result if isinstance(result, tuple) else (result, None)

    decoder_ckpt = find_decoder(model_dir)
    log(f"[engine] loading decoder from {decoder_ckpt}")
    decoder_model = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(decoder_ckpt),
        device=device,
    )
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=dtype,
        compile=compile_model,
    )
    log(f"[engine] ready in {time.perf_counter() - started:.1f}s")
    return engine, llama_queue, llama_thread


def shutdown_engine(llama_queue, llama_thread) -> None:
    try:
        llama_queue.put(None)  # worker-loop exit sentinel
        if llama_thread is not None:
            llama_thread.join(timeout=10)
    except Exception:
        pass  # daemon thread; process exit cleans up anyway


# ---------------------------------------------------------------------------
# Reference audio
# ---------------------------------------------------------------------------

def audio_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def reference_text_for(audio_path: Path) -> str:
    for suffix in (".txt", ".lab"):
        text_path = audio_path.with_suffix(suffix)
        if text_path.is_file():
            return text_path.read_text(encoding="utf-8").strip()
    return ""


def collect_references(args: argparse.Namespace) -> list[dict]:
    """Return [{path, text, seconds}] for the clone references."""
    if args.ref_audio:
        text = ""
        if args.ref_text_file:
            text = Path(args.ref_text_file).read_text(encoding="utf-8").strip()
        elif args.ref_text:
            text = args.ref_text
        else:
            text = reference_text_for(args.ref_audio)
        refs = [{"path": args.ref_audio.resolve(), "text": text}]
    else:
        voice_dir = args.voice_dir.resolve()
        if not voice_dir.is_dir():
            raise SystemExit(f"--voice-dir not found: {voice_dir}")
        audio_files = sorted(
            p for p in voice_dir.iterdir()
            if p.suffix.lower() in REFERENCE_AUDIO_SUFFIXES
        )
        if not audio_files:
            raise SystemExit(f"no reference audio in {voice_dir}")
        refs = [
            {"path": p, "text": reference_text_for(p)}
            for p in audio_files[: args.max_refs]
        ]
        skipped = len(audio_files) - len(refs)
        if skipped > 0:
            log(f"[refs] using first {len(refs)} of {len(audio_files)} files "
                f"(--max-refs {args.max_refs}); {skipped} skipped")

    total = 0.0
    for ref in refs:
        ref["seconds"] = audio_seconds(ref["path"])
        duration = f"{ref['seconds']:.1f}s" if ref["seconds"] else "?"
        has_text = "with transcript" if ref["text"] else "NO TRANSCRIPT"
        log(f"[refs] {ref['path'].name} ({duration}, {has_text})")
        total += ref["seconds"] or 0.0
        if not ref["text"]:
            log("[refs]   warning: cloning is more stable with a matching "
                "reference transcript (same-stem .txt/.lab file)")
    if total > REFERENCE_SWEET_SPOT_SECONDS:
        log(f"[refs] warning: {total:.0f}s of reference audio exceeds the "
            f"{REFERENCE_SWEET_SPOT_SECONDS:.0f}s sweet spot; long references "
            "make the prompt very large and can destabilize generation")
    return refs


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def load_jobs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Return (jobs, transcript_meta). Each job: id, text, start, end."""
    if args.text:
        return [{"id": "line_0001", "text": args.text, "start": None, "end": None}], {}

    data = json.loads(args.transcript.read_text(encoding="utf-8"))
    jobs = []
    for seg in data.get("segments", []):
        text = seg.get("tagged_target_text") or seg.get("target_text")
        if not text:
            log(f"[jobs] skipping {seg.get('id')}: no target/tagged text")
            continue
        jobs.append(
            {
                "id": seg.get("id") or f"seg_{len(jobs) + 1:04d}",
                "text": text,
                "start": seg.get("start"),
                "end": seg.get("end"),
            }
        )
    if not jobs:
        raise SystemExit(
            "transcript has no segments with target_text/tagged_target_text; "
            "run poc_asr_ser_translate.py with translation enabled first."
        )
    return jobs, data


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(engine, request_cls, jobs: list[dict], references,
               out_dir: Path, args: argparse.Namespace) -> int:
    import soundfile as sf

    failures = 0
    for index, job in enumerate(jobs, start=1):
        text = job["text"]
        if args.language != "auto":
            text = f"[{args.language}] {text}"

        request = request_cls(
            text=text,
            references=references,
            reference_id=None,
            max_new_tokens=args.max_new_tokens,
            chunk_length=args.chunk_length,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            seed=args.seed,
            use_memory_cache="on",  # encode the references once, not per line
            streaming=False,
            format="wav",
        )

        log(f"[tts] ({index}/{len(jobs)}) {job['id']}: {job['text'][:60]}")
        started = time.perf_counter()
        sample_rate, audio = None, None
        error = None
        try:
            for result in engine.inference(request):
                if result.code == "error":
                    error = str(result.error)
                    break
                if result.code == "final":
                    sample_rate, audio = result.audio
        except Exception as exc:
            error = str(exc)

        elapsed = time.perf_counter() - started
        if audio is None:
            failures += 1
            job.update(status=f"failed: {error or 'no audio produced'}")
            log(f"[tts]   FAILED after {elapsed:.1f}s: {error}")
            continue

        wav_path = out_dir / f"{job['id']}.wav"
        sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")
        duration = len(audio) / sample_rate
        job.update(
            status="ok",
            wav=str(wav_path.relative_to(REPO_ROOT)) if wav_path.is_relative_to(REPO_ROOT) else str(wav_path),
            sample_rate=sample_rate,
            duration=round(duration, 3),
            generation_seconds=round(elapsed, 1),
            rtf=round(elapsed / duration, 2) if duration else None,
        )
        log(f"[tts]   {duration:.2f}s audio in {elapsed:.1f}s "
            f"(RTF {job['rtf']}) -> {wav_path.name}")
    return failures


def check_overlap_budgets(jobs: list[dict]) -> None:
    """Design-doc timing rule: a segment must not run past the next start."""
    timed = [j for j in jobs if j.get("start") is not None and j.get("status") == "ok"]
    for current, following in zip(timed, timed[1:]):
        budget = following["start"] - current["start"]
        overrun = current["duration"] - budget
        current["overlap_budget"] = round(budget, 3)
        current["budget_overrun"] = round(overrun, 3)
        if overrun > OVERLAP_TOLERANCE_SECONDS:
            log(f"[timing] {current['id']} overruns its overlap budget by "
                f"{overrun:.2f}s (audio {current['duration']:.2f}s, budget "
                f"{budget:.2f}s) — candidate for shorter translation, "
                "[speaking quickly], or time-stretch")


def mixdown(jobs: list[dict], out_path: Path) -> bool:
    """Start-anchored placement of the generated segments on one track."""
    import numpy as np
    import soundfile as sf

    placed = [j for j in jobs if j.get("start") is not None and j.get("status") == "ok"]
    if not placed:
        return False
    sample_rate = placed[0]["sample_rate"]
    total = max(j["start"] + j["duration"] for j in placed) + 0.5
    track = np.zeros(int(total * sample_rate), dtype=np.float32)
    for job in placed:
        audio, _ = sf.read(job["wav"] if Path(job["wav"]).is_absolute()
                           else str(REPO_ROOT / job["wav"]), dtype="float32")
        begin = int(job["start"] * sample_rate)
        end = min(begin + len(audio), len(track))
        track[begin:end] += audio[: end - begin]
    peak = float(np.abs(track).max() or 0.0)
    if peak > 1.0:
        track /= peak  # only overlapping segments can push past full scale
    sf.write(str(out_path), track, sample_rate, subtype="PCM_16")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("transcript", type=Path, nargs="?",
                        help="transcript.json from poc_asr_ser_translate.py "
                             "(omit when using --text).")
    parser.add_argument("--text", default=None,
                        help="Single ad-hoc line (with inline tags) instead of "
                             "a transcript.")
    parser.add_argument("--voice-dir", type=Path,
                        default=REPO_ROOT / "data" / "dubbing" / "voices" / "speaker_b",
                        help="Folder with reference audio + same-stem .txt/.lab "
                             "transcripts (default: data/dubbing/voices/speaker_b).")
    parser.add_argument("--ref-audio", type=Path, default=None,
                        help="Explicit single reference audio file "
                             "(overrides --voice-dir).")
    parser.add_argument("--ref-text", default=None,
                        help="Reference transcript string for --ref-audio.")
    parser.add_argument("--ref-text-file", type=Path, default=None,
                        help="Reference transcript file for --ref-audio.")
    parser.add_argument("--max-refs", type=int, default=1,
                        help="How many reference pieces to use from --voice-dir "
                             "(default: 1; ~20s total is the sweet spot).")
    parser.add_argument("--fish-src", type=Path, default=None,
                        help="Path to ComfyUI-FishAudioS2 (or its "
                             "fish_speech_src/).")
    parser.add_argument("--model-dir", type=Path,
                        default=REPO_ROOT / "models" / "fishaudioS2" / "s2-pro",
                        help="s2-pro checkpoint folder "
                             "(default: models/fishaudioS2/s2-pro).")
    parser.add_argument("--hf-repo", default="fishaudio/s2-pro",
                        help="HuggingFace repo for --download "
                             "(default: fishaudio/s2-pro).")
    parser.add_argument("--download", action="store_true",
                        help="Download --hf-repo into --model-dir if missing.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: <transcript dir>/fish_s2/).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Synthesize only the first N segments (0 = all).")
    parser.add_argument("--no-mixdown", action="store_true",
                        help="Skip the start-anchored single-track mixdown.")
    parser.add_argument("--language", default="zh", choices=LANGUAGES,
                        help="Language hint prepended as [xx] (default: zh; "
                             "'auto' disables the hint).")
    parser.add_argument("--device", default="auto",
                        choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--precision", default="auto",
                        choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the LLaMA (slow first run, then fast).")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=0,
                        help="0 = no limit (model decides).")
    parser.add_argument("--chunk-length", type=int, default=200,
                        help="Iterative synthesis chunk length, 100-300.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if bool(args.text) == bool(args.transcript):
        parser.error("pass exactly one of: transcript.json or --text")
    if not 100 <= args.chunk_length <= 300:
        parser.error("--chunk-length must be in 100-300 (engine constraint)")
    return args


def main() -> None:
    args = parse_args()

    fish_src = locate_fish_src(args.fish_src)
    log(f"[engine] fish_speech source: {fish_src}")
    sys.path.insert(0, str(fish_src))

    model_dir = ensure_model(args.model_dir.resolve(), args.hf_repo, args.download)

    jobs, transcript = load_jobs(args)
    if args.limit > 0:
        jobs = jobs[: args.limit]
    log(f"[jobs] {len(jobs)} line(s) to synthesize")

    reference_files = collect_references(args)

    if args.out_dir:
        out_dir = args.out_dir
    elif args.transcript:
        out_dir = args.transcript.resolve().parent / "fish_s2"
    else:
        out_dir = REPO_ROOT / "data" / "dubbing" / "_adhoc" / "fish_s2"
    out_dir.mkdir(parents=True, exist_ok=True)

    device, device_dtype = resolve_device(args.device)
    dtype = resolve_precision(args.precision, device_dtype)
    if device != "cuda":
        log(f"[engine] note: {device} is experimental for Fish S2 and slow; "
            "CUDA is the real target")

    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    references = [
        ServeReferenceAudio(audio=ref["path"].read_bytes(), text=ref["text"])
        for ref in reference_files
    ]

    engine, llama_queue, llama_thread = load_engine(model_dir, device, dtype, args.compile)
    try:
        failures = synthesize(engine, ServeTTSRequest, jobs, references, out_dir, args)
    finally:
        shutdown_engine(llama_queue, llama_thread)

    check_overlap_budgets(jobs)

    track_path = out_dir / "dubbed_track.wav"
    mixed = False
    if not args.no_mixdown and not args.text:
        mixed = mixdown(jobs, track_path)
        if mixed:
            log(f"[mix] start-anchored track: {track_path}")

    manifest = {
        "clip_id": transcript.get("clip_id"),
        "transcript": str(args.transcript) if args.transcript else None,
        "model_dir": str(model_dir),
        "device": device,
        "precision": str(dtype),
        "language": args.language,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "chunk_length": args.chunk_length,
            "seed": args.seed,
        },
        "references": [
            {"path": str(r["path"]), "seconds": r["seconds"], "text": r["text"]}
            for r in reference_files
        ],
        "mixdown": str(track_path) if mixed else None,
        "segments": jobs,
    }
    manifest_path = out_dir / "fish_s2_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    done = sum(1 for j in jobs if j.get("status") == "ok")
    log("")
    log(f"[done] {done}/{len(jobs)} segments synthesized -> {out_dir}")
    log(f"[done] manifest: {manifest_path}")
    if failures:
        raise SystemExit(f"{failures} segment(s) failed; see manifest for details")


if __name__ == "__main__":
    main()
