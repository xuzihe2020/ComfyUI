#!/usr/bin/env python3
"""Proof-of-concept: video clip -> transcript -> emotion -> Chinese translation.

Implements the first automation experiments from
docs/fish-audio-s2-dubbing-pipeline-design.md:

 1. Extract mono 16 kHz audio from a video clip with ffmpeg.
 2. ASR with faster-whisper. Start timestamps are the authoritative output
    (segment start snaps to the first word timestamp when available); end
    timestamps are informational only.
 3. Sentence-level emotion + audio event detection with the SER stack:
    SenseVoice-Small per segment (emotion label + events such as laughter/BGM),
    optionally emotion2vec+ as a second opinion with a confidence score.
    Disagreement between the two models flags the segment for human review.
 4. Translate segments to Chinese with Grok through ``aigc_shared.llm_client``.
    Auth via XAI_API_KEY from the environment or the
    repo-root .env. Skipped with a warning when no API key is set.

Outputs (under --out-dir, default data/dubbing/<clip stem>/):
    transcript.json   segments with start/end, source/target text, emotion
    transcript.srt    for quick eyeballing in a video player
    audio_16k.wav     the extracted audio used for ASR/SER

Dependencies live in the repo-level requirements.txt (the "repo-specific:
audio dubbing pipeline" section) and install into the ComfyUI venv:
    .venv/bin/python -m pip install -r requirements.txt
ffmpeg must be on PATH (macOS: brew install ffmpeg); the imageio-ffmpeg
requirement provides a bundled fallback binary otherwise.

Example:
    .venv/bin/python scripts/audio/poc_asr_ser_translate.py \
        tmp/dubbing_clips/clip_001.mp4 --emotion2vec

    XAI_API_KEY=... .venv/bin/python scripts/audio/poc_asr_ser_translate.py \
        clip.mp4 --llm-model grok-4.3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lib.envfile import env_value  # noqa: E402
from aigc_shared.llm_client import GrokClient  # noqa: E402

# SenseVoice rich-transcription tokens, e.g. "<|ja|><|ANGRY|><|Speech|>text".
SENSEVOICE_TAG_RE = re.compile(r"<\|([^|]+)\|>")
SENSEVOICE_EMOTIONS = {
    "HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED",
}
SENSEVOICE_EVENTS = {
    "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough",
}

# Mechanical mapping from SER output to the Fish S2 tag taxonomy in the design
# doc. Tag injection stays code, not prose: one leading emotion tag, plus at
# most one vocalization tag from detected events.
EMOTION_TO_TAG = {
    "happy": "delight",
    "sad": "sad",
    "angry": "angry",
    "surprised": "surprised",
    "fearful": "fearful",
    "disgusted": "disgusted",
}
EVENT_TO_TAG = {
    "Laughter": "laugh",
}

MIN_SER_SECONDS = 0.3  # segments shorter than this get no emotion pass
SER_END_PAD_SECONDS = 0.15


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Stage 1: audio extraction
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit(
            "ffmpeg not found on PATH and imageio-ffmpeg is not installed.\n"
            "Install one of them (macOS: brew install ffmpeg)."
        )


def extract_audio(video: Path, wav_path: Path) -> None:
    cmd = [
        find_ffmpeg(), "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    log(f"[extract] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{result.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Stage 2: ASR (faster-whisper)
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> tuple[str, str]:
    if requested == "auto":
        try:
            import torch

            requested = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            requested = "cpu"
    compute_type = "float16" if requested == "cuda" else "int8"
    return requested, compute_type


def run_asr(wav_path: Path, args: argparse.Namespace) -> tuple[list[dict], dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit(
            "faster-whisper is not installed. Run:\n"
            "  .venv/bin/python -m pip install -r requirements.txt"
        )

    device, compute_type = resolve_device(args.device)
    log(f"[asr] loading faster-whisper '{args.whisper_model}' on {device} ({compute_type})")
    model = WhisperModel(args.whisper_model, device=device, compute_type=compute_type)

    language = None if args.language == "auto" else args.language
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=not args.no_word_timestamps,
        condition_on_previous_text=False,
    )

    segments: list[dict] = []
    for index, seg in enumerate(segments_iter, start=1):
        start = seg.start
        # Word timestamps pin the start more tightly than segment boundaries,
        # which matters because start is the mixdown placement anchor.
        if seg.words:
            start = seg.words[0].start
        text = seg.text.strip()
        if not text:
            continue
        segments.append(
            {
                "id": f"seg_{index:04d}",
                "start": round(float(start), 3),
                "end": round(float(seg.end), 3),
                "source_text": text,
                "target_text": None,
                "emotion": None,
                "asr_avg_logprob": round(float(seg.avg_logprob), 3),
                "asr_no_speech_prob": round(float(seg.no_speech_prob), 3),
            }
        )

    asr_info = {
        "engine": "faster-whisper",
        "model": args.whisper_model,
        "device": device,
        "detected_language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "word_timestamps": not args.no_word_timestamps,
    }
    log(f"[asr] {len(segments)} segments, language={info.language} "
        f"(p={info.language_probability:.2f})")
    return segments, asr_info


# ---------------------------------------------------------------------------
# Stage 3: SER stack (SenseVoice + optional emotion2vec+)
# ---------------------------------------------------------------------------

def parse_sensevoice_output(raw_text: str) -> tuple[str | None, list[str]]:
    tokens = SENSEVOICE_TAG_RE.findall(raw_text)
    label = None
    for token in tokens:
        if token in SENSEVOICE_EMOTIONS:
            label = token.lower()
            break
    events = sorted({t for t in tokens if t in SENSEVOICE_EVENTS})
    return label, events


def run_ser(wav_path: Path, segments: list[dict], args: argparse.Namespace) -> dict:
    try:
        from funasr import AutoModel
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(
            f"SER dependencies missing ({exc.name}). Run:\n"
            "  .venv/bin/python -m pip install -r requirements.txt"
        )

    audio, sample_rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    device, _ = resolve_device(args.device)
    funasr_device = "cuda:0" if device == "cuda" else "cpu"

    if args.model_hub == "hf":
        sensevoice_id = "FunAudioLLM/SenseVoiceSmall"
        emotion2vec_id = f"emotion2vec/emotion2vec_plus_{args.emotion2vec_size}"
    else:
        sensevoice_id = "iic/SenseVoiceSmall"
        emotion2vec_id = f"iic/emotion2vec_plus_{args.emotion2vec_size}"

    log(f"[ser] loading SenseVoice ({sensevoice_id}) on {funasr_device}")
    sensevoice = AutoModel(
        model=sensevoice_id, hub=args.model_hub, device=funasr_device,
        disable_update=True,
    )
    emotion2vec = None
    if args.emotion2vec:
        log(f"[ser] loading emotion2vec ({emotion2vec_id})")
        emotion2vec = AutoModel(
            model=emotion2vec_id, hub=args.model_hub, device=funasr_device,
            disable_update=True,
        )

    bgm_segments = 0
    for seg in segments:
        start_sample = max(0, int(seg["start"] * sample_rate))
        end_sample = min(len(audio), int((seg["end"] + SER_END_PAD_SECONDS) * sample_rate))
        clip = audio[start_sample:end_sample]
        if len(clip) < int(MIN_SER_SECONDS * sample_rate):
            continue

        result = sensevoice.generate(
            input=clip, fs=sample_rate, language="auto", use_itn=False,
        )
        label, events = parse_sensevoice_output(result[0]["text"])
        if "BGM" in events:
            bgm_segments += 1

        emotion = {
            "label": label,
            "intensity": None,
            "tags": [],
            "events": events,
            "confidence": None,
            "source": "model",
            "needs_review": False,
        }

        if emotion2vec is not None:
            e2v = emotion2vec.generate(
                input=clip, fs=sample_rate,
                granularity="utterance", extract_embedding=False,
            )[0]
            scores = e2v.get("scores") or []
            labels = e2v.get("labels") or []
            if scores and labels:
                top = max(range(len(scores)), key=scores.__getitem__)
                # emotion2vec labels look like "生气/angry"
                e2v_label = labels[top].split("/")[-1].strip().lower()
                emotion["confidence"] = round(float(scores[top]), 3)
                emotion["emotion2vec_label"] = e2v_label
                if (
                    label and e2v_label not in ("unknown", "other", label)
                ):
                    emotion["needs_review"] = True

        seg["emotion"] = emotion
        seg["tagged_target_text"] = None  # filled after translation

    if bgm_segments:
        log(f"[ser] note: BGM detected in {bgm_segments}/{len(segments)} segments; "
            "if ASR quality suffers, consider a vocal-separation pass")

    return {
        "sensevoice_model": sensevoice_id,
        "emotion2vec_model": emotion2vec_id if args.emotion2vec else None,
        "bgm_segments": bgm_segments,
    }


# ---------------------------------------------------------------------------
# Stage 4: translation (OpenAI-compatible chat API, stdlib only)
# ---------------------------------------------------------------------------

TRANSLATION_SYSTEM_PROMPT = """\
You are a professional dubbing translator. Translate each Japanese dialogue
line into natural, colloquial spoken {target_language_name}.

Rules:
- Faithful meaning, natural dialogue register. Match the given emotion.
- Keep each line roughly the same spoken duration as the source; never make
  it much longer.
- Translate line by line. Never merge, split, drop, or reorder lines.
- Return ONLY strict JSON, no markdown fences, in the shape:
  {{"segments": [{{"id": "...", "target_text": "..."}}]}}
  with exactly one entry per input id.\
"""

TARGET_LANGUAGE_NAMES = {"zh": "Simplified Chinese", "en": "English"}


def parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    return json.loads(content)


def translate_segments(segments: list[dict], args: argparse.Namespace) -> dict:
    api_key = env_value(args.api_key_env)
    if not api_key:
        log(f"[translate] WARNING: {args.api_key_env} is not set (env or repo .env); "
            "skipping translation")
        return {"status": "skipped: no API key"}

    lines = [
        {
            "id": seg["id"],
            "text": seg["source_text"],
            "emotion": (seg.get("emotion") or {}).get("label"),
        }
        for seg in segments
    ]
    target_name = TARGET_LANGUAGE_NAMES.get(args.target_language, args.target_language)
    payload = {
        "model": args.llm_model,
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": TRANSLATION_SYSTEM_PROMPT.format(target_language_name=target_name),
            },
            {"role": "user", "content": json.dumps({"segments": lines}, ensure_ascii=False)},
        ],
    }

    log(f"[translate] {len(lines)} segments -> {target_name} via {args.llm_model}")
    grok = GrokClient(api_key, base_url=args.llm_base_url, timeout=180)
    translated = parse_llm_json(grok.chat_text(payload))

    by_id = {
        entry["id"]: entry.get("target_text")
        for entry in translated.get("segments", [])
        if isinstance(entry, dict) and "id" in entry
    }
    missing = [seg["id"] for seg in segments if seg["id"] not in by_id]
    for seg in segments:
        seg["target_text"] = by_id.get(seg["id"])
    if missing:
        log(f"[translate] WARNING: no translation returned for: {', '.join(missing)}")
    return {
        "status": "ok" if not missing else f"partial: {len(missing)} missing",
        "model": args.llm_model,
        "base_url": args.llm_base_url,
    }


# ---------------------------------------------------------------------------
# Tag injection + outputs
# ---------------------------------------------------------------------------

def build_tagged_text(seg: dict) -> str | None:
    text = seg.get("target_text") or seg.get("source_text")
    if not text:
        return None
    emotion = seg.get("emotion") or {}
    tags: list[str] = []
    label = emotion.get("label")
    if label and label != "neutral":
        tags.append(f"[{EMOTION_TO_TAG.get(label, label)}]")
    for event in emotion.get("events", []):
        if event in EVENT_TO_TAG:
            tags.append(f"[{EVENT_TO_TAG[event]}]")
            break
    return " ".join(tags + [text])


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[dict], srt_path: Path) -> None:
    blocks = []
    for index, seg in enumerate(segments, start=1):
        lines = [seg["source_text"]]
        if seg.get("target_text"):
            lines.append(seg["target_text"])
        emotion = seg.get("emotion") or {}
        if emotion.get("label"):
            marker = emotion["label"]
            if emotion.get("needs_review"):
                marker += " ⚠"
            lines.append(f"({marker})")
        blocks.append(
            f"{index}\n"
            f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n"
            + "\n".join(lines)
        )
    srt_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def print_summary(segments: list[dict]) -> None:
    log("")
    log(f"{'start':>8}  {'emotion':<10} {'conf':<5} {'!':<2} text -> translation")
    log("-" * 100)
    for seg in segments:
        emotion = seg.get("emotion") or {}
        label = emotion.get("label") or "-"
        confidence = emotion.get("confidence")
        conf_text = f"{confidence:.2f}" if confidence is not None else "-"
        flag = "!" if emotion.get("needs_review") else ""
        source = seg["source_text"][:40]
        target = (seg.get("target_text") or "")[:40]
        log(f"{seg['start']:>8.2f}  {label:<10} {conf_text:<5} {flag:<2} {source} -> {target}")
    flagged = sum(1 for s in segments if (s.get("emotion") or {}).get("needs_review"))
    if flagged:
        log(f"\n{flagged} segment(s) flagged for human emotion review (model disagreement)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Source video clip (or audio file).")
    parser.add_argument("--clip-id", default=None, help="Clip id (default: video stem).")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: data/dubbing/<clip id>/).",
    )
    parser.add_argument("--language", default="ja",
                        help="Source language for ASR, or 'auto' (default: ja).")
    parser.add_argument("--target-language", default="zh",
                        help="Translation target (default: zh).")
    parser.add_argument("--whisper-model", default="large-v3",
                        help="faster-whisper model size (default: large-v3; "
                             "use 'small' for quick CPU runs).")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-word-timestamps", action="store_true",
                        help="Disable word timestamps (faster, less precise starts).")
    parser.add_argument("--skip-emotion", action="store_true",
                        help="Skip the SenseVoice/emotion2vec SER stage.")
    parser.add_argument("--skip-translate", action="store_true",
                        help="Skip the LLM translation stage.")
    parser.add_argument("--emotion2vec", action="store_true",
                        help="Also run emotion2vec+ as a second opinion with confidence.")
    parser.add_argument("--emotion2vec-size", default="base", choices=("base", "large"))
    parser.add_argument("--model-hub", default="hf", choices=("hf", "ms"),
                        help="Model download hub for funasr models (default: hf).")
    parser.add_argument("--llm-base-url", default=GrokClient.DEFAULT_BASE_URL,
                        help="OpenAI-compatible API base URL (default: xAI).")
    parser.add_argument("--llm-model", default=GrokClient.DEFAULT_MODEL,
                        help=f"Chat model for translation (default: {GrokClient.DEFAULT_MODEL}).")
    parser.add_argument("--api-key-env", default="XAI_API_KEY",
                        help="Env var holding the API key (default: XAI_API_KEY).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = args.video.resolve()
    if not video.exists():
        raise SystemExit(f"input not found: {video}")

    clip_id = args.clip_id or video.stem
    out_dir = args.out_dir or (REPO_ROOT / "data" / "dubbing" / clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = out_dir / "audio_16k.wav"
    extract_audio(video, wav_path)

    segments, asr_info = run_asr(wav_path, args)
    if not segments:
        raise SystemExit("ASR produced no segments; nothing to do.")

    ser_info: dict = {"status": "skipped"}
    if not args.skip_emotion:
        ser_info = run_ser(wav_path, segments, args)

    translation_info: dict = {"status": "skipped"}
    if not args.skip_translate:
        try:
            translation_info = translate_segments(segments, args)
        except Exception as exc:  # keep ASR/SER results even if the API fails
            log(f"[translate] WARNING: translation failed: {exc}")
            translation_info = {"status": f"failed: {exc}"}

    for seg in segments:
        seg["tagged_target_text"] = build_tagged_text(seg)

    transcript = {
        "clip_id": clip_id,
        "source_video": str(video),
        "source_language": asr_info.get("detected_language", args.language),
        "target_language": args.target_language,
        "asr": asr_info,
        "ser": ser_info,
        "translation": translation_info,
        "segments": segments,
    }
    transcript_path = out_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    write_srt(segments, out_dir / "transcript.srt")

    print_summary(segments)
    log("")
    log(f"[done] transcript: {transcript_path}")
    log(f"[done] srt:        {out_dir / 'transcript.srt'}")


if __name__ == "__main__":
    main()
