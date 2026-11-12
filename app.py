from __future__ import annotations

import os
import json
import zipfile
import re
import time
import wave
import queue
import shutil
import inspect
import tempfile
import threading
import subprocess
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
import torch

# ============================================================
# PATHS AND SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "cactus_subtitle_output"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ON_SPACES = bool(os.environ.get("SPACE_ID"))
IS_LOCAL = not ON_SPACES

# Hugging Face Spaces / ZeroGPU integration. Local Windows does not need spaces.
# A real @spaces.GPU decorator is registered automatically on HF Spaces.
if ON_SPACES:
    try:
        import spaces
    except ImportError:
        raise RuntimeError(
            "Running on Hugging Face Spaces: add `spaces` to requirements.txt "
            "for automatic ZeroGPU support."
        )
else:
    class _LocalSpaces:
        @staticmethod
        def GPU(function):
            return function
    spaces = _LocalSpaces()


# A lightweight function for HF Spaces ZeroGPU registration/startup checks.
@spaces.GPU
def zerogpu_registration():
    return "ZeroGPU registered; CUDA available: " + str(torch.cuda.is_available())


# VAD models are cached between runs; calls run within @spaces.GPU context.
_VAD_MODEL = None
_VAD_UTILS = None
_VAD_DEVICE = None
_VAD_LOCK = threading.RLock()

SAMPLE_RATE = 16000
MAX_CHUNK = 29.0
SEARCH_BACK = 5.0

STOP_EVENT = threading.Event()
JOB_LOCK = threading.Lock()

# ============================================================
# LANGUAGES - 29 TOTAL
# ============================================================
WHISTLE_LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Dutch": "nl",
    "Italian": "it",
    "Polish": "pl",
}

SHRUTI_LANGUAGES = [
    "Hindi", "Marathi", "Tamil", "Bengali", "Telugu", "Gujarati", "Odia",
    "Kannada", "Urdu", "Malayalam", "Assamese", "Sanskrit", "Punjabi",
    "Bodo", "Dogri", "Kashmiri", "Konkani", "Maithili", "Manipuri",
    "Nepali", "Santali", "Sindhi",
]

LANGUAGES = dict(WHISTLE_LANGUAGES)
for name in SHRUTI_LANGUAGES:
    LANGUAGES[name] = "shruti:" + name.lower()

PRIORITY = ["English", "Hindi", "Bengali"]

LANGUAGES = {
    name: LANGUAGES[name]
    for name in PRIORITY + sorted(n for n in LANGUAGES if n not in PRIORITY)
}

# ============================================================
# MODEL DOWNLOAD AND CACHE
# ============================================================
def load_asr_model(language_code, log):
    """Download and load the appropriate Cactus ASR model, caching it locally."""
    from huggingface_hub import HfApi, hf_hub_download
    from cactus import Whistle

    if language_code.startswith("shruti:"):
        name = language_code.split(":", 1)[1]
        repo_id = "udgamlabs/Shruti" if name == "hindi" else f"udgamlabs/Shruti-{name.capitalize()}"
        folder = MODEL_DIR / name
    else:
        repo_id = "Cactus-Compute/whistle"
        folder = MODEL_DIR / "whistle"

    folder.mkdir(parents=True, exist_ok=True)
    log(f"ASR repository: {repo_id}")
    log(f"Model cache: {folder}")
    # Discover the exact .cact filename rather than guessing its capitalization.
    files = HfApi().list_repo_files(repo_id)
    cact_files = [f for f in files if f.lower().endswith(".cact")]
    if not cact_files:
        raise RuntimeError(f"No .cact speech model found in {repo_id}")
    model_file = next((f for f in cact_files if Path(f).stem.lower() in
                       ("whistle", "shruti", "shruti-" + language_code.split(":")[-1])), cact_files[0])

    required = [model_file]
    if "config.json" in files:
        required.append("config.json")

    for filename in required:
        if STOP_EVENT.is_set():
            return None
        destination = folder / filename
        if destination.is_file() and destination.stat().st_size > 0:
            log(f"Cached: {filename}")
            continue
        log(f"Downloading: {filename}")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(folder))
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"Missing downloaded model: {destination}")
        log(f"Downloaded: {filename}")

    if STOP_EVENT.is_set():
        return None
    log("Loading Cactus Whistle ASR runtime...")
    return Whistle(str(folder / model_file))

# ============================================================
# AUDIO PROCESSING
# ============================================================
def convert_audio(source, destination):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(destination),
    ]
    subprocess.run(command, check=True)


def read_wav_info(path):
    with wave.open(str(path), "rb") as wf:
        return {
            "rate": wf.getframerate(),
            "channels": wf.getnchannels(),
            "width": wf.getsampwidth(),
            "frames": wf.getnframes(),
            "duration": wf.getnframes() / wf.getframerate(),
        }


def extract_wav_chunk(source, destination, start, end):
    with wave.open(str(source), "rb") as wf:
        rate = wf.getframerate()
        first = max(0, int(start * rate))
        last = min(wf.getnframes(), int(end * rate))
        wf.setpos(first)
        audio = wf.readframes(max(0, last - first))
        with wave.open(str(destination), "wb") as out:
            out.setnchannels(wf.getnchannels())
            out.setsampwidth(wf.getsampwidth())
            out.setframerate(rate)
            out.writeframes(audio)

# ============================================================
# SILERO VAD
# ============================================================
def _get_silero_vad(device):
    """Use the official torch.hub Silero VAD with local disk and RAM caching."""
    global _VAD_MODEL, _VAD_UTILS, _VAD_DEVICE
    if _VAD_MODEL is None:
        torch.set_num_threads(1)
        print("Loading Silero VAD through torch.hub (cached after first download)...", flush=True)
        _VAD_MODEL, _VAD_UTILS = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _VAD_MODEL.eval()
        _VAD_DEVICE = "cpu"
    if _VAD_DEVICE != device:
        _VAD_MODEL = _VAD_MODEL.to(device)
        _VAD_DEVICE = device
    return _VAD_MODEL, _VAD_UTILS[0]


@spaces.GPU
def detect_speech(wav_path):
    """Run VAD on CUDA when present; gracefully use CPU otherwise."""
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # Do device detection *inside* the decorated call so ZeroGPU allocation
    # is visible on Hugging Face rather than testing CUDA at startup.
    preferred = "cuda" if torch.cuda.is_available() else "cpu"

    with _VAD_LOCK:
        try:
            model, timestamp_fn = _get_silero_vad(preferred)
            samples = torch.from_numpy(audio).to(preferred)
            print(f"Silero VAD device: {preferred}", flush=True)
            return timestamp_fn(
                samples, model, sampling_rate=SAMPLE_RATE, return_seconds=True
            )
        except (RuntimeError, AssertionError) as exc:
            if preferred == "cpu":
                raise
            print(f"Silero VAD CUDA failed ({exc}); falling back to CPU", flush=True)
            model, timestamp_fn = _get_silero_vad("cpu")
            return timestamp_fn(
                torch.from_numpy(audio), model,
                sampling_rate=SAMPLE_RATE, return_seconds=True
            )


def create_chunks(duration, speech):
    chunks = []
    start = 0.0
    while start < duration:
        hard_end = min(duration, start + MAX_CHUNK)
        if hard_end >= duration:
            chunks.append((start, duration))
            break
        candidates = [
            float(region["end"])
            for region in speech
            if hard_end - SEARCH_BACK <= float(region["end"]) <= hard_end
        ]
        end = max(candidates) if candidates else hard_end
        chunks.append((start, end))
        start = end
    return chunks

# ============================================================
# SRT UTILITIES
# ============================================================
def srt_time(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(path, cues):
    with open(path, "w", encoding="utf-8-sig", newline="\n") as file:
        for index, (start, end, text) in enumerate(cues, 1):
            file.write(
                f"{index}\n"
                f"{srt_time(start)} --> {srt_time(end)}\n"
                f"{text}\n\n"
            )


def prevent_overlaps(cues):
    fixed = []
    for index, (start, end, text) in enumerate(cues):
        if index + 1 < len(cues):
            next_start = cues[index + 1][0]
            end = min(end, next_start - 0.02)
        end = max(start + 0.01, end)
        fixed.append((start, end, text))
    return fixed

# ============================================================
# YOUTUBE HORIZONTAL SUBTITLES
# ============================================================
def make_sentence_srt(words, max_chars=42):
    cues = []
    group = []

    def flush():
        if not group:
            return
        text = " ".join(word["word"] for word in group)
        cues.append((group[0]["start"], group[-1]["end"] + 0.12, text))
        group.clear()

    for index, word in enumerate(words):
        if group:
            candidate = " ".join(item["word"] for item in group) + " " + word["word"]
            duration = word["end"] - group[0]["start"]
            if len(candidate) > max_chars or duration > 6.0:
                flush()
        group.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        punctuation = bool(re.search(r"[.!?।！？]$", word["word"]))
        pause = (
            next_word is not None
            and (next_word["start"] - word["end"]) > 0.7
        )
        if punctuation or pause:
            flush()
    flush()
    return prevent_overlaps(cues)

# ============================================================
# SHORTS / REELS SUBTITLES
# ============================================================
def make_short_srt(words, max_words=3, uppercase=True, strip_punctuation=True):
    cues = []
    group = []

    def flush():
        if not group:
            return
        tokens = []
        for word in group:
            token = word["word"]
            if strip_punctuation:
                token = token.rstrip(",;:.…")
            if token:
                tokens.append(token)
        text = " ".join(tokens)
        if uppercase:
            text = text.upper()
        if text:
            cues.append((group[0]["start"], group[-1]["end"] + 0.08, text))
        group.clear()

    for index, word in enumerate(words):
        if group:
            candidate = " ".join(item["word"] for item in group) + " " + word["word"]
            if len(group) >= max_words or len(candidate) > 24:
                flush()
        group.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        if next_word is None or (next_word["start"] - word["end"]) > 0.4:
            flush()
    flush()
    return prevent_overlaps(cues)

# ============================================================
# WORD-LEVEL SUBTITLES
# ============================================================
def make_word_srt(words, uppercase=True, strip_punctuation=True):
    cues = []
    for word in words:
        text = word["word"]
        if strip_punctuation:
            text = text.rstrip(",;:.…")
        if uppercase:
            text = text.upper()
        if text:
            cues.append((word["start"], word["end"], text))
    return cues

# ============================================================
# TRANSCRIPTION PIPELINE
# ============================================================
def run_pipeline(
    source,
    language_name,
    max_chars,
    max_words,
    uppercase,
    strip_punctuation,
    log,
    progress,
):
    language = LANGUAGES[language_name]
    folder = OUTPUT_DIR / Path(source).stem
    original_file_name = Path(source).stem
    folder.mkdir(parents=True, exist_ok=True)
    wav_path = folder / "audio_16k.wav"

    log(f"Input: {Path(source).name}")
    log(f"Language: {language_name}")
    progress(0.02, desc="Preparing audio")

    if not wav_path.exists() or wav_path.stat().st_mtime < Path(source).stat().st_mtime:
        convert_audio(source, wav_path)
    else:
        log("Using cached audio")

    if STOP_EVENT.is_set():
        return None

    info = read_wav_info(wav_path)
    if info["rate"] != 16000 or info["channels"] != 1 or info["width"] != 2:
        raise RuntimeError("Audio must be 16kHz mono 16-bit WAV")

    duration = info["duration"]
    log(f"Duration: {duration / 60:.2f} minutes")

    progress(0.08, desc="Detecting speech · Silero VAD")
    speech = detect_speech(wav_path)
    if STOP_EVENT.is_set():
        return None

    chunks = create_chunks(duration, speech)
    log(f"Chunking complete: {len(chunks)} chunks")
    progress(0.12, desc=f"Chunking complete · {len(chunks)} chunks")

    is_shruti = language.startswith("shruti:")
    log("Backend: Cactus Shruti" if is_shruti else "Backend: Cactus Whistle")
    progress(0.15, desc="Loading speech model")
    model = load_asr_model(language, log)
    if model is None:
        return None

    words = []
    failed = 0
    raw_responses = []

    with tempfile.TemporaryDirectory() as temp:
        chunk_path = Path(temp) / "chunk.wav"
        for index, (start, end) in enumerate(chunks, 1):
            if STOP_EVENT.is_set():
                log("Stopped")
                return None

            extract_wav_chunk(wav_path, chunk_path, start, end)

            try:
                # Regional Shruti models select their own language.
                # Whistle uses the selected language code.
                kwargs = {"word_timestamps": True}
                if not is_shruti:
                    kwargs["language"] = language
                result = model.transcribe(str(chunk_path), **kwargs)

                raw_responses.append({
                    "chunk_index": index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "model_response": result,
                })

                if not isinstance(result, dict):
                    raise ValueError("Invalid transcription response")

                for item in result.get("words", []):
                    try:
                        text = str(item.get("word", "")).strip()
                        ws = float(item["start"]) + start
                        we = float(item["end"]) + start
                    except (KeyError, ValueError, TypeError):
                        continue
                    if text and we > ws:
                        words.append({"word": text, "start": ws, "end": we})

            except Exception as exc:
                raw_responses.append({
                    "chunk_index": index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "error": str(exc),
                })
                failed += 1
                log(f"Chunk {index} failed: {exc}")

            progress(
                index / len(chunks),
                desc=f"Transcribing {index}/{len(chunks)} · {len(words)} words",
            )

    progress(0.96, desc="Building subtitle files")
    words.sort(key=lambda item: (item["start"], item["end"]))

    if not words:
        raise RuntimeError("No words returned by transcription")

    log(f"Words: {len(words)}")
    log(f"Failed chunks: {failed}")

    sentence_cues = make_sentence_srt(words, int(max_chars))
    short_cues = make_short_srt(words, int(max_words), uppercase, strip_punctuation)
    word_cues = make_word_srt(words, uppercase, strip_punctuation)

    sentence_path = folder / f"sentence_{original_file_name}.srt"
    short_path = folder / f"short_{original_file_name}.srt"
    word_path = folder / f"word_level_{original_file_name}.srt"
    transcript_path = folder / f"transcript_{original_file_name}.txt"

    write_srt(sentence_path, sentence_cues)
    write_srt(short_path, short_cues)
    write_srt(word_path, word_cues)

    transcript = " ".join(word["word"] for word in words)
    transcript_path.write_text(transcript, encoding="utf-8-sig")

    json_path = folder / f"original_model_{original_file_name}.json"
    json_path.write_text(
        json.dumps({
            "source_file": Path(source).name,
            "language": language_name,
            "backend": "Cactus Shruti" if is_shruti else "Cactus Whistle",
            "audio_duration_seconds": duration,
            "chunk_responses": raw_responses,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    zip_path = folder / f"subtitles_{original_file_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in (sentence_path, short_path, word_path, transcript_path, json_path):
            archive.write(item, arcname=item.name)

    progress(1.0, desc="Export completed")
    log("All subtitle files saved")
    log(f"Output: {folder}")

    return {
        "sentence": sentence_path,
        "short": short_path,
        "word": word_path,
        "transcript": transcript_path,
        "json": json_path,
        "zip": zip_path,
    }

# ============================================================
# ORIGINAL BEAUTIFUL CSS + STABILITY FIXES
# ============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root, .dark, .gradio-container {
    --body-background-fill: transparent;
    --background-fill-primary: #101015;
    --background-fill-secondary: #101015;
    --block-background-fill: transparent;
    --block-border-color: rgba(255,255,255,.08);
    --border-color-primary: rgba(255,255,255,.08);
    --input-background-fill: #0a0a0e;
    --input-border-color: rgba(255,255,255,.10);
    --body-text-color: #ececf1;
    --body-text-color-subdued: #8b8b9a;
    --block-label-text-color: #9a9aab;
    --block-title-text-color: #c9c9d6;
    --color-accent: #8b7cf6;
}

/* Always reserve space for the scrollbar so the layout never shifts sideways */
html {
    overflow-y: scroll !important;
    scrollbar-gutter: stable;
}

html, body {
    margin: 0 !important;
    width: 100% !important;
    overflow-x: hidden !important;
}

body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background:
        radial-gradient(ellipse 70% 45% at 50% -8%, rgba(124,92,255,.28), transparent 70%),
        radial-gradient(ellipse 40% 30% at 90% 10%, rgba(56,189,248,.10), transparent 70%),
        #09090c !important;
    background-attachment: fixed !important;
}

.gradio-container,
.gradio-container > .main,
.gradio-container .main {
    width: 100% !important;
    max-width: none !important;
}

.gradio-container {
    margin: 0 !important;
    padding-left: clamp(10px,1.5vw,24px) !important;
    padding-right: clamp(10px,1.5vw,24px) !important;
    box-sizing: border-box !important;
}

footer {
    display: none !important;
}

/* HERO */
#hero {
    text-align: center;
    padding: 10px 10px 8px;
}

#hero .title {
    margin: 0;
    font-size: clamp(30px,5.5vw,54px);
    font-weight: 800;
    letter-spacing: -.5px;
    background: linear-gradient(180deg, #ffffff 25%, #a99bff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

#hero .tagline {
    margin: 6px 0 12px;
    color: #8b8b9a;
    font-size: 15px;
}

#hero .links {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
}

#hero .links a {
    display: inline-flex;
    align-items: center;
    padding: 7px 16px;
    border-radius: 999px;
    color: #dcdcea !important;
    text-decoration: none !important;
    font-size: 13.5px;
    font-weight: 500;
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.12);
    transition: background .2s ease, border-color .2s ease;
}

#hero .links a:hover {
    border-color: #8b7cf6;
    background: rgba(139,124,246,.12);
}

/* SUPPORTED LANGUAGES */
#langs {
    max-width: 1100px;
    margin: 0 auto 14px !important;
}

.lang-box {
    text-align: center;
    padding: 4px 10px;
}

.lang-title {
    color: #c9c9d6;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.lang-group {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
}

.lang-group b {
    color: #b9aeff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-right: 6px;
}

.lang-group span {
    padding: 3px 11px;
    border-radius: 999px;
    font-size: 12px;
    color: #dcdcea;
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.10);
}

/* GLASS CARDS */
.card {
    background: rgba(255,255,255,.025) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,.35);
}

.card h3 {
    color: #c9c9d6 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.block {
    border-radius: 12px !important;
}

ul.options {
    z-index: 9999 !important;
}

/* INPUTS */
textarea, input[type=text] {
    color: #ececf1 !important;
}

textarea::placeholder,
input::placeholder {
    color: #5d5d6c !important;
}

textarea:focus,
input:focus {
    border-color: #8b7cf6 !important;
    box-shadow: 0 0 0 3px rgba(139,124,246,.18) !important;
}

/* FIXED HEIGHT: NO PAGE JUMPING */
#log textarea {
    height: 170px !important;
    min-height: 170px !important;
    max-height: 170px !important;
    overflow-y: auto !important;
    resize: none !important;
    color: #aab4c4 !important;
}

#log textarea,
.srt textarea {
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace !important;
    font-size: 12.5px !important;
    background: #0a0a0e !important;
}

/* Fixed-height wrapper so the preview box never fights lines=12 */
.srt {
    height: 380px !important;
    min-height: 380px !important;
}

.srt textarea {
    color: #d4d4e0 !important;
    height: 330px !important;
    min-height: 330px !important;
    max-height: 330px !important;
    overflow-y: auto !important;
    resize: none !important;
}

/* Every tab panel has the same height, so switching tabs never resizes the card */
.tabitem {
    min-height: 520px !important;
}

/* Same size for empty and filled file boxes */
.card [data-testid="file"] {
    min-height: 90px;
}

/* BUTTONS */
#go {
    background: linear-gradient(135deg, #8b7cf6, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    transition: box-shadow .2s ease, filter .2s ease;
    box-shadow: 0 6px 24px rgba(99,102,241,.40);
}

#go:hover {
    box-shadow: 0 10px 30px rgba(99,102,241,.55);
    filter: brightness(1.08);
}

#stop {
    background: rgba(255,255,255,.04) !important;
    color: #e4e4ee !important;
    border: 1px solid rgba(255,255,255,.14) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
}

#stop:hover {
    border-color: #f87171 !important;
    color: #fca5a5 !important;
    background: rgba(248,113,113,.08) !important;
}

/* TABS */
button[role=tab] {
    color: #8b8b9a !important;
    font-weight: 500;
}

button[role=tab][aria-selected=true] {
    color: #fff !important;
    border-color: #8b7cf6 !important;
}

/* LAYOUT STABILITY */
#main-row,
#left-panel,
#right-panel {
    min-width: 0 !important;
    animation: none !important;
    transform: none !important;
}

/* Progress animation is INSIDE the Status component. */
.live-status {
    padding: 10px 12px;
    background: #0a0a0e;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 10px;
    min-height: 75px;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.live-status-text {
    color: #b9aeff;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 9px;
}
.live-status-track {
    height: 7px;
    background: #292536;
    border-radius: 99px;
    overflow: hidden;
}
.live-status-fill {
    height: 100%;
    background: linear-gradient(90deg,#6366f1,#a99bff,#38bdf8);
    border-radius: inherit;
    transition: width .2s ease;
}
.live-status-fill.is-active {
    background-size: 200% 100%;
    animation: cactus-status-shimmer 1.3s linear infinite;
}
.live-status-percent {
    text-align: right;
    color: #9a9aab;
    font-size: 11px;
    margin-top: 5px;
}
@keyframes cactus-status-shimmer {
    0% { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}

@media (max-width:800px) {
    .gradio-container {
        padding: 12px !important;
    }
    #hero .title {
        font-size: 30px;
    }
}
"""

# ============================================================
# LIVE STATUS
# ============================================================
from html import escape as escape_html


def status_html(message, percent=0, active=False):
    """Render stage, chunk count and animated progress inside Status."""
    message = escape_html(str(message))
    percent = max(0, min(100, float(percent)))
    active_class = " is-active" if active else ""
    return (
        '<div class="live-status">'
        f'<div class="live-status-text">{message}</div>'
        '<div class="live-status-track">'
        f'<div class="live-status-fill{active_class}" '
        f'style="width:{percent:.1f}%"></div>'
        '</div>'
        f'<div class="live-status-percent">{percent:.0f}%</div>'
        '</div>'
    )


def languages_html():
    """Supported-language chips, built from the same dictionaries as the dropdown."""
    western = ["English"] + sorted(n for n in WHISTLE_LANGUAGES if n != "English")
    indian = sorted(SHRUTI_LANGUAGES)

    def chips(items):
        return "".join(f"<span>{escape_html(n)}</span>" for n in items)

    return (
        '<div class="lang-box">'
        '<div class="lang-title">Supported Languages</div>'
        f'<div class="lang-group"><b>Whistle</b>{chips(western)}</div>'
        f'<div class="lang-group"><b>Shruti</b>{chips(indian)}</div>'
        '</div>'
    )

# ============================================================
# GRADIO HANDLER
# ============================================================
def process(
    uploaded_file,
    local_path,
    language_name,
    max_chars,
    max_words,
    uppercase,
    strip_punctuation,
):
    """Stream live status from a worker while preserving all output files."""
    if not JOB_LOCK.acquire(blocking=False):
        yield (status_html("Another transcription is running"), "", *([None] * 6), *([''] * 5))
        return

    STOP_EVENT.clear()
    events = queue.Queue()
    logs = []

    def log(message):
        print(message, flush=True)
        events.put(("log", str(message)))

    def update_progress(value, desc=""):
        events.put(("progress", float(value), str(desc)))

    def output(status, files=None, previews=None, percent=0, active=False):
        files = files or [None] * 6
        previews = previews or [""] * 5
        return (
            status_html(status, percent, active),
            "\n".join(logs[-100:]),
            *files,
            *previews,
        )

    try:
        if IS_LOCAL and local_path and local_path.strip():
            source = Path(local_path.strip().strip('"').strip("'")).expanduser().resolve()
        elif uploaded_file:
            source = Path(uploaded_file).resolve()
        else:
            yield output("Select a file first")
            return

        if not source.is_file():
            yield output(f"File not found: {source}")
            return

        result_holder = {}

        def worker():
            try:
                result_holder["result"] = run_pipeline(
                    source,
                    language_name,
                    max_chars,
                    max_words,
                    uppercase,
                    strip_punctuation,
                    log,
                    update_progress,
                )
            except Exception as exc:
                traceback.print_exc()
                result_holder["error"] = str(exc)
            finally:
                events.put(("finished",))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        message = "Preparing audio"
        percent = 0.0
        yield output(message, percent=percent, active=True)

        while True:
            try:
                event = events.get(timeout=0.2)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

            if event[0] == "log":
                logs.append(event[1])
                if len(logs) > 100:
                    del logs[:-100]
            elif event[0] == "progress":
                percent = event[1] * 100
                message = event[2] or message
            elif event[0] == "finished":
                break

            yield output(message, percent=percent, active=True)

        thread.join()
        while not events.empty():
            event = events.get_nowait()
            if event[0] == "log":
                logs.append(event[1])

        if "error" in result_holder:
            yield output(f"Error: {result_holder['error']}", percent=percent)
            return

        result = result_holder.get("result")
        if result is None:
            yield output("Stopped", percent=percent)
            return

        def preview(path):
            return path.read_text(encoding="utf-8-sig")[:20000]

        file_keys = ("word", "sentence", "short", "transcript", "json", "zip")
        preview_keys = ("word", "sentence", "short", "transcript", "json")
        yield output(
            "Completed successfully ✔",
            files=[str(result[key]) for key in file_keys],
            previews=[preview(result[key]) for key in preview_keys],
            percent=100,
        )
    except Exception as exc:
        traceback.print_exc()
        yield output(f"Error: {exc}")
    finally:
        JOB_LOCK.release()


def request_stop():
    STOP_EVENT.set()
    return status_html("Stopping after current chunk...", active=True)

# ============================================================
# TAB / FILE LAYOUT STABILITY
# ============================================================
# Render all tab contents at startup (needs a recent Gradio); ignored on older versions.
TAB_KWARGS = (
    {"render_children": True}
    if "render_children" in inspect.signature(gr.Tab.__init__).parameters
    else {}
)

# Fixed height for file boxes so empty and filled states are the same size.
FILE_HEIGHT = 120

# ============================================================
# ORIGINAL-STYLE FULL-SCREEN WEBUI
# ============================================================
def build_ui():
    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="zinc",
    )

    block_kwargs = {"title": "Cactus-Compute Subtitle"}
    launch_kwargs = {}

    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        block_kwargs.update(css=CSS, theme=theme)
    else:
        launch_kwargs.update(css=CSS, theme=theme)

    with gr.Blocks(**block_kwargs) as demo:
        gr.HTML("""
        <div id="hero">
            <h1 class="title">CACTUS-COMPUTE SUBTITLE</h1>
            <p class="tagline">
                Cactus-Compute Whistle + Shruti |
                29 Languages |
                Sentence SRT · Short SRT · Word Level SRT
            </p>
            <div class="links">
                <a href="https://github.com/NeuralFalconYT/cactus-compute-subtitle" target="_blank">GitHub</a>
                <a href="https://huggingface.co/Cactus-Compute/whistle" target="_blank">Whistle Model</a>
                <a href="https://huggingface.co/udgamlabs" target="_blank">Shruti Models</a>
            </div>
        </div>
        """)

        gr.HTML(languages_html(), elem_id="langs")

        with gr.Row(elem_id="main-row", equal_height=False):

            # LEFT INPUT CARD
            with gr.Column(
                scale=1,
                min_width=340,
                elem_id="left-panel",
                elem_classes=["card"],
            ):
                gr.Markdown("### Input")

                local_path = gr.Textbox(
                    label="Paste file path (local only)",
                    placeholder=r"C:\videos\video.mp4",
                    lines=1,
                    visible=IS_LOCAL,
                )

                uploaded_file = gr.File(
                    label="Drag & drop audio / video",
                    type="filepath",
                )

                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Language",
                    filterable=True,
                )

                with gr.Accordion("Subtitle Options", open=False):
                    max_chars = gr.Slider(
                        minimum=25, maximum=65, value=42, step=1,
                        label="Sentence SRT max characters",
                    )
                    max_words = gr.Slider(
                        minimum=1, maximum=3, value=3, step=1,
                        label="Short SRT max words per cue",
                    )
                    uppercase = gr.Checkbox(
                        value=True,
                        label="UPPERCASE Short + Word Level",
                    )
                    strip_punctuation = gr.Checkbox(
                        value=True,
                        label="Remove trailing punctuation",
                    )

                generate = gr.Button(
                    "Generate Subtitles",
                    variant="primary",
                    elem_id="go",
                )
                stop_button = gr.Button("Stop", elem_id="stop")

            # RIGHT OUTPUT CARD
            with gr.Column(
                scale=2,
                min_width=340,
                elem_id="right-panel",
                elem_classes=["card"],
            ):
                gr.Markdown("### Output")
                gr.Markdown("**Status**")
                status = gr.HTML(
                    value=status_html("Ready"),
                    elem_id="status",
                )

                logs = gr.Textbox(
                    label="Summary",
                    lines=6,
                    max_lines=6,
                    interactive=False,
                    autoscroll=True,
                    elem_id="log",
                )

                with gr.Tabs():
                    with gr.Tab("1 · Word Level SRT", **TAB_KWARGS):
                        word_file = gr.File(label="Word Level SRT", height=FILE_HEIGHT)
                        word_preview = gr.Textbox(
                            label="Preview", lines=12, max_lines=12,
                            interactive=False, autoscroll=False,
                            elem_classes=["srt"],
                        )

                    with gr.Tab("2 · Sentence SRT", **TAB_KWARGS):
                        sentence_file = gr.File(label="Sentence SRT", height=FILE_HEIGHT)
                        sentence_preview = gr.Textbox(
                            label="Preview", lines=12, max_lines=12,
                            interactive=False, autoscroll=False,
                            elem_classes=["srt"],
                        )

                    with gr.Tab("3 · Short SRT", **TAB_KWARGS):
                        short_file = gr.File(label="Short SRT", height=FILE_HEIGHT)
                        short_preview = gr.Textbox(
                            label="Preview", lines=12, max_lines=12,
                            interactive=False, autoscroll=False,
                            elem_classes=["srt"],
                        )

                    with gr.Tab("4 · Transcript TXT", **TAB_KWARGS):
                        transcript_file = gr.File(label="Transcript", height=FILE_HEIGHT)
                        transcript_preview = gr.Textbox(
                            label="Transcript Preview", lines=12, max_lines=12,
                            interactive=False, autoscroll=False,
                            show_copy_button=True,
                            elem_classes=["srt"],
                        )

                    with gr.Tab("5 · JSON LOG", **TAB_KWARGS):
                        json_file = gr.File(label="Original Model Response JSON", height=FILE_HEIGHT)
                        json_preview = gr.Textbox(
                            label="Raw JSON Preview", lines=12, max_lines=12,
                            interactive=False, autoscroll=False,
                            elem_classes=["srt"],
                        )

                    with gr.Tab("6 · Export ZIP", **TAB_KWARGS):
                        zip_file = gr.File(label="Download All Outputs (.zip)", height=FILE_HEIGHT)
                        gr.Markdown("Includes 3 SRT files, transcript TXT and original model JSON.")

        generate.click(
            fn=process,
            inputs=[
                uploaded_file,
                local_path,
                language,
                max_chars,
                max_words,
                uppercase,
                strip_punctuation,
            ],
            outputs=[
                status,
                logs,
                word_file,
                sentence_file,
                short_file,
                transcript_file,
                json_file,
                zip_file,
                word_preview,
                sentence_preview,
                short_preview,
                transcript_preview,
                json_preview,
            ],
            concurrency_limit=1,
        )

        stop_button.click(
            fn=request_stop,
            inputs=[],
            outputs=[status],
            queue=False,
        )

    return demo, launch_kwargs

# ============================================================
# START
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("CACTUS-COMPUTE SUBTITLE")
    print(f"Languages: {len(LANGUAGES)}")
    print("Dropdown:", ", ".join(LANGUAGES))
    print(f"Model directory: {MODEL_DIR}")
    print("=" * 55)

    if not shutil.which("ffmpeg"):
        print("WARNING: ffmpeg not found in PATH")

    demo, launch_kwargs = build_ui()
    demo.queue().launch(
        inbrowser=IS_LOCAL,
        show_error=True,
        **launch_kwargs,
    )
