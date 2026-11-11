
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
from silero_vad import load_silero_vad, get_speech_timestamps
# ============================================================
# PATHS AND SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "needle_output"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ON_SPACES = bool(os.environ.get("SPACE_ID"))
IS_LOCAL = not ON_SPACES
SAMPLE_RATE = 16000
MAX_CHUNK = 29.0
SEARCH_BACK = 5.0
STOP_EVENT = threading.Event()
JOB_LOCK = threading.Lock()
# ============================================================
# LANGUAGES - 29 TOTAL
# ============================================================
NEEDLE_LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Dutch": "nl",
    "Italian": "it",
    "Polish": "pl",
}
SHRUTI_LANGUAGES = [
    "Hindi",
    "Marathi",
    "Tamil",
    "Bengali",
    "Telugu",
    "Gujarati",
    "Odia",
    "Kannada",
    "Urdu",
    "Malayalam",
    "Assamese",
    "Sanskrit",
    "Punjabi",
    "Bodo",
    "Dogri",
    "Kashmiri",
    "Konkani",
    "Maithili",
    "Manipuri",
    "Nepali",
    "Santali",
    "Sindhi",
]
LANGUAGES = dict(NEEDLE_LANGUAGES)
for name in SHRUTI_LANGUAGES:
    LANGUAGES[name] = "shruti:" + name.lower()
PRIORITY = ["English", "Hindi", "Bengali"]
LANGUAGES = {
    name: LANGUAGES[name]
    for name in PRIORITY + sorted(
        name for name in LANGUAGES
        if name not in PRIORITY
    )
}
# ============================================================
# MODEL DOWNLOAD AND CACHE
# ============================================================
def load_shruti_model(language_code, log):
    from huggingface_hub import hf_hub_download
    from cactus import Whistle
    name = language_code.split(":", 1)[1]
    if name == "hindi":
        repo_id = "udgamlabs/Shruti"
        filename = "shruti.cact"
    else:
        repo_id = f"udgamlabs/shruti-{name}"
        filename = f"shruti-{name}.cact"
    folder = MODEL_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    log(f"Model: {name}")
    log(f"Cache: {folder}")
    for item in (filename, "config.json"):
        if STOP_EVENT.is_set():
            return None
        destination = folder / item
        if (
            destination.is_file()
            and destination.stat().st_size > 0
        ):
            log(f"Cached: {item}")
            continue
        log(f"Downloading {item}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=item,
            local_dir=str(folder),
        )
        if (
            not destination.is_file()
            or destination.stat().st_size == 0
        ):
            raise RuntimeError(
                f"Missing model file: {destination}"
            )
        log(f"Downloaded: {item}")
    if STOP_EVENT.is_set():
        return None
    model_path = folder / filename
    log("Loading Cactus Whistle...")
    return Whistle(str(model_path))
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
    subprocess.run(
        command,
        check=True,
    )
def read_wav_info(path):
    with wave.open(str(path), "rb") as wf:
        return {
            "rate": wf.getframerate(),
            "channels": wf.getnchannels(),
            "width": wf.getsampwidth(),
            "frames": wf.getnframes(),
            "duration": (
                wf.getnframes() / wf.getframerate()
            ),
        }
def extract_wav_chunk(
    source,
    destination,
    start,
    end,
):
    with wave.open(str(source), "rb") as wf:
        rate = wf.getframerate()
        first = max(0, int(start * rate))
        last = min(
            wf.getnframes(),
            int(end * rate),
        )
        wf.setpos(first)
        audio = wf.readframes(
            max(0, last - first)
        )
        with wave.open(str(destination), "wb") as out:
            out.setnchannels(wf.getnchannels())
            out.setsampwidth(wf.getsampwidth())
            out.setframerate(rate)
            out.writeframes(audio)
# ============================================================
# SILERO VAD
# ============================================================
def detect_speech(wav_path):
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(
        raw,
        dtype=np.int16,
    ).astype(np.float32)
    audio /= 32768.0
    model = load_silero_vad()
    speech = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
    )
    return speech
def create_chunks(duration, speech):
    chunks = []
    start = 0.0
    while start < duration:
        hard_end = min(
            duration,
            start + MAX_CHUNK,
        )
        if hard_end >= duration:
            chunks.append((start, duration))
            break
        candidates = [
            float(region["end"])
            for region in speech
            if (
                hard_end - SEARCH_BACK
                <= float(region["end"])
                <= hard_end
            )
        ]
        end = (
            max(candidates)
            if candidates
            else hard_end
        )
        chunks.append((start, end))
        start = end
    return chunks
# ============================================================
# SRT UTILITIES
# ============================================================
def srt_time(seconds):
    milliseconds = max(
        0,
        round(seconds * 1000),
    )
    hours, milliseconds = divmod(
        milliseconds,
        3600000,
    )
    minutes, milliseconds = divmod(
        milliseconds,
        60000,
    )
    seconds, milliseconds = divmod(
        milliseconds,
        1000,
    )
    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d},"
        f"{milliseconds:03d}"
    )
def write_srt(path, cues):
    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="\n",
    ) as file:
        for index, (start, end, text) in enumerate(
            cues,
            1,
        ):
            file.write(
                f"{index}\n"
                f"{srt_time(start)} --> "
                f"{srt_time(end)}\n"
                f"{text}\n\n"
            )
def prevent_overlaps(cues):
    fixed = []
    for index, (start, end, text) in enumerate(cues):
        if index + 1 < len(cues):
            next_start = cues[index + 1][0]
            end = min(
                end,
                next_start - 0.02,
            )
        end = max(
            start + 0.01,
            end,
        )
        fixed.append((start, end, text))
    return fixed
# ============================================================
# YOUTUBE HORIZONTAL SUBTITLES
# ============================================================
def make_youtube_srt(words, max_chars=42):
    cues = []
    group = []
    def flush():
        if not group:
            return
        text = " ".join(
            word["word"]
            for word in group
        )
        cues.append((
            group[0]["start"],
            group[-1]["end"] + 0.12,
            text,
        ))
        group.clear()
    for index, word in enumerate(words):
        if group:
            candidate = (
                " ".join(
                    item["word"]
                    for item in group
                )
                + " "
                + word["word"]
            )
            duration = (
                word["end"]
                - group[0]["start"]
            )
            if (
                len(candidate) > max_chars
                or duration > 6.0
            ):
                flush()
        group.append(word)
        next_word = (
            words[index + 1]
            if index + 1 < len(words)
            else None
        )
        punctuation = bool(
            re.search(
                r"[.!?।！？]$",
                word["word"],
            )
        )
        pause = (
            next_word is not None
            and (
                next_word["start"]
                - word["end"]
            ) > 0.7
        )
        if punctuation or pause:
            flush()
    flush()
    return prevent_overlaps(cues)
# ============================================================
# SHORTS / REELS SUBTITLES
# ============================================================
def make_shorts_srt(
    words,
    max_words=3,
    uppercase=True,
    strip_punctuation=True,
):
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
            cues.append((
                group[0]["start"],
                group[-1]["end"] + 0.08,
                text,
            ))
        group.clear()
    for index, word in enumerate(words):
        if group:
            candidate = (
                " ".join(
                    item["word"]
                    for item in group
                )
                + " "
                + word["word"]
            )
            if (
                len(group) >= max_words
                or len(candidate) > 24
            ):
                flush()
        group.append(word)
        next_word = (
            words[index + 1]
            if index + 1 < len(words)
            else None
        )
        if (
            next_word is None
            or (
                next_word["start"]
                - word["end"]
            ) > 0.4
        ):
            flush()
    flush()
    return prevent_overlaps(cues)
# ============================================================
# WORD-LEVEL SUBTITLES
# ============================================================
def make_word_srt(
    words,
    uppercase=True,
    strip_punctuation=True,
):
    cues = []
    for word in words:
        text = word["word"]
        if strip_punctuation:
            text = text.rstrip(",;:.…")
        if uppercase:
            text = text.upper()
        if text:
            cues.append((
                word["start"],
                word["end"],
                text,
            ))
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
    folder = (
        OUTPUT_DIR
        / Path(source).stem
    )
    original_file_name = Path(source).stem
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )
    wav_path = folder / "audio_16k.wav"
    log(f"Input: {Path(source).name}")
    log(f"Language: {language_name}")
    progress(0, desc="Preparing audio")
    if (
        not wav_path.exists()
        or wav_path.stat().st_mtime
        < Path(source).stat().st_mtime
    ):
        convert_audio(source, wav_path)
    else:
        log("Using cached audio")
    if STOP_EVENT.is_set():
        return None
    info = read_wav_info(wav_path)
    if (
        info["rate"] != 16000
        or info["channels"] != 1
        or info["width"] != 2
    ):
        raise RuntimeError(
            "Audio must be 16kHz mono 16-bit WAV"
        )
    duration = info["duration"]
    log(
        f"Duration: {duration / 60:.2f} minutes"
    )
    progress(0, desc="Silero VAD")
    speech = detect_speech(wav_path)
    if STOP_EVENT.is_set():
        return None
    chunks = create_chunks(
        duration,
        speech,
    )
    log(f"Chunks: {len(chunks)}")
    is_shruti = language.startswith("shruti:")
    if is_shruti:
        log("Backend: Cactus Shruti")
        model = load_shruti_model(
            language,
            log,
        )
        if model is None:
            return None
    else:
        log("Backend: Original Needle")
        import needle
    words = []
    failed = 0
    raw_responses = []
    with tempfile.TemporaryDirectory() as temp:
        chunk_path = Path(temp) / "chunk.wav"
        for index, (start, end) in enumerate(
            chunks,
            1,
        ):
            if STOP_EVENT.is_set():
                log("Stopped")
                return None
            extract_wav_chunk(
                wav_path,
                chunk_path,
                start,
                end,
            )
            try:
                if is_shruti:
                    # Shruti model selects its own language.
                    result = model.transcribe(
                        str(chunk_path),
                        word_timestamps=True,
                    )
                else:
                    # Original Needle call.
                    result = needle.transcribe(
                        str(chunk_path),
                        language=language,
                        word_timestamps=True,
                    )
                raw_responses.append({
                    "chunk_index": index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "model_response": result,
                })
                if not isinstance(result, dict):
                    raise ValueError(
                        "Invalid transcription response"
                    )
                for item in result.get("words", []):
                    try:
                        text = str(
                            item.get("word", "")
                        ).strip()
                        ws = (
                            float(item["start"])
                            + start
                        )
                        we = (
                            float(item["end"])
                            + start
                        )
                    except (
                        KeyError,
                        ValueError,
                        TypeError,
                    ):
                        continue
                    if text and we > ws:
                        words.append({
                            "word": text,
                            "start": ws,
                            "end": we,
                        })
            except Exception as exc:
                raw_responses.append({
                    "chunk_index": index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "error": str(exc),
                })
                failed += 1
                log(
                    f"Chunk {index} failed: {exc}"
                )
            progress(
                index / len(chunks),
                desc=(
                    f"Transcribing {index}/{len(chunks)}"
                    f" · {len(words)} words"
                ),
            )
    words.sort(
        key=lambda item: (
            item["start"],
            item["end"],
        )
    )
    if not words:
        raise RuntimeError(
            "No words returned by transcription"
        )
    log(f"Words: {len(words)}")
    log(f"Failed chunks: {failed}")
    youtube_cues = make_youtube_srt(
        words,
        int(max_chars),
    )
    shorts_cues = make_shorts_srt(
        words,
        int(max_words),
        uppercase,
        strip_punctuation,
    )
    word_cues = make_word_srt(
        words,
        uppercase,
        strip_punctuation,
    )
    youtube_path = folder / f"youtube_{original_file_name}.srt"
    shorts_path = folder / f"shorts_{original_file_name}.srt"
    word_path = folder / f"word_level_{original_file_name}.srt"
    transcript_path = folder / f"transcript_{original_file_name}.txt"
    write_srt(youtube_path, youtube_cues)
    write_srt(shorts_path, shorts_cues)
    write_srt(word_path, word_cues)
    transcript = " ".join(
        word["word"]
        for word in words
    )
    transcript_path.write_text(
        transcript,
        encoding="utf-8-sig",
    )
    json_path = folder / f"original_model_{original_file_name}.json"
    json_path.write_text(
        json.dumps({
            "source_file": Path(source).name,
            "language": language_name,
            "backend": "Cactus Shruti" if is_shruti else "Needle",
            "audio_duration_seconds": duration,
            "chunk_responses": raw_responses,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    zip_path = folder / f"subtitles_{original_file_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in (youtube_path, shorts_path, word_path, transcript_path, json_path):
            archive.write(item, arcname=item.name)
    log("All subtitle files saved")
    log(f"Output: {folder}")
    return {
        "youtube": youtube_path,
        "shorts": shorts_path,
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
html, body {
    margin: 0 !important;
    width: 100% !important;
    overflow-x: hidden !important;
}
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background:
        radial-gradient(
            ellipse 70% 45% at 50% -8%,
            rgba(124,92,255,.28),
            transparent 70%
        ),
        radial-gradient(
            ellipse 40% 30% at 90% 10%,
            rgba(56,189,248,.10),
            transparent 70%
        ),
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
    padding: 38px 10px 26px;
}
#hero .title {
    margin: 0;
    font-size: clamp(30px,5.5vw,54px);
    font-weight: 800;
    letter-spacing: -.5px;
    background: linear-gradient(
        180deg,
        #ffffff 25%,
        #a99bff 100%
    );
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}
#hero .tagline {
    margin: 10px 0 20px;
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
#status textarea {
    height: 55px !important;
    min-height: 55px !important;
    max-height: 55px !important;
    overflow-y: auto !important;
    resize: none !important;
    color: #b9aeff !important;
}
#log textarea {
    height: 170px !important;
    min-height: 170px !important;
    max-height: 170px !important;
    overflow-y: auto !important;
    resize: none !important;
    color: #aab4c4 !important;
}
#status textarea,
#log textarea,
.srt textarea {
    font-family:
        'JetBrains Mono',
        'Consolas',
        'Courier New',
        monospace !important;
    font-size: 12.5px !important;
    background: #0a0a0e !important;
}
.srt textarea {
    color: #d4d4e0 !important;
    height: 330px !important;
    min-height: 330px !important;
    max-height: 330px !important;
    overflow-y: auto !important;
    resize: none !important;
}
/* BUTTONS */
#go {
    background: linear-gradient(
        135deg,
        #8b7cf6,
        #6366f1
    ) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    transition:
        box-shadow .2s ease,
        filter .2s ease;
    box-shadow:
        0 6px 24px rgba(99,102,241,.40);
}
#go:hover {
    box-shadow:
        0 10px 30px rgba(99,102,241,.55);
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
    progress=gr.Progress(),
):
    if not JOB_LOCK.acquire(blocking=False):
        yield (
            "Another transcription is running",
            "",
            None, None, None, None, None, None,
            "", "", "", "", "",
        )
        return
    STOP_EVENT.clear()
    logs = []
    def log(message):
        print(message, flush=True)
        logs.append(message)
        if len(logs) > 100:
            del logs[:-100]
    def output(
        status,
        youtube=None,
        shorts=None,
        word=None,
        transcript=None,
        original_json=None,
        export_zip=None,
        yt_text="",
        shorts_text="",
        word_text="",
        transcript_text="",
        json_text="",
    ):
        return (
            status,
            "\n".join(logs),
            youtube,
            shorts,
            word,
            transcript,
            original_json,
            export_zip,
            yt_text,
            shorts_text,
            word_text,
            transcript_text,
            json_text,
        )
    try:
        if IS_LOCAL and local_path and local_path.strip():
            source = Path(
                local_path.strip().strip('"').strip("'")
            ).expanduser().resolve()
        elif uploaded_file:
            source = Path(uploaded_file).resolve()
        else:
            yield output("Select a file first")
            return
        if not source.is_file():
            yield output(f"File not found: {source}")
            return
        yield output("Starting transcription...")
        result = run_pipeline(
            source,
            language_name,
            max_chars,
            max_words,
            uppercase,
            strip_punctuation,
            log,
            progress,
        )
        if result is None:
            yield output("Stopped")
            return
        def preview(path):
            return path.read_text(
                encoding="utf-8-sig"
            )[:20000]
        yield output(
            "Completed successfully ✔",
            str(result["youtube"]),
            str(result["shorts"]),
            str(result["word"]),
            str(result["transcript"]),
            str(result["json"]),
            str(result["zip"]),
            preview(result["youtube"]),
            preview(result["shorts"]),
            preview(result["word"]),
            preview(result["transcript"]),
            preview(result["json"]),
        )
    except Exception as exc:
        traceback.print_exc()
        log(f"ERROR: {exc}")
        yield output(f"Error: {exc}")
    finally:
        JOB_LOCK.release()
def request_stop():
    STOP_EVENT.set()
    return "■ Stopping after current step..."
# ============================================================
# ORIGINAL-STYLE FULL-SCREEN WEBUI
# ============================================================
def build_ui():
    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="zinc",
    )
    block_kwargs = {
        "title": "Needle Subtitle",
    }
    launch_kwargs = {}
    if "css" in inspect.signature(
        gr.Blocks.__init__
    ).parameters:
        block_kwargs.update(
            css=CSS,
            theme=theme,
        )
    else:
        launch_kwargs.update(
            css=CSS,
            theme=theme,
        )
    with gr.Blocks(**block_kwargs) as demo:
        gr.HTML("""
        <div id="hero">
            <h1 class="title">NEEDLE SUBTITLE</h1>
            <p class="tagline">
                Needle + Cactus Shruti |
                29 Languages |
                YouTube, Shorts & Word-Level SRT
            </p>
            <div class="links">
                <a href="https://github.com/NeuralFalconYT/needle-subtitle"
                   target="_blank">
                    GitHub
                </a>
                <a href="https://huggingface.co/Cactus-Compute/needle3"
                                   target="_blank">
                                    Needle3 Models
                                </a>
                <a href="https://huggingface.co/udgamlabs"
                   target="_blank">
                    Shruti Models
                </a>
            </div>
        </div>
        """)
        with gr.Row(
            elem_id="main-row",
            equal_height=False,
        ):
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
                    placeholder=r"G:\videos**\a**udio.mp4",
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
                with gr.Accordion(
                    "Subtitle Options",
                    open=False,
                ):
                    max_chars = gr.Slider(
                        minimum=25,
                        maximum=65,
                        value=42,
                        step=1,
                        label="YouTube max characters",
                    )
                    max_words = gr.Slider(
                        minimum=1,
                        maximum=3,
                        value=3,
                        step=1,
                        label="Shorts max words per cue",
                    )
                    uppercase = gr.Checkbox(
                        value=True,
                        label="UPPERCASE Shorts + Word Level",
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
                stop_button = gr.Button(
                    "Stop",
                    elem_id="stop",
                )
            # RIGHT OUTPUT CARD
            with gr.Column(
                scale=2,
                min_width=340,
                elem_id="right-panel",
                elem_classes=["card"],
            ):
                gr.Markdown("### Output")
                status = gr.Textbox(
                    label="Status",
                    lines=1,
                    interactive=False,
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
                    with gr.Tab("1 · YouTube / Horizontal"):
                        youtube_file = gr.File(
                            label="Sentence-level SRT",
                        )
                        youtube_preview = gr.Textbox(
                            label="Preview",
                            lines=12,
                            max_lines=12,
                            interactive=False,
                            autoscroll=False,
                            elem_classes=["srt"],
                        )
                    with gr.Tab("2 · Shorts / Reels"):
                        shorts_file = gr.File(
                            label="Vertical SRT",
                        )
                        shorts_preview = gr.Textbox(
                            label="Preview",
                            lines=12,
                            max_lines=12,
                            interactive=False,
                            autoscroll=False,
                            elem_classes=["srt"],
                        )
                    with gr.Tab("3 · Word Level"):
                        word_file = gr.File(
                            label="Word-level SRT",
                        )
                        word_preview = gr.Textbox(
                            label="Preview",
                            lines=12,
                            max_lines=12,
                            interactive=False,
                            autoscroll=False,
                            elem_classes=["srt"],
                        )
                    with gr.Tab("4 · Transcript TXT"):
                        transcript_file = gr.File(
                            label="Transcript",
                        )
                        transcript_preview = gr.Textbox(
                            label="Transcript Preview",
                            lines=12,
                            max_lines=12,
                            interactive=False,
                            autoscroll=False,
                            show_copy_button=True,
                            elem_classes=["srt"],
                        )
                    with gr.Tab("5 · JSON Log"):
                        json_file = gr.File(label="Original Model Response JSON")
                        json_preview = gr.Textbox(
                            label="Raw JSON Preview", lines=12, max_lines=12,
                            interactive=False, autoscroll=False,
                            elem_classes=["srt"],
                        )
                    with gr.Tab("6 · Export ZIP"):
                        zip_file = gr.File(label="Download All Outputs (.zip)")
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
                youtube_file,
                shorts_file,
                word_file,
                transcript_file,
                json_file,
                zip_file,
                youtube_preview,
                shorts_preview,
                word_preview,
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
    print("NEEDLE SUBTITLE")
    print(f"Languages: {len(LANGUAGES)}")
    print("Dropdown:", ", ".join(LANGUAGES))
    print(f"Model directory: {MODEL_DIR}")
    print("=" * 55)
    if not shutil.which("ffmpeg"):
        print("WARNING: ffmpeg not found in PATH")
    demo, launch_kwargs = build_ui()
    demo.queue().launch(
        server_name="127.0.0.1" if IS_LOCAL else "0.0.0.0",
        inbrowser=IS_LOCAL,
        show_error=True,
        **launch_kwargs,
    )
