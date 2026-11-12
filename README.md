# Cactus Compute Subtitle Generation

A lightweight, multilingual subtitle generator for audio and video, powered by **Cactus Compute ASR**, **Whistle**, **Udgam Labs Shruti**, and **Silero VAD**.

Built with Gradio, the app supports **29 languages**, word-level timestamps, automatic model downloads, and multiple ready-to-use subtitle formats.

## Features

- **29 supported languages:** English, Hindi, and Bengali appear first in the language selector; the others follow alphabetically.
- **Unified Cactus ASR runtime:** Both Whistle and regional Shruti models run through `from cactus import Whistle`.
- **Audio and video input:** Upload a file through Gradio, or paste a file path when running locally.
- **Long-audio support:** FFmpeg converts input to 16 kHz mono WAV; Silero VAD helps select speech boundaries for chunks of up to 29 seconds.
- **Word timestamps:** Timing is based on the timestamps returned by the ASR model.
- **Automatic model downloads and cache:** `.cact` models are downloaded on first use and cached under `./model/`.
- **Live progress:** The Status area includes an animated progress bar and processing stage; the Summary shows multiline logs.
- **Stop control:** Requests a stop between operations; an active download or transcription may need to finish first.
- **Six export tabs:** Word Level SRT, Sentence SRT, Short SRT, Transcript TXT, JSON Log, and Export ZIP.
- **Local and Hugging Face Spaces support:** The local file-path field is hidden on Spaces; a ZeroGPU registration hook is included.

## Supported languages

| Model | Languages | Count |
| --- | --- | ---: |
| **Cactus Whistle** | English, Dutch, French, German, Italian, Polish, Spanish | 7 |
| **Udgam Labs Shruti** | Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu | 22 |
| **Total** | | **29** |

Whistle uses [`Cactus-Compute/whistle`](https://huggingface.co/Cactus-Compute/whistle). For the other 22 languages, the app selects the corresponding Shruti model repository. Hindi uses [`udgamlabs/Shruti`](https://huggingface.co/udgamlabs/Shruti).

> Recognition accuracy, word-timestamp quality, and model availability depend on the language, audio quality, and selected model.

## Requirements

- Python compatible with the versions in `requirements.txt`
- FFmpeg available to the application
- Internet connection for installing dependencies and downloading models on first use
- Free disk space for model files and temporary audio

### Installation

```bash
git clone https://github.com/NeuralFalconYT/cactus-compute-subtitle.git
cd cactus-compute-subtitle
python -m pip install -r requirements.txt
```

This repository currently specifies `cactus-compute==4.1.0` in `requirements.txt`. The app imports `Whistle` from the `cactus` module; it does **not** require the separate `cactus-needle` package.

Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and verify it is available:

```bash
ffmpeg -version
```

### Run locally

```bash
python app.py
```

Open the local URL printed by Gradio (often `http://127.0.0.1:7860`).

### Windows installation helper

The repo includes `windows_install.py`, intended to set up a virtual environment and FFmpeg. **At the time of this README update, that script still contains legacy `run_needle_subtitle.bat` naming and an initial Markdown code fence.** Clean up those issues before running it as Python. For now, use the manual installation steps above.

## Usage

1. Upload an audio/video file, or paste its file path when running locally.
2. Choose the transcription language.
3. Optionally adjust the Sentence SRT character limit, Short SRT words per cue, uppercase formatting, and punctuation options.
4. Click **Generate Subtitles** and follow the Status progress bar and Summary logs.
5. Download the subtitle files individually or use **Export ZIP**.

The **Paste file path (local only)** input is automatically hidden on Hugging Face Spaces (`visible=IS_LOCAL`).

## Outputs

The app saves generated files to:

```text
cactus_subtitle_output/
└── <input-file-stem>/
    ├── audio_16k.wav
    ├── word_level_<input-file-stem>.srt
    ├── sentence_<input-file-stem>.srt
    ├── short_<input-file-stem>.srt
    ├── transcript_<input-file-stem>.txt
    ├── original_model_<input-file-stem>.json
    └── subtitles_<input-file-stem>.zip
```

| Tab | Export |
| --- | --- |
| **1 · Word Level SRT** | One word per SRT cue |
| **2 · Sentence SRT** | Readable grouped subtitles with a configurable character limit |
| **3 · Short SRT** | Short subtitle cues with configurable words per cue |
| **4 · Transcript TXT** | Plain-text transcript |
| **5 · JSON Log** | Original chunk-level ASR model responses, with timing and error information |
| **6 · Export ZIP** | ZIP containing the three SRTs, transcript, and original model JSON |

The ZIP includes the five export files; the intermediate WAV is not included.

## Processing pipeline

```text
Audio / Video
     |
     v
FFmpeg: 16 kHz mono WAV
     |
     v
Silero VAD: speech boundaries
     |
     v
Audio chunks up to 29 seconds
     |
     +--> Cactus Whistle (7 languages)
     |
     +--> Udgam Labs Shruti (22 languages)
     |
     v
Cactus Whistle runtime: speech recognition + word timestamps
     |
     v
Word Level SRT / Sentence SRT / Short SRT
Transcript TXT / Original Model JSON / Export ZIP
```

Models are cached under `model/whistle/` or `model/<language>/`. The app detects the `.cact` file name in each selected Hugging Face repository instead of hardcoding it.

## Troubleshooting

- **`No module named 'cactus'`:** Install `requirements.txt` in the Python environment that launches `app.py`.
- **FFmpeg not found:** Install FFmpeg and ensure its executable is on `PATH`.
- **Model download fails:** Check your internet connection, model repository availability, and disk space.
- **No words returned:** Inspect the Summary and original JSON log (if generated), and verify the audio contains clear speech.
- **Long processing times:** Transcription happens chunk by chunk, and performance depends on your hardware and selected model.
- **ZeroGPU startup errors on Spaces:** Verify the Space hardware and the `spaces` Python dependency.

## Credits and licenses

This repository is an independent interface built around third-party software and models. The original authors retain their respective rights. Please review upstream licenses and usage terms.

- [Cactus Compute Whistle](https://huggingface.co/Cactus-Compute/whistle)
- [Udgam Labs Shruti](https://huggingface.co/udgamlabs)
- [Cactus Compute](https://github.com/cactus-compute)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Gradio](https://www.gradio.app/)
- [FFmpeg](https://ffmpeg.org/)

## Disclaimer

This project does not claim ownership of the underlying ASR models, runtimes, or libraries. Applicable licenses and terms remain with their respective developers.
