# Needle Subtitle

A multilingual subtitle generator for audio and video, powered by **Cactus Needle / Needle 3**, **Udgam Labs Shruti**, and **Silero VAD**. The Gradio interface offers a full-width dark design with purple accents, a transcription progress indicator, and downloadable subtitle files.

## Features

- **29 language choices:** English, Hindi, Bengali first, followed by the other languages alphabetically.
- **Two speech-to-text backends:** Needle for seven European languages and Shruti (via Cactus `Whistle`) for 22 Indian languages.
- **Long recordings:** FFmpeg converts input to 16 kHz mono WAV; Silero VAD helps choose boundaries for chunks of up to 29 seconds.
- **Word timestamps:** Subtitle timing is based on the word timestamps returned by the model.
- **Local model cache:** Shruti model files are downloaded on first use and cached in `./model/<language>/`.
- **Stop control:** Requests a stop between processing steps; an in-progress transcription or download may need to finish its current operation.
- **Six output tabs:** YouTube SRT, Shorts/Reels SRT, word-level SRT, transcript TXT, original model response JSON, and a ZIP containing the exports.

## Supported languages

**Needle backend (7):** English, Dutch, French, German, Italian, Polish, Spanish.

**Shruti backend (22):** Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu.

The language menu places **English, Hindi, Bengali** first, then lists the remaining languages alphabetically. Model accuracy and availability vary by language.

## Requirements

- Python 3.10+ (check the supported Python versions for the installed Cactus packages)
- FFmpeg installed and available on `PATH`
- Internet access for initial package installation and model downloads
- Enough free disk space for the selected models and temporary audio

### Install

```bash
git clone https://github.com/NeuralFalconYT/needle-subtitle.git
cd needle-subtitle
python -m pip install -r requirements.txt
```

The application imports `needle` for the original seven languages and `Whistle` from `cactus` for Shruti. If you encounter `ModuleNotFoundError: No module named 'cactus'`, install the Cactus compute runtime compatible with your OS and Python version:

```bash
python -m pip install cactus-compute==4.1.0
```

For Needle, use the compatible `cactus-needle` version available for your environment. Check the repository's `requirements.txt` before installing; its current version pins may need adjustment.

Install FFmpeg from https://ffmpeg.org/download.html and verify:

```bash
ffmpeg -version
```

### Run

```bash
python app.py
```

Open the URL printed by Gradio, typically `http://127.0.0.1:7860` when running locally.

## Usage

1. Paste an audio/video file path when running locally, or upload a file through the web interface.
2. Select a language. The app selects Needle or Shruti based on the language.
3. Optionally adjust YouTube character limits, Shorts words per cue, uppercase, and trailing punctuation settings.
4. Click **Generate Subtitles**. Watch the progress indicator and download the files from the output tabs when finished.

**Note:** The local path field is hidden on Hugging Face Spaces. Upload a file instead.

## Outputs

The current application writes outputs under:

```text
./needle_output/<input-file-stem>/
```

The UI exposes six tabs:

| Tab | Output |
| --- | --- |
| 1 · YouTube / Horizontal | Single-line, sentence-style `.srt` |
| 2 · Shorts / Reels | Short, typically 1–3-word `.srt` cues |
| 3 · Word Level | One-word-per-cue `.srt` |
| 4 · Transcript TXT | Plain-text transcript |
| 5 · JSON Log | Original model responses, with chunk context |
| 6 · Export ZIP | Archive containing the generated exports |

The JSON log is useful for inspecting the model's returned timestamps and transcription results. If a chunk fails, check the console/log output.

## Processing pipeline

```text
Audio / Video
    |
    v
FFmpeg: 16 kHz, mono, 16-bit WAV
    |
    v
Silero VAD: speech boundaries
    |
    v
Chunks up to 29 seconds
    |
    +--> Needle (English, French, Spanish, German, Dutch, Italian, Polish)
    |
    +--> Shruti / Cactus Whistle (22 languages)
    |
    v
Word-level timestamps + original responses
    |
    v
YouTube SRT / Shorts SRT / Word SRT / TXT / JSON / ZIP
```

For Shruti, model files are cached under `model/<language>/`. The Hindi model uses `udgamlabs/Shruti`; other languages use language-specific repositories. If a repository name or file name differs from the naming convention expected by the app, download may fail and the mapping in `app.py` will need correction.

## Troubleshooting

- **FFmpeg not found:** Install FFmpeg and add its `bin` folder to your system `PATH`.
- **Cactus import fails:** Verify that `cactus-compute` is installed in the same Python environment running `app.py`.
- **Model download fails:** Check your internet connection and confirm the model repository and file names on Hugging Face.
- **No words returned:** Inspect the model response JSON (when generated) and terminal output; some audio or model/language combinations may not produce word timestamps.
- **Long transcription:** The application processes chunks sequentially, so total time depends on audio duration, model and hardware.

## Model credits and licensing

This project is an application/wrapper around third-party models and software. The original authors retain their respective rights. Review the upstream licenses and model usage terms before redistribution or commercial use.

- **Cactus Needle / Needle 3:** https://huggingface.co/Cactus-Compute/needle3
- **Needle source:** https://github.com/cactus-compute/needle
- **Udgam Labs Shruti:** https://huggingface.co/udgamlabs
- **Cactus compute runtime:** https://github.com/cactus-compute
- **Silero VAD:** https://github.com/snakers4/silero-vad
- **Gradio:** https://www.gradio.app/

## Disclaimer

This repository does not claim ownership of the third-party models, runtimes, or libraries used by the application. All applicable rights and terms remain with their respective upstream authors and projects.
