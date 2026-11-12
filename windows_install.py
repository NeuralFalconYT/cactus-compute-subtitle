
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import urllib.request
import zipfile
import tempfile
from pathlib import Path

# ============================================================
# CACTUS COMPUTE SUBTITLE - WINDOWS INSTALLER
# ============================================================

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
APP = ROOT / "app.py"
FFMPEG_DIR = ROOT / "FFmpeg"
LAUNCHER = ROOT / "run_cactus_subtitle.bat"

FFMPEG_URL = (
    "https://www.gyan.dev/ffmpeg/builds/"
    "ffmpeg-release-essentials.zip"
)


def header(message):
    print("\n" + "=" * 65)
    print(message)
    print("=" * 65)


def run(*args):
    command = [str(arg) for arg in args]
    print("\n>>>", subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


def check_windows():
    if os.name != "nt":
        raise RuntimeError(
            "This installer is intended for Windows."
        )


def check_project():
    header("Checking project files")

    for path in (APP, REQUIREMENTS):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    print("app.py found")
    print("requirements.txt found")


# ============================================================
# VIRTUAL ENVIRONMENT SELECTION
# ============================================================

def choose_environment():
    header("Python Installation Mode")

    print("Do you want to use a virtual environment?")
    print()
    print("[Y] Yes - Create/use .venv (Recommended)")
    print("[N] No  - Use current Python environment")
    print()

    while True:
        choice = input(
            "Use virtual environment? (Y/N): "
        ).strip().lower()

        if choice in ("y", "yes"):
            print("\nSelected: Virtual environment")
            return True

        if choice in ("n", "no"):
            print("\nSelected: Current Python environment")
            return False

        print("Invalid choice. Enter Y or N.")


def get_venv_python():
    return VENV / "Scripts" / "python.exe"


def setup_venv():
    header("1. Setting up virtual environment")

    python = get_venv_python()

    if python.is_file():
        result = subprocess.run(
            [str(python), "-c", "import sys"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Existing .venv is broken. "
                "Rename or delete it and rerun the installer."
            )

        print("Existing virtual environment detected.")
        print("Reusing:", VENV)
        return python

    print("Creating virtual environment...")

    run(sys.executable, "-m", "venv", VENV)

    if not python.is_file():
        raise RuntimeError(
            "Failed to create virtual environment."
        )

    print("Virtual environment created.")
    return python


def setup_python(use_venv):
    if use_venv:
        return setup_venv()

    header("1. Using current Python environment")

    python = Path(sys.executable).resolve()

    print("Python:", python)
    print("No virtual environment will be created.")
    print()
    print(
        "WARNING: Dependencies will be installed "
        "into this Python environment."
    )

    return python


# ============================================================
# DEPENDENCIES
# ============================================================

def install_dependencies(python):
    header("2. Installing Python dependencies")

    print("Using Python:", python)

    run(
        python,
        "-m",
        "pip",
        "--version",
    )

    run(
        python,
        "-m",
        "pip",
        "install",
        "-r",
        REQUIREMENTS,
    )

    print("\nDependencies installed successfully.")


# ============================================================
# FFMPEG
# ============================================================

def valid_executable(path):
    if not path or not Path(path).is_file():
        return False

    try:
        result = subprocess.run(
            [str(path), "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )

        return result.returncode == 0

    except (OSError, subprocess.TimeoutExpired):
        return False


def find_ffmpeg():
    # Prefer the project's local FFmpeg installation.
    if FFMPEG_DIR.exists():
        for ffmpeg in FFMPEG_DIR.rglob("ffmpeg.exe"):
            ffprobe = ffmpeg.with_name("ffprobe.exe")

            if (
                valid_executable(ffmpeg)
                and valid_executable(ffprobe)
            ):
                return ffmpeg.parent

    # Otherwise use FFmpeg installed on Windows PATH.
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if (
        ffmpeg
        and ffprobe
        and valid_executable(ffmpeg)
        and valid_executable(ffprobe)
    ):
        return Path(ffmpeg).parent

    return None


def download_ffmpeg(destination):
    print("Downloading FFmpeg...")
    print(FFMPEG_URL)

    curl = shutil.which("curl.exe")

    if curl:
        result = subprocess.run(
            [
                curl,
                "-L",
                "--fail",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--progress-bar",
                "-o",
                str(destination),
                FFMPEG_URL,
            ],
            check=False,
        )

        if result.returncode == 0:
            return

        print("\ncurl failed. Trying Python downloader.")

    def progress(count, block_size, total):
        if total > 0:
            downloaded = count * block_size
            percentage = min(
                100,
                downloaded * 100 / total,
            )

            print(
                f"\rDownloading: {percentage:.1f}%",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(
        FFMPEG_URL,
        destination,
        reporthook=progress,
    )

    print()


def safe_extract(archive, destination):
    destination = destination.resolve()

    for member in archive.infolist():
        target = (
            destination / member.filename
        ).resolve()

        if not target.is_relative_to(destination):
            raise RuntimeError(
                "Unsafe path found in FFmpeg ZIP."
            )

    archive.extractall(destination)


def setup_ffmpeg():
    header("3. Checking FFmpeg")

    existing = find_ffmpeg()

    if existing:
        print("FFmpeg already available:")
        print(existing)
        return existing

    print("FFmpeg not found.")
    print("Installing FFmpeg locally...")

    with tempfile.TemporaryDirectory(
        prefix="cactus_ffmpeg_"
    ) as temporary:
        temp = Path(temporary)
        archive_path = temp / "ffmpeg.zip"
        extracted = temp / "extracted"

        extracted.mkdir()

        download_ffmpeg(archive_path)

        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError(
                "Downloaded FFmpeg ZIP is invalid."
            )

        print("Extracting FFmpeg...")

        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, extracted)

        candidates = list(
            extracted.rglob("ffmpeg.exe")
        )

        if not candidates:
            raise RuntimeError(
                "ffmpeg.exe missing from archive."
            )

        source_bin = candidates[0].parent
        source_probe = source_bin / "ffprobe.exe"

        if not source_probe.is_file():
            raise RuntimeError(
                "ffprobe.exe missing from archive."
            )

        target_bin = FFMPEG_DIR / "bin"
        target_bin.mkdir(
            parents=True,
            exist_ok=True,
        )

        for filename in (
            "ffmpeg.exe",
            "ffprobe.exe",
            "ffplay.exe",
        ):
            source = source_bin / filename

            if source.is_file():
                shutil.copy2(
                    source,
                    target_bin / filename,
                )

    if not (
        valid_executable(target_bin / "ffmpeg.exe")
        and valid_executable(target_bin / "ffprobe.exe")
    ):
        raise RuntimeError(
            "FFmpeg verification failed."
        )

    print("FFmpeg installed successfully.")
    return target_bin


# ============================================================
# VERIFY INSTALLATION
# ============================================================

def verify_installation(python, ffmpeg_bin):
    header("4. Verifying installation")

    run(
        python,
        "-c",
        (
            "import gradio; "
            "import torch; "
            "import huggingface_hub; "
            "from cactus import Whistle; "
            "print('Python dependencies OK')"
        ),
    )

    ffmpeg = ffmpeg_bin / "ffmpeg.exe"
    ffprobe = ffmpeg_bin / "ffprobe.exe"

    if not (
        valid_executable(ffmpeg)
        and valid_executable(ffprobe)
    ):
        raise RuntimeError(
            "FFmpeg verification failed."
        )

    print("FFmpeg OK")
    print("Cactus runtime import OK")


# ============================================================
# WINDOWS LAUNCHER
# ============================================================

def create_launcher(python, ffmpeg_bin, use_venv):
    header("5. Creating Windows launcher")

    mode = (
        "Virtual Environment"
        if use_venv
        else "Current Python Environment"
    )

    # Escape percent signs because Windows BAT expands %VAR%.
    project_path = str(ROOT).replace("%", "%%")
    python_path = str(python).replace("%", "%%")
    app_path = str(APP).replace("%", "%%")
    ffmpeg_path = str(ffmpeg_bin).replace("%", "%%")

    content = f"""@echo off
setlocal
title Cactus Compute Subtitle

echo.
echo ==========================================================
echo                  CACTUS COMPUTE SUBTITLE
echo ==========================================================
echo.
echo Python Mode: {mode}
echo.

cd /d "{project_path}"
set "PATH={ffmpeg_path};%PATH%"
set "PYTHONUTF8=1"

echo Starting subtitle generator...
echo.

"{python_path}" "{app_path}"

set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ==========================================================
echo Application closed. Exit code: %EXIT_CODE%
echo ==========================================================
echo.

pause
exit /b %EXIT_CODE%
"""

    LAUNCHER.write_text(
        content,
        encoding="utf-8",
    )

    print("Launcher created:")
    print(LAUNCHER)
    print("Python mode:", mode)


# ============================================================
# MAIN INSTALLER
# ============================================================

def main():
    header(
        "CACTUS COMPUTE SUBTITLE - WINDOWS INSTALLER"
    )

    check_windows()
    check_project()

    print("Project:", ROOT)
    print("Python:", sys.version.split()[0])

    use_venv = choose_environment()

    python = setup_python(use_venv)

    install_dependencies(python)

    ffmpeg_bin = setup_ffmpeg()

    verify_installation(
        python,
        ffmpeg_bin,
    )

    create_launcher(
        python,
        ffmpeg_bin,
        use_venv,
    )

    header("INSTALLATION COMPLETE")

    print("Everything is ready!")
    print()
    print("Python:", python)
    print(
        "Virtual environment:",
        "YES" if use_venv else "NO",
    )
    print()
    print("Start the application using:")
    print("  run_cactus_subtitle.bat")
    print()
    print(
        "Models will download automatically "
        "when you select a language for the first time."
    )
    print()
    print("Project:", ROOT)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        sys.exit(130)

    except Exception as error:
        header("INSTALLATION FAILED")
        print(f"\nError: {error}")
        sys.exit(1)

    finally:
        if os.name == "nt" and sys.stdin.isatty():
            input("\nPress Enter to exit...")
