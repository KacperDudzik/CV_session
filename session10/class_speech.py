"""Session 10 — Speech Recognition.

Converted from `class_speech.ipynb` into reusable functions you can run one by one.

Idea in one line: sound -> list of numbers -> spectrogram (an "image") -> Transformer (Whisper) -> text.

Plan (as in class):
1. Load a .wav — a sound is a list of numbers (16,000 per second).
2. Spectrogram — short windows -> frequency × time image (STFT).
3. Whisper — pretrained Transformer encoder + decoder -> text.
4. Optional: your own recording (mono, 16 kHz).

Needs: soundfile, torch, transformers, scipy (for own-voice resample).
First Whisper run downloads ~290 MB (openai/whisper-base).

Run examples:
  python class_speech.py load
  python class_speech.py spectrogram
  python class_speech.py whisper
  python class_speech.py own
  python class_speech.py all
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Paths, constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
SAMPLE_WAV = HERE / "sample_command.wav"
MY_VOICE = HERE / "my_voice.wav"
OUTPUT_DIR = HERE / "_output"

# Whisper / STFT settings used in the notebook
TARGET_SR = 16_000
N_FFT = 400          # 25 ms at 16 kHz
HOP_LENGTH = 160     # 10 ms at 16 kHz
WHISPER_MODEL = "openai/whisper-base"


def savefig(fig: plt.Figure, name: str) -> Path:
    """Save a figure under _output/ and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


# ---------------------------------------------------------------------------
# 1. A sound is a list of numbers
# ---------------------------------------------------------------------------

def load_audio(path: str | Path = SAMPLE_WAV) -> tuple[np.ndarray, int]:
    """Load a .wav as float32 samples + sampling rate.

    A microphone measures air pressure TARGET_SR times per second;
    soundfile just reads those numbers from disk.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {path}")
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo -> mono
    print(f"{len(audio):,} numbers at {sr:,} per second = {len(audio) / sr:.2f} seconds")
    return audio, int(sr)


def peek_samples(audio: np.ndarray, start: int = 30_000, n: int = 8) -> np.ndarray:
    """Print a few samples from the middle — proof that sound is just numbers."""
    chunk = audio[start : start + n]
    print("8 numbers from the middle:", np.round(chunk, 4))
    return chunk


def demo_load(path: str | Path = SAMPLE_WAV) -> tuple[np.ndarray, int]:
    """Step 1 — load sample_command.wav and peek at the numbers."""
    print("=== 1. Load audio (sound = list of numbers) ===")
    audio, sr = load_audio(path)
    peek_samples(audio)
    return audio, sr


# ---------------------------------------------------------------------------
# 2. Spectrogram: list of numbers -> image (frequency × time)
# ---------------------------------------------------------------------------

def make_spectrogram(
    audio: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """STFT magnitude spectrogram: shape (frequencies, time steps).

    Windows of n_fft samples (25 ms at 16 kHz), one every hop_length (10 ms).
    """
    wave = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    spec = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft),
        return_complex=True,
    ).abs()
    print("spectrogram shape (frequencies, time steps):", tuple(spec.shape))
    return spec


def plot_waveform_and_spectrogram(
    audio: np.ndarray,
    sr: int,
    spec: torch.Tensor | None = None,
    save_name: str = "waveform_spectrogram.png",
) -> Path:
    """Waveform (1-D) on top, log-spectrogram (2-D) below — same sound, two views."""
    if spec is None:
        spec = make_spectrogram(audio)

    t = np.arange(len(audio)) / sr
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2))
    axes[0].plot(t, audio, color="#2E6DB4", lw=0.5)
    axes[0].set_xlim(0, t[-1])
    axes[0].set_title("waveform: one number per sample")
    axes[0].set_xlabel("seconds")

    axes[1].imshow(
        (spec + 1e-6).log10(),
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, t[-1], 0, sr / 2000],
    )
    axes[1].set_title("spectrogram: an image (frequency x time)")
    axes[1].set_xlabel("seconds")
    axes[1].set_ylabel("kHz")
    plt.tight_layout()
    return savefig(fig, save_name)


def demo_spectrogram(path: str | Path = SAMPLE_WAV) -> torch.Tensor:
    """Step 2 — build and plot the spectrogram for the sample command."""
    print("=== 2. Spectrogram (numbers -> image) ===")
    audio, sr = load_audio(path)
    spec = make_spectrogram(audio)
    plot_waveform_and_spectrogram(audio, sr, spec)
    return spec


# ---------------------------------------------------------------------------
# 3. Whisper: Transformer that listens
# ---------------------------------------------------------------------------

def load_whisper(model_id: str = WHISPER_MODEL, device: str = "cpu"):
    """Load the Hugging Face automatic-speech-recognition pipeline (Whisper)."""
    from transformers import pipeline, logging as hf_logging

    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore")
    print(f"loading Whisper ({model_id}) on {device} …")
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device)
    print("ready")
    return asr


def transcribe(
    asr,
    audio: np.ndarray,
    sr: int = TARGET_SR,
    language: str | None = "en",
    task: str = "transcribe",
) -> str:
    """Run Whisper: spectrogram -> encoder -> decoder -> text.

    Pass language=None to let Whisper detect the language.
    Use task='translate' to force English translation.
    """
    kwargs: dict = {"task": task}
    if language is not None:
        kwargs["language"] = language
    result = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs=kwargs)
    text = result["text"].strip()
    print("Whisper heard:", text)
    return text


def demo_whisper(path: str | Path = SAMPLE_WAV) -> str:
    """Step 3 — transcribe sample_command.wav with Whisper-base."""
    print("=== 3. Whisper transcription ===")
    audio, sr = load_audio(path)
    asr = load_whisper()
    return transcribe(asr, audio, sr, language="en", task="transcribe")


# ---------------------------------------------------------------------------
# 4. Optional: your own voice
# ---------------------------------------------------------------------------

def prepare_recording(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any .wav, force mono, resample to 16 kHz (what Whisper expects)."""
    from scipy.signal import resample_poly

    path = Path(path)
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr).astype("float32")
        print(f"resampled {sr} -> {TARGET_SR} Hz")
    else:
        print(f"already {TARGET_SR} Hz")
    print(f"{len(audio):,} samples = {len(audio) / TARGET_SR:.2f} s")
    return audio, TARGET_SR


def demo_own_voice(
    path: str | Path = MY_VOICE,
    asr=None,
    task: str = "transcribe",
) -> str | None:
    """Step 4 — transcribe my_voice.wav if it exists next to this script."""
    print("=== 4. Own recording (optional) ===")
    path = Path(path)
    if not path.is_file():
        print(f"No file called {path.name} yet: record one next to class_speech.py and re-run.")
        return None
    mine, sr = prepare_recording(path)
    if asr is None:
        asr = load_whisper()
    # language=None -> Whisper detects language itself
    return transcribe(asr, mine, sr, language=None, task=task)


# ---------------------------------------------------------------------------
# CLI runners
# ---------------------------------------------------------------------------

def run_all() -> None:
    """Run steps 1–3 (and 4 if my_voice.wav is present)."""
    audio, sr = demo_load()
    print()
    spec = make_spectrogram(audio)
    plot_waveform_and_spectrogram(audio, sr, spec)
    print()
    asr = load_whisper()
    transcribe(asr, audio, sr, language="en", task="transcribe")
    print()
    demo_own_voice(asr=asr)


COMMANDS = {
    "load": demo_load,
    "spectrogram": demo_spectrogram,
    "whisper": demo_whisper,
    "own": demo_own_voice,
    "all": run_all,
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print("Usage: python class_speech.py <command>")
        print("Commands:", ", ".join(COMMANDS))
        return
    cmd = argv[0]
    if cmd not in COMMANDS:
        raise SystemExit(f"Unknown command {cmd!r}. Choose from: {', '.join(COMMANDS)}")
    COMMANDS[cmd]()


if __name__ == "__main__":
    main()
