#!/usr/bin/env python3
"""Trim, denoise, and lightly level the Bismillah recitation for memorization.

No reverb or echo — the voice should stay dry and clear for learning.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf
from scipy import signal

SR = 48000
SRC = Path("/workspace/bismillah.mp3")
OUT_WAV = Path("/workspace/bismillah_mastered.wav")
OUT_MP3 = Path("/workspace/bismillah_mastered.mp3")


def load_mono(path: Path, sr: int = SR) -> np.ndarray:
    tmp = Path("/tmp/bismillah_src.wav")
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-i", str(path),
            "-ac", "1", "-ar", str(sr), "-acodec", "pcm_f32le", str(tmp),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    audio, file_sr = sf.read(tmp, dtype="float32")
    assert file_sr == sr
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float64)


def rms_envelope(x: np.ndarray, sr: int, hop_ms: float = 8.0, win_ms: float = 30.0) -> np.ndarray:
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = max(hop, int(sr * win_ms / 1000.0))
    n = 1 + max(0, (len(x) - win) // hop)
    env = np.empty(n)
    for i in range(n):
        s = i * hop
        chunk = x[s : s + win]
        env[i] = np.sqrt(np.mean(chunk * chunk))
    return env


def detect_voice_bounds(x: np.ndarray, sr: int) -> tuple[int, int]:
    env = rms_envelope(x, sr)
    env_db = 20.0 * np.log10(env + 1e-12)
    noise_db = float(np.percentile(env_db, 15))
    peak_db = float(env_db.max())
    thresh = max(noise_db + 16.0, peak_db - 32.0, -50.0)
    active = env_db > thresh
    min_run = 4
    start_f = end_f = None
    run = 0
    for i, a in enumerate(active):
        run = run + 1 if a else 0
        if run >= min_run:
            start_f = i - min_run + 1
            break
    run = 0
    for i in range(len(active) - 1, -1, -1):
        run = run + 1 if active[i] else 0
        if run >= min_run:
            end_f = i + min_run - 1
            break
    if start_f is None or end_f is None:
        raise RuntimeError("Could not detect voiced region")
    hop = max(1, int(sr * 0.008))
    win = max(hop, int(sr * 0.030))
    start = max(0, start_f * hop)
    end = min(len(x), end_f * hop + win)
    return start, end


def highpass(x: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    sos = signal.butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x)


def high_shelf(x: np.ndarray, sr: int, freq: float, gain_db: float) -> np.ndarray:
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    cosw = np.cos(w0)
    sinw = np.sin(w0)
    alpha = sinw / 2 * np.sqrt((a + 1 / a) * (1 / 0.7 - 1) + 2)
    b0 = a * ((a + 1) + (a - 1) * cosw + 2 * np.sqrt(a) * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cosw)
    b2 = a * ((a + 1) + (a - 1) * cosw - 2 * np.sqrt(a) * alpha)
    a0 = (a + 1) - (a - 1) * cosw + 2 * np.sqrt(a) * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cosw)
    a2 = (a + 1) - (a - 1) * cosw - 2 * np.sqrt(a) * alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a_c = np.array([1.0, a1 / a0, a2 / a0])
    return signal.filtfilt(b, a_c, x)


def peak_eq(x: np.ndarray, sr: int, freq: float, q: float, gain_db: float) -> np.ndarray:
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    b0 = 1 + alpha * a
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / a
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a_c = np.array([1.0, a1 / a0, a2 / a0])
    return signal.filtfilt(b, a_c, x)


def fade(x: np.ndarray, sr: int, fade_in_ms: float, fade_out_ms: float) -> np.ndarray:
    n_in = int(sr * fade_in_ms / 1000.0)
    n_out = int(sr * fade_out_ms / 1000.0)
    y = x.copy()
    if n_in > 0:
        y[:n_in] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_in))
    if n_out > 0:
        y[-n_out:] *= 0.5 - 0.5 * np.cos(np.linspace(np.pi, 0, n_out))
    return y


def peak_normalize(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    ceiling = 10 ** (peak_db / 20.0)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-12:
        return x
    return x * (ceiling / peak)


def main() -> None:
    x = load_mono(SRC, SR)
    x = x - np.mean(x)

    start, end = detect_voice_bounds(x, SR)
    print(f"Voice bounds: {start / SR:.3f}s – {end / SR:.3f}s of {len(x) / SR:.3f}s")

    noise_end = min(start, max(int(0.2 * SR), start - int(0.12 * SR)))
    noise = x[:noise_end] if noise_end > int(0.2 * SR) else x[: int(0.8 * SR)]

    # Tiny pre-roll so the first letter is not clipped; no extra tail
    pre = int(0.035 * SR)
    post = int(0.060 * SR)
    body = x[max(0, start - pre) : min(len(x), end + post)]

    y = highpass(body, SR, 80.0)
    y = nr.reduce_noise(
        y=y.astype(np.float32),
        sr=SR,
        y_noise=noise.astype(np.float32),
        stationary=True,
        prop_decrease=0.65,
        n_fft=2048,
        hop_length=512,
        n_std_thresh_stationary=1.6,
    ).astype(np.float64)

    # Light cleanup only: cut hiss, slight presence for letter clarity
    y = high_shelf(y, SR, 10000.0, -2.5)
    y = peak_eq(y, SR, 3000.0, 1.0, 1.0)

    y = fade(y, SR, fade_in_ms=8.0, fade_out_ms=40.0)
    y = peak_normalize(y, peak_db=-1.0)
    y = y - np.mean(y)

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)))
    print(
        f"Out duration {len(y) / SR:.3f}s  peak {peak:.3f} "
        f"({20 * np.log10(peak + 1e-12):.1f} dB)  rms {rms:.3f}"
    )

    # Mono WAV; dual-mono MP3 so players that expect stereo still work
    sf.write(OUT_WAV, y.astype(np.float64), SR, subtype="PCM_24")
    stereo = np.stack([y, y], axis=1)
    tmp_stereo = Path("/tmp/bismillah_mastered_stereo.wav")
    sf.write(tmp_stereo, stereo.astype(np.float64), SR, subtype="PCM_24")
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-i", str(tmp_stereo),
            "-codec:a", "libmp3lame", "-b:a", "320k",
            "-ar", "48000", str(OUT_MP3),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Wrote {OUT_WAV} and {OUT_MP3}")


if __name__ == "__main__":
    main()
