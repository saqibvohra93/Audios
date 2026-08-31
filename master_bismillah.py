#!/usr/bin/env python3
"""Clean, trim, and master the Bismillah recitation with natural mosque ambience."""

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


def db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def rms_envelope(x: np.ndarray, sr: int, hop_ms: float = 10.0, win_ms: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = max(hop, int(sr * win_ms / 1000.0))
    n = 1 + max(0, (len(x) - win) // hop)
    env = np.empty(n)
    times = np.empty(n)
    for i in range(n):
        s = i * hop
        chunk = x[s : s + win]
        env[i] = np.sqrt(np.mean(chunk * chunk))
        times[i] = (s + win / 2) / sr
    return times, env


def detect_voice_bounds(x: np.ndarray, sr: int) -> tuple[int, int]:
    """Find first/last voiced samples using an adaptive RMS threshold."""
    times, env = rms_envelope(x, sr, hop_ms=8.0, win_ms=30.0)
    env_db = 20.0 * np.log10(env + 1e-12)
    # Noise floor from the quietest 15% of frames
    noise_db = float(np.percentile(env_db, 15))
    peak_db = float(env_db.max())
    thresh = max(noise_db + 16.0, peak_db - 32.0, -50.0)
    active = env_db > thresh
    # require a short run of active frames so a click doesn't count
    min_run = 4
    start_f = None
    run = 0
    for i, a in enumerate(active):
        run = run + 1 if a else 0
        if run >= min_run:
            start_f = i - min_run + 1
            break
    end_f = None
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


def highpass(x: np.ndarray, sr: int, cutoff: float = 75.0) -> np.ndarray:
    sos = signal.butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = signal.butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x)


def peak_eq(x: np.ndarray, sr: int, freq: float, q: float, gain_db: float) -> np.ndarray:
    """Second-order peaking EQ (RBJ cookbook), zero-phase via filtfilt-style SOS."""
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


def deess(x: np.ndarray, sr: int, freq: float = 7000.0, thresh_db: float = -22.0, ratio: float = 3.0) -> np.ndarray:
    """Gentle broadband de-esser driven by 5–9 kHz energy."""
    sos = signal.butter(2, [5000, 9000], btype="bandpass", fs=sr, output="sos")
    band = signal.sosfilt(sos, x)
    hop = int(sr * 0.004)
    win = int(sr * 0.012)
    gain = np.ones(len(x))
    thresh = 10 ** (thresh_db / 20.0)
    for i in range(0, len(x) - win, hop):
        e = np.sqrt(np.mean(band[i : i + win] ** 2))
        if e > thresh:
            over = e / thresh
            red = over ** (1.0 - 1.0 / ratio)
            g = 1.0 / red
            gain[i : i + win] = np.minimum(gain[i : i + win], g)
    # smooth gain
    sos_g = signal.butter(1, 30.0, btype="lowpass", fs=sr, output="sos")
    gain = signal.sosfiltfilt(sos_g, gain)
    gain = np.clip(gain, 0.55, 1.0)
    return x * gain


def compress(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 2.0,
    attack_ms: float = 12.0,
    release_ms: float = 90.0,
    makeup_db: float = 1.5,
) -> np.ndarray:
    thresh = 10 ** (threshold_db / 20.0)
    atk = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    rel = np.exp(-1.0 / (sr * release_ms / 1000.0))
    env = 0.0
    gain = np.empty_like(x)
    for i, s in enumerate(x):
        mag = abs(s)
        if mag > env:
            env = atk * env + (1 - atk) * mag
        else:
            env = rel * env + (1 - rel) * mag
        if env > thresh:
            # soft knee ~6 dB
            over = env / thresh
            desired = thresh * (over ** (1.0 / ratio))
            gain[i] = desired / (env + 1e-12)
        else:
            gain[i] = 1.0
    makeup = 10 ** (makeup_db / 20.0)
    return x * gain * makeup


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0)
    spec[1:] /= np.sqrt(np.maximum(freqs[1:], 1e-12))
    y = np.fft.irfft(spec, n)
    y /= np.max(np.abs(y)) + 1e-12
    return y


def mosque_impulse(sr: int, rt60: float = 1.65, length_s: float = 2.6) -> tuple[np.ndarray, np.ndarray]:
    """Warm stone-hall IR: early reflections + damped late reverb, stereo decorrelated."""
    n = int(sr * length_s)
    t = np.arange(n) / sr
    env = np.exp(-t * (6.907755 / rt60))
    # extra HF damping envelope (highs die faster)
    hf_env = np.exp(-t * (6.907755 / (rt60 * 0.55)))

    rng_l = np.random.default_rng(13)
    rng_r = np.random.default_rng(29)
    late_l = pink_noise(n, rng_l) * env
    late_r = pink_noise(n, rng_r) * env
    # warm band-limit on the tail
    sos = signal.butter(2, [180, 5200], btype="band", fs=sr, output="sos")
    late_l = signal.sosfilt(sos, late_l)
    late_r = signal.sosfilt(sos, late_r)
    # tilt highs down over time
    late_l = lowpass(late_l * hf_env + late_l * (1 - hf_env) * 0.35, sr, 6500)
    late_r = lowpass(late_r * hf_env + late_r * (1 - hf_env) * 0.35, sr, 6500)

    ir_l = np.zeros(n)
    ir_r = np.zeros(n)
    # Early reflections typical of a modest mosque hall (ms, L, R)
    taps = [
        (17, 0.62, 0.20),
        (24, 0.22, 0.55),
        (33, 0.38, 0.18),
        (41, 0.16, 0.34),
        (52, 0.24, 0.14),
        (64, 0.12, 0.22),
        (78, 0.14, 0.10),
        (93, 0.08, 0.13),
        (112, 0.09, 0.06),
        (134, 0.05, 0.08),
        (158, 0.05, 0.04),
        (186, 0.03, 0.05),
    ]
    for dms, al, ar in taps:
        i = int(round(dms * sr / 1000.0))
        if i < n:
            ir_l[i] += al
            ir_r[i] += ar

    late_gain = 0.42
    ir_l += late_l * late_gain / (np.max(np.abs(late_l)) + 1e-12)
    ir_r += late_r * late_gain / (np.max(np.abs(late_r)) + 1e-12)

    # Normalize IR energy so wet mix is predictable
    e = np.sqrt(0.5 * (np.sum(ir_l**2) + np.sum(ir_r**2)))
    ir_l /= e
    ir_r /= e
    return ir_l, ir_r


def apply_echo(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Subtle stereo echo (mosque courtyard), not a karaoke slap-back."""
    delays = [
        (92, 0.09, 0.035),   # ms, L, R
        (168, 0.035, 0.075),
        (245, 0.028, 0.020),
    ]
    y_l = x.copy()
    y_r = x.copy()
    for dms, al, ar in delays:
        d = int(round(dms * sr / 1000.0))
        if d >= len(x):
            continue
        delayed = np.concatenate([np.zeros(d), x[:-d]])
        # slightly darken repeats so they sit behind the voice
        delayed = lowpass(delayed, sr, 4200)
        y_l += delayed * al
        y_r += delayed * ar
    return y_l, y_r


def true_peak_limit(x: np.ndarray, ceiling: float = 10 ** (-1.0 / 20.0)) -> np.ndarray:
    """Look-ahead peak limiter (~3 ms) with 4x oversampled peak estimate."""
    mag = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    over = signal.resample_poly(mag, 4, 1)
    la = max(1, int(0.003 * SR * 4))
    pad = np.pad(over, (la, 0), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view

    look = sliding_window_view(pad, la + 1).max(axis=1)[: len(over)]
    gain_os = np.where(look > ceiling, ceiling / (look + 1e-12), 1.0)
    # 8 ms smoothing
    alpha = np.exp(-1.0 / (SR * 4 * 0.008))
    gain_os = signal.lfilter([1 - alpha], [1, -alpha], gain_os)
    gain = signal.resample_poly(gain_os, 1, 4)
    if len(gain) < len(mag):
        gain = np.pad(gain, (0, len(mag) - len(gain)), mode="edge")
    gain = gain[: len(mag)]
    y = x * (gain[:, None] if x.ndim == 2 else gain)
    peak = float(np.max(np.abs(y)))
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y


def fade(x: np.ndarray, sr: int, fade_in_ms: float, fade_out_ms: float) -> np.ndarray:
    n_in = int(sr * fade_in_ms / 1000.0)
    n_out = int(sr * fade_out_ms / 1000.0)
    y = x.copy()
    if n_in > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_in))
        if y.ndim == 1:
            y[:n_in] *= ramp
        else:
            y[:n_in] *= ramp[:, None]
    if n_out > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(np.pi, 0, n_out))
        if y.ndim == 1:
            y[-n_out:] *= ramp
        else:
            y[-n_out:] *= ramp[:, None]
    return y


def lufs_approx(x: np.ndarray, sr: int) -> float:
    """K-weighted integrated loudness approximation (mono or stereo)."""
    if x.ndim == 1:
        left = right = x
    else:
        left, right = x[:, 0], x[:, 1]
    # K-weight: pre-filter + RLB
    sos1 = signal.butter(1, 1500, btype="highpass", fs=sr, output="sos")  # rough HS
    # BS.1770 pre-filter approximated by high shelf + highpass
    b, a = signal.iirfilter(2, 38.0, btype="highpass", ftype="butter", fs=sr)
    m = 0.5 * (left + right) if x.ndim > 1 else x
    m = signal.lfilter(b, a, m)
    # high shelf ~4dB at 1.5k
    m = high_shelf(m, sr, 1500.0, 4.0)
    mean_sq = np.mean(m**2)
    return -0.691 + 10 * np.log10(mean_sq + 1e-12)


def main() -> None:
    x = load_mono(SRC, SR)
    x = x - np.mean(x)

    start, end = detect_voice_bounds(x, SR)
    print(f"Voice bounds: {start/SR:.3f}s – {end/SR:.3f}s of {len(x)/SR:.3f}s")

    # Noise profile from leading silence (exclude a tiny pre-roll)
    noise_end = max(int(0.15 * SR), start - int(0.12 * SR))
    noise_end = min(noise_end, start)
    noise = x[:noise_end] if noise_end > int(0.2 * SR) else x[: int(0.8 * SR)]

    # Keep a short pre-roll so the first consonant isn't clipped, but no dead air
    pre = int(0.035 * SR)
    body = x[max(0, start - pre) : end]

    # --- cleanup ---
    y = highpass(body, SR, 75.0)
    y = nr.reduce_noise(
        y=y.astype(np.float32),
        sr=SR,
        y_noise=noise.astype(np.float32),
        stationary=True,
        prop_decrease=0.72,
        n_fft=2048,
        hop_length=512,
        n_std_thresh_stationary=1.6,
    ).astype(np.float64)

    # Tame the unusually strong 8–16 kHz band (hiss / mp3 grit) without dulling the voice
    y = high_shelf(y, SR, 9500.0, -3.5)
    y = peak_eq(y, SR, 280.0, 0.9, 1.2)     # warmth
    y = peak_eq(y, SR, 3200.0, 1.1, 1.6)    # presence / intelligibility
    y = peak_eq(y, SR, 6500.0, 1.4, -1.2)   # slight harshness cut
    y = deess(y, SR, thresh_db=-24.0, ratio=2.6)
    y = compress(y, SR, threshold_db=-16.0, ratio=2.1, attack_ms=14.0, release_ms=100.0, makeup_db=1.8)

    # Pad for reverb/echo tail before convolution
    tail_pad = int(2.4 * SR)
    dry = np.concatenate([y, np.zeros(tail_pad)])

    # Echo (subtle) then convolution reverb
    echo_l, echo_r = apply_echo(dry, SR)

    ir_l, ir_r = mosque_impulse(SR, rt60=1.85, length_s=2.7)
    predelay = int(0.028 * SR)
    dry_pd = np.concatenate([np.zeros(predelay), dry])
    rev_l = signal.fftconvolve(dry_pd, ir_l, mode="full")[: len(echo_l)]
    rev_r = signal.fftconvolve(dry_pd, ir_r, mode="full")[: len(echo_r)]

    wet = 0.22  # reverb send — present mosque hall, recitation stays forward
    left = echo_l + wet * rev_l
    right = echo_r + wet * rev_r

    # Keep the reciter centered; let only ambience occupy the sides
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right) * 0.78
    left = mid + side
    right = mid - side
    stereo = np.stack([left, right], axis=1)

    # Trim trailing silence after the reverb has died away, keep a short natural tail
    mag = np.max(np.abs(stereo), axis=1)
    # find last sample above -48 dB after the dry voice end
    voice_end = len(y)
    thresh = 10 ** (-48.0 / 20.0)
    last = np.where(mag[voice_end:] > thresh)[0]
    if len(last):
        cut = voice_end + int(last[-1]) + int(0.08 * SR)
    else:
        cut = voice_end + int(0.9 * SR)
    cut = min(len(stereo), cut)
    stereo = stereo[:cut]

    stereo = fade(stereo, SR, fade_in_ms=8.0, fade_out_ms=280.0)

    # Loudness: aim about -16 LUFS, true peak -1 dBTP
    loud = lufs_approx(stereo, SR)
    target = -16.0
    stereo = stereo * (10 ** ((target - loud) / 20.0))
    stereo = true_peak_limit(stereo, ceiling=10 ** (-1.0 / 20.0))

    # DC / tiny denormal cleanup
    stereo = stereo - np.mean(stereo, axis=0)

    peak = float(np.max(np.abs(stereo)))
    rms = float(np.sqrt(np.mean(stereo**2)))
    print(f"Out duration {len(stereo)/SR:.3f}s  peak {peak:.3f} ({db(np.array([peak]))[0]:.1f} dB)  rms {rms:.3f}")
    print(f"LUFS~ {lufs_approx(stereo, SR):.1f}  L/R corr {np.corrcoef(stereo[:,0], stereo[:,1])[0,1]:.3f}")

    sf.write(OUT_WAV, stereo.astype(np.float64), SR, subtype="PCM_24")
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-i", str(OUT_WAV),
            "-codec:a", "libmp3lame", "-b:a", "320k",
            "-ar", "48000", str(OUT_MP3),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Wrote {OUT_WAV} and {OUT_MP3}")


if __name__ == "__main__":
    main()
