from __future__ import annotations

import numpy as np
from scipy import signal


def remove_dc(signal_data: np.ndarray) -> np.ndarray:
    centered = np.asarray(signal_data, dtype=float)
    return centered - np.mean(centered, axis=0, keepdims=True)


def lowpass_preview(signal_data: np.ndarray, sampling_rate: float, cutoff_hz: float = 40.0) -> np.ndarray:
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    nyquist = sampling_rate / 2.0
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = signal.butter(4, normalized_cutoff, btype="low")
    return signal.filtfilt(b, a, np.asarray(signal_data, dtype=float), axis=0)


def build_preview_signal(
    signal_data: np.ndarray,
    sampling_rate: float,
    *,
    remove_baseline: bool = True,
    apply_lowpass: bool = False,
    cutoff_hz: float = 40.0,
) -> np.ndarray:
    preview = np.asarray(signal_data, dtype=float).copy()
    if remove_baseline:
        preview = remove_dc(preview)
    if apply_lowpass:
        preview = lowpass_preview(preview, sampling_rate, cutoff_hz=cutoff_hz)
    return preview
