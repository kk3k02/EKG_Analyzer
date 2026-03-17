from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import welch
except ImportError:  # pragma: no cover - exercised by fallback path tests
    welch = None


MIN_SAMPLES_FOR_ANALYSIS = 16
DEFAULT_MAX_FREQUENCY_HZ = 40.0


@dataclass(slots=True)
class FrequencyOverviewResult:
    frequencies_hz: np.ndarray
    values: np.ndarray
    y_label: str
    method: str
    message: str | None = None


def compute_frequency_overview(
    signal_segment: np.ndarray,
    sampling_rate: float,
    *,
    max_frequency_hz: float = DEFAULT_MAX_FREQUENCY_HZ,
) -> FrequencyOverviewResult:
    """Compute a frequency-domain overview for one ECG lead segment."""
    if sampling_rate <= 0:
        return FrequencyOverviewResult(
            frequencies_hz=np.array([], dtype=float),
            values=np.array([], dtype=float),
            y_label="Power spectral density",
            method="invalid",
            message="Sampling rate is missing or invalid.",
        )

    segment = np.asarray(signal_segment, dtype=float).reshape(-1)
    finite_mask = np.isfinite(segment)
    if not np.any(finite_mask):
        return FrequencyOverviewResult(
            frequencies_hz=np.array([], dtype=float),
            values=np.array([], dtype=float),
            y_label="Power spectral density",
            method="invalid",
            message="Signal segment is empty.",
        )

    segment = segment[finite_mask]
    if segment.size < MIN_SAMPLES_FOR_ANALYSIS:
        return FrequencyOverviewResult(
            frequencies_hz=np.array([], dtype=float),
            values=np.array([], dtype=float),
            y_label="Power spectral density",
            method="too_short",
            message="Visible signal segment is too short for frequency analysis.",
        )

    segment = segment - float(np.mean(segment))

    if welch is not None:
        frequencies_hz, values = _compute_welch_psd(segment, sampling_rate)
        y_label = "Power spectral density"
        method = "welch"
    else:
        frequencies_hz, values = _compute_fft_amplitude(segment, sampling_rate)
        y_label = "Amplitude spectrum"
        method = "fft"

    frequency_mask = (frequencies_hz >= 0.0) & (frequencies_hz <= max_frequency_hz)
    frequencies_hz = frequencies_hz[frequency_mask]
    values = values[frequency_mask]

    if frequencies_hz.size == 0:
        return FrequencyOverviewResult(
            frequencies_hz=np.array([], dtype=float),
            values=np.array([], dtype=float),
            y_label=y_label,
            method=method,
            message="No frequency bins are available in the selected range.",
        )

    return FrequencyOverviewResult(
        frequencies_hz=frequencies_hz,
        values=values,
        y_label=y_label,
        method=method,
    )


def _compute_welch_psd(signal_segment: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(1024, signal_segment.size)
    if nperseg < MIN_SAMPLES_FOR_ANALYSIS:
        nperseg = signal_segment.size
    frequencies_hz, values = welch(signal_segment, fs=sampling_rate, nperseg=nperseg, scaling="density")
    return np.asarray(frequencies_hz, dtype=float), np.asarray(values, dtype=float)


def _compute_fft_amplitude(signal_segment: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    frequencies_hz = np.fft.rfftfreq(signal_segment.size, d=1.0 / float(sampling_rate))
    values = np.abs(np.fft.rfft(signal_segment)) / float(signal_segment.size)
    return np.asarray(frequencies_hz, dtype=float), np.asarray(values, dtype=float)
