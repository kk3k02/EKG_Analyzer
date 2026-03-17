from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np

try:
    from scipy import signal as scipy_signal
except ImportError:  # pragma: no cover - exercised by fallback tests
    scipy_signal = None


DEFAULT_HIGHPASS_CUTOFF_HZ = 0.5
DEFAULT_LOWPASS_CUTOFF_HZ = 40.0
DEFAULT_NOTCH_Q = 30.0
SUPPORTED_MAINS_FREQUENCIES_HZ = (50.0, 60.0)
FILTER_ORDER = 4


@dataclass(slots=True)
class CutoffFilterConfig:
    enabled: bool = False
    cutoff: float = DEFAULT_LOWPASS_CUTOFF_HZ


@dataclass(slots=True)
class BandpassFilterConfig:
    enabled: bool = False
    low: float = DEFAULT_HIGHPASS_CUTOFF_HZ
    high: float = DEFAULT_LOWPASS_CUTOFF_HZ


@dataclass(slots=True)
class NotchFilterConfig:
    enabled: bool = False
    mains_frequency_hz: float = 50.0
    quality_factor: float = DEFAULT_NOTCH_Q


@dataclass(slots=True)
class SignalFilterConfig:
    dc_removal: bool = False
    highpass: CutoffFilterConfig = field(default_factory=lambda: CutoffFilterConfig(cutoff=DEFAULT_HIGHPASS_CUTOFF_HZ))
    lowpass: CutoffFilterConfig = field(default_factory=lambda: CutoffFilterConfig(cutoff=DEFAULT_LOWPASS_CUTOFF_HZ))
    bandpass: BandpassFilterConfig = field(default_factory=BandpassFilterConfig)
    notch: NotchFilterConfig = field(default_factory=NotchFilterConfig)

    def any_enabled(self) -> bool:
        return any(
            (
                self.dc_removal,
                self.bandpass.enabled,
                self.highpass.enabled and not self.bandpass.enabled,
                self.lowpass.enabled and not self.bandpass.enabled,
                self.notch.enabled,
            )
        )


def default_filter_config() -> SignalFilterConfig:
    return SignalFilterConfig()


def remove_dc_offset(signal_data: np.ndarray) -> np.ndarray:
    centered = np.asarray(signal_data, dtype=float)
    return centered - np.mean(centered, axis=0, keepdims=True)


def apply_highpass(signal_data: np.ndarray, sampling_rate: float, cutoff_hz: float) -> np.ndarray:
    return _apply_butterworth_filter(signal_data, sampling_rate, cutoff_hz, btype="highpass")


def apply_lowpass(signal_data: np.ndarray, sampling_rate: float, cutoff_hz: float) -> np.ndarray:
    return _apply_butterworth_filter(signal_data, sampling_rate, cutoff_hz, btype="lowpass")


def apply_bandpass(signal_data: np.ndarray, sampling_rate: float, low_cutoff_hz: float, high_cutoff_hz: float) -> np.ndarray:
    filtered = np.asarray(signal_data, dtype=float).copy()
    nyquist = _validate_sampling_rate(sampling_rate)
    if nyquist is None or scipy_signal is None:
        return filtered
    low_normalized = _normalize_cutoff(low_cutoff_hz, nyquist, "Bandpass low cutoff")
    high_normalized = _normalize_cutoff(high_cutoff_hz, nyquist, "Bandpass high cutoff")
    if low_normalized is None or high_normalized is None or low_normalized >= high_normalized:
        warnings.warn("Bandpass cutoffs are invalid, skipping bandpass filtering.", RuntimeWarning, stacklevel=2)
        return filtered
    b, a = scipy_signal.butter(FILTER_ORDER, [low_normalized, high_normalized], btype="bandpass")
    return _apply_zero_phase_filter(filtered, b, a, "bandpass")


def apply_notch_filter(
    signal_data: np.ndarray,
    sampling_rate: float,
    mains_frequency_hz: float,
    quality_factor: float = DEFAULT_NOTCH_Q,
) -> np.ndarray:
    filtered = np.asarray(signal_data, dtype=float).copy()
    nyquist = _validate_sampling_rate(sampling_rate)
    if nyquist is None or scipy_signal is None:
        return filtered
    if quality_factor <= 0:
        warnings.warn("Notch Q factor must be positive, skipping notch filtering.", RuntimeWarning, stacklevel=2)
        return filtered
    if mains_frequency_hz not in SUPPORTED_MAINS_FREQUENCIES_HZ:
        warnings.warn(
            f"Unsupported mains frequency {mains_frequency_hz:.1f} Hz, skipping notch filtering.",
            RuntimeWarning,
            stacklevel=2,
        )
        return filtered
    if mains_frequency_hz >= nyquist:
        warnings.warn(
            f"Sampling rate is too low for notch filtering at {mains_frequency_hz:.1f} Hz, skipping it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return filtered
    normalized_frequency = mains_frequency_hz / nyquist
    b, a = scipy_signal.iirnotch(normalized_frequency, quality_factor)
    return _apply_zero_phase_filter(filtered, b, a, "notch")


def preprocess_signal(signal_data: np.ndarray, sampling_rate: float, config: SignalFilterConfig) -> np.ndarray:
    processed_signal = np.asarray(signal_data, dtype=float).copy()
    if not config.any_enabled():
        return processed_signal

    if config.dc_removal:
        processed_signal = remove_dc_offset(processed_signal)

    if config.bandpass.enabled:
        processed_signal = apply_bandpass(
            processed_signal,
            sampling_rate,
            config.bandpass.low,
            config.bandpass.high,
        )
    else:
        if config.highpass.enabled:
            processed_signal = apply_highpass(processed_signal, sampling_rate, config.highpass.cutoff)
        if config.lowpass.enabled:
            processed_signal = apply_lowpass(processed_signal, sampling_rate, config.lowpass.cutoff)

    if config.notch.enabled:
        processed_signal = apply_notch_filter(
            processed_signal,
            sampling_rate,
            config.notch.mains_frequency_hz,
            config.notch.quality_factor,
        )

    return processed_signal


def build_preview_signal(
    signal_data: np.ndarray,
    sampling_rate: float,
    *,
    config: SignalFilterConfig | None = None,
    **_kwargs: object,
) -> np.ndarray:
    return preprocess_signal(signal_data, sampling_rate, config or default_filter_config())


def _apply_butterworth_filter(signal_data: np.ndarray, sampling_rate: float, cutoff_hz: float, *, btype: str) -> np.ndarray:
    filtered = np.asarray(signal_data, dtype=float).copy()
    nyquist = _validate_sampling_rate(sampling_rate)
    if nyquist is None or scipy_signal is None:
        return filtered
    normalized_cutoff = _normalize_cutoff(cutoff_hz, nyquist, f"{btype.title()} cutoff")
    if normalized_cutoff is None:
        return filtered
    b, a = scipy_signal.butter(FILTER_ORDER, normalized_cutoff, btype=btype)
    return _apply_zero_phase_filter(filtered, b, a, btype)


def _validate_sampling_rate(sampling_rate: float) -> float | None:
    if scipy_signal is None:
        warnings.warn("SciPy is unavailable, skipping signal filtering.", RuntimeWarning, stacklevel=3)
        return None
    if sampling_rate <= 0:
        warnings.warn("Sampling rate is missing or invalid, skipping signal filtering.", RuntimeWarning, stacklevel=3)
        return None
    return sampling_rate / 2.0


def _normalize_cutoff(cutoff_hz: float, nyquist_hz: float, label: str) -> float | None:
    if cutoff_hz <= 0:
        warnings.warn(f"{label} must be positive, skipping that filter.", RuntimeWarning, stacklevel=3)
        return None
    if cutoff_hz >= nyquist_hz:
        warnings.warn(
            f"{label} exceeds Nyquist frequency, skipping that filter.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    return cutoff_hz / nyquist_hz


def _apply_zero_phase_filter(signal_data: np.ndarray, b: np.ndarray, a: np.ndarray, filter_name: str) -> np.ndarray:
    try:
        return scipy_signal.filtfilt(b, a, signal_data, axis=0)
    except ValueError:
        warnings.warn(
            f"Signal is too short for stable {filter_name} filtering, skipping that filter.",
            RuntimeWarning,
            stacklevel=3,
        )
        return signal_data
