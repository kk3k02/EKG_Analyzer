from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy import signal as scipy_signal
except ImportError:  # pragma: no cover - fallback path
    scipy_signal = None


MIN_SAMPLES_FOR_ANALYSIS = 16
DEFAULT_MAX_FREQUENCY_HZ = 40.0
DEFAULT_STFT_WINDOW = 256
DEFAULT_STFT_OVERLAP = 128


@dataclass(slots=True)
class PreparedSignalSegment:
    """Validated one-dimensional signal segment with matching time axis."""

    signal: np.ndarray
    time_axis: np.ndarray
    start_time: float
    end_time: float


@dataclass(slots=True)
class PSDResult:
    """Frequency-domain power spectrum for one signal segment."""

    frequencies_hz: np.ndarray
    power: np.ndarray
    y_label: str
    message: str | None = None


@dataclass(slots=True)
class SpectrogramResult:
    """STFT spectrogram data prepared for plotting."""

    frequencies_hz: np.ndarray
    times_s: np.ndarray
    power_db: np.ndarray
    nperseg: int
    noverlap: int
    nfft: int
    message: str | None = None


def prepare_signal_segment(
    signal: np.ndarray,
    time_axis: np.ndarray,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> PreparedSignalSegment:
    """Return a clipped signal slice for the requested time range."""
    segment = np.asarray(signal, dtype=float).reshape(-1)
    time_values = np.asarray(time_axis, dtype=float).reshape(-1)

    if segment.size == 0 or time_values.size == 0:
        raise ValueError("Brak danych sygnalu do analizy.")
    if segment.size != time_values.size:
        raise ValueError("Sygnal i os czasu maja rozna dlugosc.")

    finite_mask = np.isfinite(segment) & np.isfinite(time_values)
    if not np.any(finite_mask):
        raise ValueError("Wybrany fragment nie zawiera poprawnych probek.")

    segment = segment[finite_mask]
    time_values = time_values[finite_mask]

    clip_start = float(time_values[0] if start_time is None else start_time)
    clip_end = float(time_values[-1] if end_time is None else end_time)
    bounded_start = max(float(time_values[0]), min(clip_start, float(time_values[-1])))
    bounded_end = max(float(time_values[0]), min(clip_end, float(time_values[-1])))

    if bounded_end <= bounded_start:
        raise ValueError("Zakres czasu jest niepoprawny.")

    start_idx = int(np.searchsorted(time_values, bounded_start, side="left"))
    end_idx = int(np.searchsorted(time_values, bounded_end, side="right"))
    start_idx = max(0, min(start_idx, time_values.size - 1))
    end_idx = max(start_idx + 1, min(end_idx, time_values.size))

    sliced_signal = segment[start_idx:end_idx]
    sliced_time = time_values[start_idx:end_idx]
    if sliced_signal.size == 0:
        raise ValueError("Wybrany zakres czasu nie zawiera probek.")

    return PreparedSignalSegment(
        signal=sliced_signal,
        time_axis=sliced_time,
        start_time=float(sliced_time[0]),
        end_time=float(sliced_time[-1]),
    )


def compute_fft_psd(
    signal_segment: np.ndarray,
    sampling_rate: float,
    *,
    max_frequency_hz: float = DEFAULT_MAX_FREQUENCY_HZ,
) -> PSDResult:
    """Compute one-sided PSD using FFT with DC removal and Hann window."""
    if sampling_rate <= 0:
        return PSDResult(
            frequencies_hz=np.array([], dtype=float),
            power=np.array([], dtype=float),
            y_label="Gestosc mocy widmowej",
            message="Czestotliwosc probkowania jest niepoprawna albo nieznana.",
        )

    segment = np.asarray(signal_segment, dtype=float).reshape(-1)
    segment = segment[np.isfinite(segment)]
    if segment.size < MIN_SAMPLES_FOR_ANALYSIS:
        return PSDResult(
            frequencies_hz=np.array([], dtype=float),
            power=np.array([], dtype=float),
            y_label="Gestosc mocy widmowej",
            message="Wybrany fragment sygnalu jest zbyt krotki do analizy FFT.",
        )

    centered = segment - float(np.mean(segment))
    window = np.hanning(centered.size)
    windowed = centered * window

    spectrum = np.fft.rfft(windowed)
    frequencies_hz = np.fft.rfftfreq(windowed.size, d=1.0 / float(sampling_rate))
    power = (np.abs(spectrum) ** 2) / max(float(sampling_rate) * float(np.sum(window**2)), np.finfo(float).tiny)

    if power.size > 1:
        if windowed.size % 2 == 0 and power.size > 2:
            power[1:-1] *= 2.0
        elif windowed.size % 2 == 1:
            power[1:] *= 2.0

    frequency_limit = max(0.0, float(max_frequency_hz))
    mask = (frequencies_hz >= 0.0) & (frequencies_hz <= frequency_limit)
    filtered_frequencies = np.asarray(frequencies_hz[mask], dtype=float)
    filtered_power = np.asarray(power[mask], dtype=float)

    if filtered_frequencies.size == 0:
        return PSDResult(
            frequencies_hz=np.array([], dtype=float),
            power=np.array([], dtype=float),
            y_label="Gestosc mocy widmowej",
            message="Brak skladowych widmowych w wybranym zakresie czestotliwosci.",
        )

    return PSDResult(
        frequencies_hz=filtered_frequencies,
        power=filtered_power,
        y_label="Gestosc mocy widmowej",
    )


def compute_stft_spectrogram(
    signal_segment: np.ndarray,
    sampling_rate: float,
    *,
    segment_start_time: float = 0.0,
    nperseg: int = DEFAULT_STFT_WINDOW,
    noverlap: int = DEFAULT_STFT_OVERLAP,
    nfft: int | None = None,
    max_frequency_hz: float = DEFAULT_MAX_FREQUENCY_HZ,
) -> SpectrogramResult:
    """Compute a log-scaled STFT spectrogram for one segment."""
    if scipy_signal is None:
        return SpectrogramResult(
            frequencies_hz=np.array([], dtype=float),
            times_s=np.array([], dtype=float),
            power_db=np.empty((0, 0), dtype=float),
            nperseg=0,
            noverlap=0,
            nfft=0,
            message="SciPy jest niedostepne, wiec nie mozna obliczyc spektrogramu.",
        )

    if sampling_rate <= 0:
        return SpectrogramResult(
            frequencies_hz=np.array([], dtype=float),
            times_s=np.array([], dtype=float),
            power_db=np.empty((0, 0), dtype=float),
            nperseg=0,
            noverlap=0,
            nfft=0,
            message="Czestotliwosc probkowania jest niepoprawna albo nieznana.",
        )

    segment = np.asarray(signal_segment, dtype=float).reshape(-1)
    segment = segment[np.isfinite(segment)]
    if segment.size < MIN_SAMPLES_FOR_ANALYSIS:
        return SpectrogramResult(
            frequencies_hz=np.array([], dtype=float),
            times_s=np.array([], dtype=float),
            power_db=np.empty((0, 0), dtype=float),
            nperseg=0,
            noverlap=0,
            nfft=0,
            message="Wybrany fragment sygnalu jest zbyt krotki do analizy STFT.",
        )

    normalized_nperseg, normalized_noverlap, normalized_nfft = normalize_stft_parameters(
        signal_length=segment.size,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )

    frequencies_hz, times_s, stft_values = scipy_signal.stft(
        segment,
        fs=float(sampling_rate),
        window="hann",
        nperseg=normalized_nperseg,
        noverlap=normalized_noverlap,
        nfft=normalized_nfft,
        detrend="constant",
        boundary=None,
        padded=False,
    )
    power_db = 10.0 * np.log10(np.maximum(np.abs(stft_values) ** 2, np.finfo(float).tiny))

    frequency_limit = max(0.0, float(max_frequency_hz))
    mask = (frequencies_hz >= 0.0) & (frequencies_hz <= frequency_limit)
    frequencies_hz = np.asarray(frequencies_hz[mask], dtype=float)
    power_db = np.asarray(power_db[mask, :], dtype=float)

    if frequencies_hz.size == 0 or power_db.size == 0:
        return SpectrogramResult(
            frequencies_hz=np.array([], dtype=float),
            times_s=np.array([], dtype=float),
            power_db=np.empty((0, 0), dtype=float),
            nperseg=normalized_nperseg,
            noverlap=normalized_noverlap,
            nfft=normalized_nfft,
            message="Brak danych spektrogramu w wybranym zakresie czestotliwosci.",
        )

    return SpectrogramResult(
        frequencies_hz=frequencies_hz,
        times_s=np.asarray(times_s + float(segment_start_time), dtype=float),
        power_db=power_db,
        nperseg=normalized_nperseg,
        noverlap=normalized_noverlap,
        nfft=normalized_nfft,
    )


def normalize_stft_parameters(
    *,
    signal_length: int,
    nperseg: int,
    noverlap: int,
    nfft: int | None,
) -> tuple[int, int, int]:
    """Clamp STFT parameters to a safe configuration for the input length."""
    if signal_length < MIN_SAMPLES_FOR_ANALYSIS:
        raise ValueError("Signal is too short for STFT normalization.")

    normalized_nperseg = max(MIN_SAMPLES_FOR_ANALYSIS, int(nperseg))
    normalized_nperseg = min(normalized_nperseg, int(signal_length))

    normalized_noverlap = max(0, int(noverlap))
    normalized_noverlap = min(normalized_noverlap, normalized_nperseg - 1)

    if nfft is None:
        normalized_nfft = normalized_nperseg
    else:
        normalized_nfft = max(int(nfft), normalized_nperseg)

    return normalized_nperseg, normalized_noverlap, normalized_nfft
