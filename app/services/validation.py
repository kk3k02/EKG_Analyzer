from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.models.ecg_record import ECGRecord


def ensure_2d_signal(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=float)
    if array.ndim == 1:
        return array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError("Signal must be 1D or 2D.")
    return array


def sanitize_signal(signal: np.ndarray) -> np.ndarray:
    array = ensure_2d_signal(signal)
    if array.size == 0:
        raise ValueError("Signal is empty.")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def build_time_axis(n_samples: int, sampling_rate: float) -> np.ndarray:
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    return np.arange(n_samples, dtype=float) / float(sampling_rate)


def validate_ecg_record_data(record: ECGRecord) -> None:
    record.signal = sanitize_signal(record.signal)

    if record.sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    if record.signal.shape[0] == 0:
        raise ValueError("Signal is empty.")
    if record.time_axis.ndim != 1:
        raise ValueError("Time axis must be 1D.")
    if record.time_axis.shape[0] != record.signal.shape[0]:
        raise ValueError("Time axis length must match the number of samples.")
    if not np.isfinite(record.time_axis).all():
        raise ValueError("Time axis contains invalid values.")
    if len(record.lead_names) != record.signal.shape[1]:
        raise ValueError("Lead names length must match the number of leads.")
    if not record.lead_names:
        record.lead_names = [f"Lead {index + 1}" for index in range(record.signal.shape[1])]
