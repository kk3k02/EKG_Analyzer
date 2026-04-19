from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.services.validation import validate_ecg_record_data


@dataclass(slots=True)
class ECGRecord:
    """Canonical in-memory ECG container shared by all implemented loaders.

    The model is intentionally simple for Stage 1. `annotations` stays available
    as an extension point for future beat labels, R-peak markers and ML-related
    metadata, but the current iteration does not populate clinical annotations.
    """

    source_format: str
    file_path: str
    sampling_rate: float
    lead_names: list[str]
    signal: np.ndarray
    time_axis: np.ndarray
    units: str = "mV"
    metadata: dict[str, Any] = field(default_factory=dict)
    annotations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float)
        self.time_axis = np.asarray(self.time_axis, dtype=float)
        self.lead_names = list(self.lead_names)
        validate_ecg_record_data(self)

    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_leads(self) -> int:
        return int(self.signal.shape[1])

    @property
    def duration_seconds(self) -> float:
        if self.n_samples <= 1:
            return 0.0
        return float(self.time_axis[-1] - self.time_axis[0])

    @property
    def file_name(self) -> str:
        return Path(self.file_path).name

    def copy_with(
        self,
        *,
        signal: np.ndarray | None = None,
        time_axis: np.ndarray | None = None,
        sampling_rate: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ECGRecord":
        """Return a validated copy with selected fields replaced."""
        return ECGRecord(
            source_format=self.source_format,
            file_path=self.file_path,
            sampling_rate=self.sampling_rate if sampling_rate is None else sampling_rate,
            lead_names=self.lead_names.copy(),
            signal=self.signal.copy() if signal is None else signal,
            time_axis=self.time_axis.copy() if time_axis is None else time_axis,
            units=self.units,
            metadata=self.metadata.copy() if metadata is None else metadata,
            annotations=self.annotations.copy(),
        )

    def slice_samples(self, start_idx: int, end_idx: int, *, rebase_time_axis: bool = False) -> "ECGRecord":
        """Return a validated record containing only the requested sample interval."""
        start = max(0, min(int(start_idx), self.n_samples))
        end = max(start, min(int(end_idx), self.n_samples))
        if end <= start:
            raise ValueError("Selected ECG fragment is empty.")

        time_axis = self.time_axis[start:end].copy()
        metadata = self.metadata.copy()
        metadata["fragment_source_file"] = self.file_path
        metadata["fragment_sample_range"] = (start, end)
        metadata["fragment_time_range"] = (float(time_axis[0]), float(time_axis[-1]))

        if rebase_time_axis:
            time_axis = time_axis - float(time_axis[0])
            metadata["fragment_time_axis_rebased"] = True

        return ECGRecord(
            source_format=self.source_format,
            file_path=self.file_path,
            sampling_rate=self.sampling_rate,
            lead_names=self.lead_names.copy(),
            signal=self.signal[start:end].copy(),
            time_axis=time_axis,
            units=self.units,
            metadata=metadata,
            annotations=self._slice_annotations(start, end),
        )

    def _slice_annotations(self, start_idx: int, end_idx: int) -> list[dict[str, Any]]:
        sliced: list[dict[str, Any]] = []
        for annotation in self.annotations:
            sample_index = annotation.get("sample")
            if not isinstance(sample_index, (int, np.integer)):
                continue
            sample_value = int(sample_index)
            if start_idx <= sample_value < end_idx:
                copied = annotation.copy()
                copied["sample"] = sample_value - start_idx
                sliced.append(copied)
        return sliced
