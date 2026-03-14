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
