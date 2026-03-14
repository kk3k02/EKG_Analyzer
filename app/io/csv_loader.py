from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.io.base_loader import BaseECGLoader
from app.models.ecg_record import ECGRecord
from app.services.validation import build_time_axis
from app.utils.file_utils import normalize_path


DEFAULT_CSV_SAMPLING_RATE = 250.0


@dataclass(slots=True)
class CSVParseConfig:
    separator: str
    has_header: bool
    time_in_first_column: bool


class CSVECGLoader(BaseECGLoader):
    def load(self, file_path: str) -> ECGRecord:
        normalized_path = normalize_path(file_path)
        config = self._detect_config(normalized_path)
        dataframe = pd.read_csv(
            normalized_path,
            sep=config.separator,
            header=0 if config.has_header else None,
            engine="python",
        )
        dataframe = dataframe.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if dataframe.empty:
            raise ValueError("CSV/TXT file does not contain usable signal data.")

        dataframe = dataframe.apply(pd.to_numeric, errors="coerce")
        dataframe = dataframe.interpolate(limit_direction="both").bfill().ffill()
        if dataframe.isna().any().any():
            raise ValueError("CSV/TXT contains non-numeric values that could not be parsed.")

        signal_frame = dataframe.copy()
        metadata: dict[str, object] = {
            "separator": config.separator,
            "header_detected": config.has_header,
        }

        if config.time_in_first_column:
            time_axis = signal_frame.iloc[:, 0].to_numpy(dtype=float)
            signal_frame = signal_frame.iloc[:, 1:]
            sampling_rate = self._infer_sampling_rate(time_axis)
            metadata["time_axis_source"] = "file"
        else:
            sampling_rate = DEFAULT_CSV_SAMPLING_RATE
            time_axis = build_time_axis(len(signal_frame), sampling_rate)
            metadata["time_axis_source"] = "generated"
            metadata["sampling_rate_defaulted"] = True

        signal = signal_frame.to_numpy(dtype=float)
        lead_names = self._resolve_lead_names(signal_frame, has_header=config.has_header)

        return ECGRecord(
            source_format="csv",
            file_path=normalized_path,
            sampling_rate=sampling_rate,
            lead_names=lead_names,
            signal=signal,
            time_axis=time_axis,
            units="mV",
            metadata=metadata,
            annotations=[],
        )

    def _detect_config(self, file_path: str) -> CSVParseConfig:
        with Path(file_path).open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
            separator = dialect.delimiter
        except csv.Error:
            separator = ","

        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False

        preview = pd.read_csv(
            file_path,
            sep=separator,
            header=0 if has_header else None,
            nrows=20,
            engine="python",
        )
        first_column_name = str(preview.columns[0]) if len(preview.columns) else ""
        time_in_first_column = self._looks_like_time_column(
            preview.iloc[:, 0].to_numpy(),
            column_name=first_column_name,
            has_header=has_header,
        )
        return CSVParseConfig(separator=separator, has_header=has_header, time_in_first_column=time_in_first_column)

    def _looks_like_time_column(self, values: np.ndarray, *, column_name: str, has_header: bool) -> bool:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
        if numeric.size < 3:
            return False
        if has_header:
            normalized_name = column_name.strip().lower()
            if any(token in normalized_name for token in ("time", "czas", "sec", "timestamp", "t[s]")):
                return True
        diffs = np.diff(numeric)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size < max(2, int(0.7 * diffs.size)):
            return False
        median_step = float(np.median(positive_diffs))
        starts_near_zero = abs(float(numeric[0])) <= max(median_step * 0.1, 1e-6)
        return starts_near_zero and 0.0001 <= median_step <= 1.0

    def _infer_sampling_rate(self, time_axis: np.ndarray) -> float:
        diffs = np.diff(time_axis)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size == 0:
            raise ValueError("Could not infer sampling rate from CSV/TXT time axis.")
        median_step = float(np.median(positive_diffs))
        if median_step <= 0:
            raise ValueError("Invalid time axis step in CSV/TXT file.")
        return 1.0 / median_step

    def _resolve_lead_names(self, dataframe: pd.DataFrame, *, has_header: bool) -> list[str]:
        if has_header:
            return [str(column) for column in dataframe.columns]
        return [f"Lead {index + 1}" for index in range(dataframe.shape[1])]
