from __future__ import annotations

from pathlib import Path

import numpy as np

from app.io.base_loader import BaseECGLoader
from app.models.ecg_record import ECGRecord
from app.services.validation import build_time_axis
from app.utils.file_utils import normalize_path


class WFDBECGLoader(BaseECGLoader):
    def load(self, file_path: str) -> ECGRecord:
        import wfdb

        normalized_path = Path(normalize_path(file_path))
        record_name = normalized_path.with_suffix("")

        if normalized_path.suffix.lower() == ".dat":
            header_path = record_name.with_suffix(".hea")
            if not header_path.exists():
                raise FileNotFoundError("WFDB header file (.hea) is missing for the selected .dat file.")
        elif normalized_path.suffix.lower() == ".hea":
            data_path = record_name.with_suffix(".dat")
            if not data_path.exists():
                raise FileNotFoundError("WFDB data file (.dat) is missing for the selected record.")

        record = wfdb.rdrecord(str(record_name))
        signal = np.asarray(record.p_signal if record.p_signal is not None else record.d_signal, dtype=float)
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]

        sampling_rate = float(record.fs)
        time_axis = build_time_axis(signal.shape[0], sampling_rate)

        metadata = {
            "record_name": record.record_name,
            "base_time": str(record.base_time) if record.base_time else None,
            "base_date": str(record.base_date) if record.base_date else None,
            "comments": list(record.comments or []),
            "sig_len": int(record.sig_len),
        }

        return ECGRecord(
            source_format="wfdb",
            file_path=str(normalized_path),
            sampling_rate=sampling_rate,
            lead_names=list(record.sig_name),
            signal=signal,
            time_axis=time_axis,
            units="mV",
            metadata=metadata,
            annotations=[],
        )
