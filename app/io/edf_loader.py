from __future__ import annotations

from pathlib import Path

import numpy as np

from app.io.base_loader import BaseECGLoader
from app.models.ecg_record import ECGRecord
from app.services.validation import build_time_axis
from app.utils.file_utils import normalize_path


class EDFECGLoader(BaseECGLoader):
    def load(self, file_path: str) -> ECGRecord:
        import mne

        normalized_path = normalize_path(file_path)
        raw = mne.io.read_raw_edf(normalized_path, preload=True, verbose="ERROR")

        signal = raw.get_data().T
        sampling_rate = float(raw.info["sfreq"])
        time_axis = build_time_axis(signal.shape[0], sampling_rate)

        metadata = {
            "meas_date": str(raw.info.get("meas_date")),
            "patient_info": raw.info.get("subject_info"),
            "nchan": int(raw.info["nchan"]),
            "highpass": raw.info.get("highpass"),
            "lowpass": raw.info.get("lowpass"),
        }

        return ECGRecord(
            source_format="edf",
            file_path=str(Path(normalized_path)),
            sampling_rate=sampling_rate,
            lead_names=list(raw.ch_names),
            signal=np.asarray(signal, dtype=float),
            time_axis=time_axis,
            units="uV",
            metadata=metadata,
            annotations=[],
        )
