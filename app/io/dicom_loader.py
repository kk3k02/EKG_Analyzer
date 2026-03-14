from __future__ import annotations

from app.io.base_loader import BaseECGLoader
from app.models.ecg_record import ECGRecord


class DICOMECGLoader(BaseECGLoader):
    def load(self, file_path: str) -> ECGRecord:
        raise NotImplementedError(
            "DICOM waveform loader is prepared as an extension point for Stage 2. "
            "WFDB, EDF and CSV/TXT are implemented in Stage 1."
        )
