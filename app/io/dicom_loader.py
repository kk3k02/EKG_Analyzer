from __future__ import annotations

from app.io.base_loader import BaseECGLoader
from app.models.ecg_record import ECGRecord


class DICOMECGLoader(BaseECGLoader):
    """Partial DICOM waveform loader placeholder for a future iteration.

    Stage 1 keeps the loader entry point and factory wiring so the application
    can communicate the limitation explicitly, but full DICOM waveform parsing
    is not implemented yet.
    """

    def load(self, file_path: str) -> ECGRecord:
        # TODO: Implement waveform extraction from DICOM once Stage 2 broadens
        # the I/O scope and the expected DICOM subsets are defined.
        raise NotImplementedError(
            "Obsługa DICOM waveform w tej wersji aplikacji jest jeszcze niepełna. "
            "Ta ścieżka pozostaje stubem architektonicznym dla kolejnych etapów."
        )
