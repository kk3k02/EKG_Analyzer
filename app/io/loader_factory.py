from __future__ import annotations

from pathlib import Path

from app.io.base_loader import BaseECGLoader
from app.io.csv_loader import CSVECGLoader
from app.io.dicom_loader import DICOMECGLoader
from app.io.edf_loader import EDFECGLoader
from app.io.wfdb_loader import WFDBECGLoader
from app.utils.file_utils import CSV_EXTENSIONS, DICOM_EXTENSIONS, EDF_EXTENSIONS, WFDB_DATA_EXTENSIONS, WFDB_HEADER_EXTENSIONS


class LoaderFactory:
    @staticmethod
    def create_loader(file_path: str) -> BaseECGLoader:
        extension = Path(file_path).suffix.lower()
        if extension in CSV_EXTENSIONS:
            return CSVECGLoader()
        if extension in EDF_EXTENSIONS:
            return EDFECGLoader()
        if extension in WFDB_HEADER_EXTENSIONS | WFDB_DATA_EXTENSIONS:
            return WFDBECGLoader()
        if extension in DICOM_EXTENSIONS:
            return DICOMECGLoader()
        raise ValueError(f"Unsupported file format: {extension or 'unknown'}")
