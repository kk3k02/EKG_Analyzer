from __future__ import annotations

from pathlib import Path


CSV_EXTENSIONS = {".csv", ".txt"}
EDF_EXTENSIONS = {".edf"}
DICOM_EXTENSIONS = {".dcm"}
WFDB_HEADER_EXTENSIONS = {".hea"}
WFDB_DATA_EXTENSIONS = {".dat", ".atr", ".qrs"}


def normalize_path(file_path: str) -> str:
    return str(Path(file_path).expanduser().resolve())


def file_extension(file_path: str) -> str:
    return Path(file_path).suffix.lower()
