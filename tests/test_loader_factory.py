from __future__ import annotations

import pytest

from app.io.csv_loader import CSVECGLoader
from app.io.dicom_loader import DICOMECGLoader
from app.io.edf_loader import EDFECGLoader
from app.io.loader_factory import LoaderFactory
from app.io.wfdb_loader import WFDBECGLoader


def test_loader_factory_resolves_known_formats() -> None:
    assert isinstance(LoaderFactory.create_loader("record.csv"), CSVECGLoader)
    assert isinstance(LoaderFactory.create_loader("record.edf"), EDFECGLoader)
    assert isinstance(LoaderFactory.create_loader("record.hea"), WFDBECGLoader)
    assert isinstance(LoaderFactory.create_loader("record.dcm"), DICOMECGLoader)


def test_loader_factory_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        LoaderFactory.create_loader("record.xyz")


def test_dicom_loader_reports_partial_support() -> None:
    with pytest.raises(NotImplementedError, match="DICOM waveform"):
        DICOMECGLoader().load("record.dcm")
