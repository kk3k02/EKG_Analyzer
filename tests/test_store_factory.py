from __future__ import annotations

import pytest

from app.io.csv_store import CSVECGStore
from app.io.dicom_store import DICOMECGStore
from app.io.edf_store import EDFECGStore
from app.io.store_factory import StoreFactory
from app.io.wfdb_store import WFDBECGStore


def test_store_factory_resolves_known_formats() -> None:
    assert isinstance(StoreFactory.create_loader("record.csv"), CSVECGStore)
    assert isinstance(StoreFactory.create_loader("record.edf"), EDFECGStore)
    assert isinstance(StoreFactory.create_loader("record.hea"), WFDBECGStore)
    assert isinstance(StoreFactory.create_loader("record.dcm"), DICOMECGStore)


def test_store_factory_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        StoreFactory.create_loader("record.xyz")
