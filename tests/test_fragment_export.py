from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from app.io.csv_store import CSVECGStore
from app.models.ecg_record import ECGRecord


def test_exported_fragment_round_trips_through_csv_loader() -> None:
    root = Path(tempfile.mkdtemp(dir=Path.cwd()))
    file_path = root / "fragment.csv"
    record = ECGRecord(
        source_format="wfdb",
        file_path="sample.hea",
        sampling_rate=250.0,
        lead_names=["I", "II"],
        signal=np.array([[0.2, 1.2], [0.3, 1.3], [0.4, 1.4]]),
        time_axis=np.array([1.0, 1.004, 1.008]),
        units="mV",
        metadata={},
        annotations=[],
    )

    saved_path = CSVECGStore().save(record, str(file_path))
    loaded = CSVECGStore().load(saved_path)

    assert loaded.file_path == str(file_path.resolve())
    assert loaded.lead_names == ["I", "II"]
    assert np.allclose(loaded.signal, record.signal)
    assert np.allclose(loaded.time_axis, record.time_axis)
    assert np.isclose(loaded.sampling_rate, record.sampling_rate)
