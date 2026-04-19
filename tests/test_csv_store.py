from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from app.io.csv_store import CSVECGStore


def _make_temp_file(name: str, content: str) -> Path:
    root = Path(tempfile.mkdtemp(dir=Path.cwd()))
    file_path = root / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_csv_loader_reads_time_axis_from_first_column() -> None:
    file_path = _make_temp_file("sample.csv", "time,I,II\n0.0,0.1,0.2\n0.004,0.2,0.3\n0.008,0.3,0.4\n")

    record = CSVECGStore().load(str(file_path))

    assert record.n_samples == 3
    assert record.n_leads == 2
    assert np.isclose(record.sampling_rate, 250.0)
    assert record.lead_names == ["I", "II"]
    assert np.allclose(record.time_axis, [0.0, 0.004, 0.008])
    assert record.metadata["time_axis_source"] == "file"
    assert record.metadata["sampling_rate_editable"] is False


def test_csv_loader_generates_time_axis_without_explicit_time_column() -> None:
    file_path = _make_temp_file("sample.txt", "0.1;0.2\n0.2;0.3\n0.3;0.4\n")

    record = CSVECGStore().load(str(file_path))

    assert record.n_samples == 3
    assert record.n_leads == 2
    assert np.isclose(record.sampling_rate, 250.0)
    assert np.allclose(record.time_axis, [0.0, 0.004, 0.008])
    assert record.metadata["sampling_rate_defaulted"] is True
    assert record.metadata["sampling_rate_editable"] is True
