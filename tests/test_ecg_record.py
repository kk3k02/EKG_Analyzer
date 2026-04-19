from __future__ import annotations

import numpy as np
import pytest

from app.models.ecg_record import ECGRecord


def test_ecg_record_promotes_1d_signal() -> None:
    record = ECGRecord(
        source_format="csv",
        file_path="sample.csv",
        sampling_rate=250.0,
        lead_names=["I"],
        signal=np.array([0.1, 0.2, 0.3]),
        time_axis=np.array([0.0, 0.004, 0.008]),
        units="mV",
        metadata={},
        annotations=[],
    )
    assert record.signal.shape == (3, 1)
    assert record.n_leads == 1


def test_ecg_record_rejects_invalid_time_axis_length() -> None:
    with pytest.raises(ValueError):
        ECGRecord(
            source_format="csv",
            file_path="sample.csv",
            sampling_rate=250.0,
            lead_names=["I"],
            signal=np.array([0.1, 0.2, 0.3]),
            time_axis=np.array([0.0, 0.004]),
            units="mV",
            metadata={},
            annotations=[],
        )


def test_ecg_record_can_slice_selected_fragment() -> None:
    record = ECGRecord(
        source_format="wfdb",
        file_path="sample.hea",
        sampling_rate=250.0,
        lead_names=["I", "II"],
        signal=np.array(
            [
                [0.1, 1.1],
                [0.2, 1.2],
                [0.3, 1.3],
                [0.4, 1.4],
            ]
        ),
        time_axis=np.array([0.0, 0.004, 0.008, 0.012]),
        units="mV",
        metadata={"patient_id": "abc"},
        annotations=[{"sample": 1, "symbol": "N"}, {"sample": 3, "symbol": "V"}],
    )

    fragment = record.slice_samples(1, 3)

    assert fragment.n_samples == 2
    assert np.allclose(fragment.signal, [[0.2, 1.2], [0.3, 1.3]])
    assert np.allclose(fragment.time_axis, [0.004, 0.008])
    assert fragment.annotations == [{"sample": 0, "symbol": "N"}]
    assert fragment.metadata["fragment_source_file"] == "sample.hea"
    assert fragment.metadata["fragment_sample_range"] == (1, 3)
