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
