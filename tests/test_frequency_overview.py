from __future__ import annotations

import numpy as np
import pytest

from app.services import frequency_overview


def test_compute_frequency_overview_uses_welch_when_available() -> None:
    sampling_rate = 250.0
    time_axis = np.arange(0.0, 10.0, 1.0 / sampling_rate)
    signal = 0.8 * np.sin(2.0 * np.pi * 8.0 * time_axis) + 0.2

    result = frequency_overview.compute_frequency_overview(signal, sampling_rate)

    peak_frequency = float(result.frequencies_hz[np.argmax(result.values)])
    assert result.message is None
    assert result.method == "welch"
    assert result.y_label == "Power spectral density"
    assert peak_frequency == pytest.approx(8.0, abs=1.0)
    assert np.all(result.frequencies_hz <= frequency_overview.DEFAULT_MAX_FREQUENCY_HZ)


def test_compute_frequency_overview_falls_back_to_fft(monkeypatch) -> None:
    sampling_rate = 250.0
    time_axis = np.arange(0.0, 8.0, 1.0 / sampling_rate)
    signal = np.sin(2.0 * np.pi * 12.0 * time_axis)

    monkeypatch.setattr(frequency_overview, "welch", None)

    result = frequency_overview.compute_frequency_overview(signal, sampling_rate)

    peak_frequency = float(result.frequencies_hz[np.argmax(result.values)])
    assert result.message is None
    assert result.method == "fft"
    assert result.y_label == "Amplitude spectrum"
    assert peak_frequency == pytest.approx(12.0, abs=0.5)


def test_compute_frequency_overview_reports_short_segments() -> None:
    result = frequency_overview.compute_frequency_overview(np.ones(8), 250.0)

    assert result.method == "too_short"
    assert result.message == "Visible signal segment is too short for frequency analysis."


def test_compute_frequency_overview_rejects_invalid_sampling_rate() -> None:
    result = frequency_overview.compute_frequency_overview(np.ones(128), 0.0)

    assert result.method == "invalid"
    assert result.message == "Sampling rate is missing or invalid."
