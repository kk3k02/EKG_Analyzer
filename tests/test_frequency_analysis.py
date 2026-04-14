from __future__ import annotations

import numpy as np
import pytest

from app.services import frequency_analysis


def test_compute_fft_psd_returns_positive_frequencies_only() -> None:
    sampling_rate = 250.0
    time_axis = np.arange(0.0, 8.0, 1.0 / sampling_rate)
    signal = np.sin(2.0 * np.pi * 15.0 * time_axis)

    result = frequency_analysis.compute_fft_psd(signal, sampling_rate, max_frequency_hz=60.0)

    peak_frequency = float(result.frequencies_hz[np.argmax(result.power)])
    assert result.message is None
    assert np.all(result.frequencies_hz >= 0.0)
    assert peak_frequency == pytest.approx(15.0, abs=0.5)


def test_compute_fft_psd_limits_frequency_range() -> None:
    sampling_rate = 360.0
    time_axis = np.arange(0.0, 5.0, 1.0 / sampling_rate)
    signal = np.sin(2.0 * np.pi * 30.0 * time_axis)

    result = frequency_analysis.compute_fft_psd(signal, sampling_rate, max_frequency_hz=20.0)

    assert result.message is None
    assert np.all(result.frequencies_hz <= 20.0)


def test_compute_stft_spectrogram_returns_consistent_shape() -> None:
    sampling_rate = 250.0
    time_axis = np.arange(0.0, 6.0, 1.0 / sampling_rate)
    signal = np.sin(2.0 * np.pi * 10.0 * time_axis) + 0.5 * np.sin(2.0 * np.pi * 20.0 * time_axis)

    result = frequency_analysis.compute_stft_spectrogram(
        signal,
        sampling_rate,
        nperseg=128,
        noverlap=96,
        nfft=256,
        max_frequency_hz=50.0,
    )

    assert result.message is None
    assert result.power_db.shape == (result.frequencies_hz.size, result.times_s.size)
    assert np.all(result.frequencies_hz >= 0.0)
    assert np.all(result.frequencies_hz <= 50.0)


def test_compute_stft_spectrogram_adjusts_invalid_parameters() -> None:
    sampling_rate = 250.0
    time_axis = np.arange(0.0, 2.0, 1.0 / sampling_rate)
    signal = np.sin(2.0 * np.pi * 8.0 * time_axis)

    result = frequency_analysis.compute_stft_spectrogram(
        signal,
        sampling_rate,
        nperseg=2048,
        noverlap=4096,
        nfft=64,
        max_frequency_hz=40.0,
    )

    assert result.message is None
    assert result.nperseg == signal.size
    assert result.noverlap == signal.size - 1
    assert result.nfft == signal.size


def test_compute_fft_psd_reports_short_signal() -> None:
    result = frequency_analysis.compute_fft_psd(np.ones(8), 250.0)

    assert result.message == "Wybrany fragment sygnalu jest zbyt krotki do analizy FFT."
