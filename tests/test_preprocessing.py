from __future__ import annotations

import numpy as np
import pytest

from app.services.preprocessing import (
    SignalFilterConfig,
    apply_notch_filter,
    default_filter_config,
    preprocess_signal,
)


def test_default_filter_config_starts_disabled() -> None:
    config = default_filter_config()
    assert config.any_enabled() is False


def test_preprocess_signal_returns_copy_when_filters_are_disabled() -> None:
    signal = np.random.default_rng(42).normal(size=(1000, 2))
    config = default_filter_config()

    processed = preprocess_signal(signal, 250.0, config)

    np.testing.assert_allclose(processed, signal)
    assert processed is not signal


def test_bandpass_takes_precedence_over_separate_highpass_and_lowpass() -> None:
    sampling_rate = 250.0
    time_axis = np.arange(0.0, 4.0, 1.0 / sampling_rate)
    signal = (
        np.sin(2.0 * np.pi * 0.2 * time_axis)
        + np.sin(2.0 * np.pi * 5.0 * time_axis)
        + np.sin(2.0 * np.pi * 80.0 * time_axis)
    ).reshape(-1, 1)
    config = SignalFilterConfig(
        bandpass=type(default_filter_config().bandpass)(enabled=True, low=0.5, high=40.0),
        highpass=type(default_filter_config().highpass)(enabled=True, cutoff=1.0),
        lowpass=type(default_filter_config().lowpass)(enabled=True, cutoff=30.0),
    )

    processed = preprocess_signal(signal, sampling_rate, config)

    assert processed.shape == signal.shape
    assert np.isfinite(processed).all()


def test_notch_filter_supports_50_and_60_hz() -> None:
    sampling_rate = 500.0
    time_axis = np.arange(0.0, 2.0, 1.0 / sampling_rate)
    signal = np.sin(2.0 * np.pi * 8.0 * time_axis).reshape(-1, 1)

    filtered_50 = apply_notch_filter(signal, sampling_rate, 50.0, 30.0)
    filtered_60 = apply_notch_filter(signal, sampling_rate, 60.0, 30.0)

    assert filtered_50.shape == signal.shape
    assert filtered_60.shape == signal.shape
    assert np.isfinite(filtered_50).all()
    assert np.isfinite(filtered_60).all()


def test_preprocess_signal_warns_and_skips_invalid_sampling_rate() -> None:
    signal = np.random.default_rng(0).normal(size=(128, 1))
    config = default_filter_config()
    config.highpass.enabled = True

    with pytest.warns(RuntimeWarning, match="Sampling rate is missing or invalid"):
        processed = preprocess_signal(signal, 0.0, config)

    np.testing.assert_allclose(processed, signal)


def test_preprocess_signal_warns_and_skips_invalid_cutoff() -> None:
    signal = np.random.default_rng(1).normal(size=(512, 1))
    config = default_filter_config()
    config.lowpass.enabled = True
    config.lowpass.cutoff = 1000.0

    with pytest.warns(RuntimeWarning, match="exceeds Nyquist frequency"):
        processed = preprocess_signal(signal, 250.0, config)

    np.testing.assert_allclose(processed, signal)
