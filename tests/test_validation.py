from __future__ import annotations

import numpy as np
import pytest

from app.services.validation import build_time_axis, sanitize_signal


def test_sanitize_signal_replaces_nan_and_inf() -> None:
    signal = np.array([[0.0, np.nan], [np.inf, -np.inf]])
    sanitized = sanitize_signal(signal)
    assert np.isfinite(sanitized).all()


def test_build_time_axis_requires_positive_sampling_rate() -> None:
    with pytest.raises(ValueError):
        build_time_axis(100, 0.0)
