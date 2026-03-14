from __future__ import annotations

import numpy as np

from app.services.preprocessing import build_preview_signal


def test_preview_signal_preserves_shape() -> None:
    signal = np.random.default_rng(42).normal(size=(1000, 2))
    preview = build_preview_signal(signal, 250.0, remove_baseline=True, apply_lowpass=True)
    assert preview.shape == signal.shape
    assert np.isfinite(preview).all()
