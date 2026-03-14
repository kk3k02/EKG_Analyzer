from __future__ import annotations

import numpy as np

from app.services.selection_stats import compute_selection_stats


def test_selection_stats_describe_one_selected_lead_segment() -> None:
    time_axis = np.array([0.0, 0.004, 0.008, 0.012])
    lead_segment = np.array([0.1, 0.2, -0.1, 0.0])

    stats = compute_selection_stats(time_axis, lead_segment)

    assert stats.start_time == 0.0
    assert stats.end_time == 0.012
    assert np.isclose(stats.duration, 0.012)
    assert np.isclose(stats.minimum, -0.1)
    assert np.isclose(stats.maximum, 0.2)
    assert np.isclose(stats.mean, 0.05)
