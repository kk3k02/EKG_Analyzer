from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SelectionStats:
    """Technical summary for one selected signal segment in one lead."""

    start_time: float
    end_time: float
    duration: float
    minimum: float
    maximum: float
    mean: float
    std: float


def compute_selection_stats(time_axis: np.ndarray, signal_segment: np.ndarray) -> SelectionStats:
    """Compute technical min/max/mean/std for a single selected lead segment.

    The function is intentionally limited to one already-selected lead segment.
    It does not perform multi-lead aggregation, wave detection or clinical
    interpretation.
    """

    segment = np.asarray(signal_segment, dtype=float)
    if segment.size == 0:
        raise ValueError("Selection is empty.")
    collapsed = segment.reshape(-1)
    return SelectionStats(
        start_time=float(time_axis[0]),
        end_time=float(time_axis[-1]),
        duration=float(time_axis[-1] - time_axis[0]),
        minimum=float(np.min(collapsed)),
        maximum=float(np.max(collapsed)),
        mean=float(np.mean(collapsed)),
        std=float(np.std(collapsed)),
    )
