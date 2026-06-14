import numpy as np

from app.services.playback_window import CURSOR_WINDOW_FRACTION, compute_scroll_window


RECORD_START = 0.0
RECORD_END = 60.0
WINDOW = 10.0


def _view_start(playback_time, window=WINDOW):
    return compute_scroll_window(playback_time, window, RECORD_START, RECORD_END)[0]


def test_cursor_always_within_window():
    for t in np.linspace(0.0, RECORD_END, 200):
        view_start, view_end, cursor = compute_scroll_window(
            t, WINDOW, RECORD_START, RECORD_END
        )
        assert view_start <= cursor <= view_end + 1e-9
        assert abs((view_end - view_start) - WINDOW) < 1e-9


def test_view_start_is_monotonic_and_smooth():
    """Brak skoków: view_start rośnie monotonicznie, a delty są drobne
    (proporcjonalne do kroku czasu), nie skokowe o całe okno."""
    times = np.linspace(0.0, RECORD_END, 600)
    starts = np.array([_view_start(t) for t in times])
    diffs = np.diff(starts)
    assert np.all(diffs >= -1e-9)  # monotoniczne, nie cofa się
    step = times[1] - times[0]
    # Żadna delta nie przekracza kroku czasu (z marginesem) — brak przeskoku o okno
    assert np.max(diffs) <= step + 1e-6


def test_clamped_to_start_at_beginning():
    view_start, view_end, cursor = compute_scroll_window(
        0.0, WINDOW, RECORD_START, RECORD_END
    )
    assert view_start == RECORD_START
    assert cursor == RECORD_START


def test_clamped_to_end_at_finish():
    view_start, view_end, cursor = compute_scroll_window(
        RECORD_END, WINDOW, RECORD_START, RECORD_END
    )
    assert view_end == RECORD_END
    assert cursor == RECORD_END


def test_cursor_pinned_to_fraction_in_middle():
    cursor_time = 30.0
    view_start, view_end, cursor = compute_scroll_window(
        cursor_time, WINDOW, RECORD_START, RECORD_END
    )
    assert cursor == cursor_time
    # Kursor leży na ustalonym ułamku szerokości okna
    assert abs((cursor - view_start) - WINDOW * CURSOR_WINDOW_FRACTION) < 1e-9


def test_record_shorter_than_window_shows_full_window():
    view_start, view_end, cursor = compute_scroll_window(
        2.0, WINDOW, 0.0, 5.0
    )
    assert view_start == 0.0
    assert view_end == WINDOW


def test_zero_window_falls_back_to_default():
    view_start, view_end, _ = compute_scroll_window(
        5.0, 0.0, RECORD_START, RECORD_END
    )
    assert view_end - view_start == 10.0


def test_negative_playback_time_clamped():
    view_start, _, cursor = compute_scroll_window(
        -5.0, WINDOW, RECORD_START, RECORD_END
    )
    assert cursor == RECORD_START
    assert view_start == RECORD_START
