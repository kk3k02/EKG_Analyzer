"""Czysta logika wyliczania widocznego okna podczas odtwarzania EKG.

Wydzielona z warstwy GUI, by była testowalna bez Qt/pyqtgraph.
"""

from __future__ import annotations

CURSOR_WINDOW_FRACTION = 0.7  # ile szerokości okna leży na lewo od kursora


def compute_scroll_window(
    playback_time: float,
    window_seconds: float,
    record_start: float,
    record_end: float,
    fraction: float = CURSOR_WINDOW_FRACTION,
) -> tuple[float, float, float]:
    """Wylicz widoczne okno dla płynnego przewijania (strip-chart).

    Kursor utrzymywany jest na stałym ułamku *fraction* szerokości okna, a widok
    przesuwa się ciągle wraz z kursorem (brak granic stron → brak skoków QRS).
    Na początku i końcu nagrania widok jest „przyklejony" do brzegu, więc to
    kursor dojeżdża do krawędzi — nadal płynnie.

    Zwraca ``(view_start, view_end, cursor)`` w absolutnych sekundach.
    """
    win = window_seconds if window_seconds > 0 else 10.0
    total = record_end - record_start
    cursor = min(max(record_start + max(playback_time, 0.0), record_start), record_end)
    if total <= win:
        view_start = record_start
    else:
        view_start = cursor - win * fraction
        view_start = min(max(view_start, record_start), record_end - win)
    return view_start, view_start + win, cursor
