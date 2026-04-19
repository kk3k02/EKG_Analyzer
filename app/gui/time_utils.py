from __future__ import annotations


def format_playback_clock(total_seconds: float) -> str:
    whole_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
