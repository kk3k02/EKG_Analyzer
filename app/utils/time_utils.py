from __future__ import annotations


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)} min {remaining:.2f} s"
