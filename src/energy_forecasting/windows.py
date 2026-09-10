from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class RollingWindow:
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def rolling_windows(
    n_rows: int,
    train_size: int,
    test_size: int,
    step_size: int,
    max_windows: int | None = None,
) -> Iterator[RollingWindow]:
    """Yield expanding chronological train/test windows."""
    if train_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("train_size, test_size and step_size must be positive")

    train_end = train_size
    window_id = 0

    while train_end + test_size <= n_rows:
        yield RollingWindow(
            window_id=window_id,
            train_start=0,
            train_end=train_end,
            test_start=train_end,
            test_end=train_end + test_size,
        )
        window_id += 1
        if max_windows is not None and window_id >= max_windows:
            break
        train_end += step_size
