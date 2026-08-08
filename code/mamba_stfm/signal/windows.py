from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    values: np.ndarray


def window_bounds(total: int, width: int, stride: int) -> Iterator[tuple[int, int]]:
    if total < 0 or width <= 0 or stride <= 0:
        raise ValueError("invalid window dimensions")
    for start in range(0, max(0, total - width + 1), stride):
        yield start, start + width


def sliding_windows(data: np.ndarray, width: int, stride: int) -> Iterator[Window]:
    for start, stop in window_bounds(data.shape[-1], width, stride):
        yield Window(start, stop, data[..., start:stop])


def cue_epochs(data: np.ndarray, cues: np.ndarray, width: int, offset: int = 0) -> np.ndarray:
    epochs: list[np.ndarray] = []
    for cue in cues.astype(np.int64):
        start = int(cue) + offset
        stop = start + width
        if start >= 0 and stop <= data.shape[-1]:
            epochs.append(data[..., start:stop])
    if not epochs:
        return np.empty((0, *data.shape[:-1], width), dtype=data.dtype)
    return np.stack(epochs)


def resample_linear(data: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    if source_rate <= 0.0 or target_rate <= 0.0:
        raise ValueError("sample rates must be positive")
    target_length = int(round(data.shape[-1] * target_rate / source_rate))
    source_axis = np.linspace(0.0, 1.0, data.shape[-1], endpoint=True)
    target_axis = np.linspace(0.0, 1.0, target_length, endpoint=True)
    flat = data.reshape(-1, data.shape[-1])
    result = np.stack([np.interp(target_axis, source_axis, row) for row in flat])
    return result.reshape(*data.shape[:-1], target_length).astype(np.float32)
