from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

FloatArray = NDArray[np.float64]


def load_processed_path(path: Path) -> FloatArray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["valid"] == "1"]
    points = np.asarray(
        [[float(row["smoothed_x"]), float(row["smoothed_y"])] for row in rows],
        dtype=np.float64,
    )
    if len(points) < 5:
        raise ValueError(f"Not enough valid trajectory points in {path}")
    return points


def smooth_and_resample(
    points: FloatArray,
    samples: int,
    window: int,
    polynomial_order: int,
) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    valid_window = min(window, len(values) - (1 - len(values) % 2))
    valid_window = max(polynomial_order + 2 + (polynomial_order % 2), valid_window)
    if valid_window % 2 == 0:
        valid_window -= 1
    smoothed = savgol_filter(values, valid_window, polynomial_order, axis=0, mode="interp")
    segment_lengths = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if distance[-1] <= 0:
        raise ValueError("Trajectory has zero length")
    sample_distance = np.linspace(0.0, distance[-1], samples)
    result = np.column_stack(
        [np.interp(sample_distance, distance, smoothed[:, dimension]) for dimension in range(2)]
    )
    result[0] = values[0]
    result[-1] = values[-1]
    return result
