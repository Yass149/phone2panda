from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

Rect = tuple[float, float, float, float]


def homography_from_corners(corners: dict[str, tuple[float, float]]) -> NDArray[np.float64]:
    source = np.asarray(
        [corners["tl"], corners["tr"], corners["br"], corners["bl"]], dtype=np.float32
    )
    destination = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, destination).astype(np.float64)


def transform_points(
    points: Iterable[tuple[float, float]], homography: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = np.asarray(list(points), dtype=np.float32)
    if not len(values):
        return np.empty((0, 2), dtype=np.float64)
    transformed = cv2.perspectiveTransform(values.reshape(-1, 1, 2), homography)
    return transformed.reshape(-1, 2).astype(np.float64)


def point_in_rect(point: tuple[float, float], rect: Rect, margin: float = 0.0) -> bool:
    x, y = point
    x_min, y_min, x_max, y_max = rect
    return x_min + margin <= x <= x_max - margin and y_min + margin <= y <= y_max - margin


def signed_distance_to_rect(point: tuple[float, float], rect: Rect) -> float:
    x, y = point
    x_min, y_min, x_max, y_max = rect
    dx = max(x_min - x, 0.0, x - x_max)
    dy = max(y_min - y, 0.0, y - y_max)
    if dx > 0.0 or dy > 0.0:
        return math.hypot(dx, dy)
    return -min(x - x_min, x_max - x, y - y_min, y_max - y)


def classify_route(
    points: NDArray[np.float64], obstacle: Rect, vertical_margin: float
) -> tuple[str, float, int]:
    valid = np.isfinite(points).all(axis=1)
    x_min, y_min, x_max, y_max = obstacle
    crossing = points[
        valid
        & (points[:, 1] >= y_min - vertical_margin)
        & (points[:, 1] <= y_max + vertical_margin)
    ]
    if not len(crossing):
        return "unknown", float("nan"), 0
    crossing_x = float(np.median(crossing[:, 0]))
    midpoint = (x_min + x_max) / 2.0
    route = "left" if crossing_x < midpoint else "right"
    return route, crossing_x, len(crossing)


def path_length(points: NDArray[np.float64]) -> float:
    valid = points[np.isfinite(points).all(axis=1)]
    if len(valid) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(valid, axis=0), axis=1).sum())
