from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .geometry import Rect, transform_points


@dataclass(frozen=True)
class CornerDetection:
    point: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int
    l_shape_score: float


@dataclass(frozen=True)
class ObjectCandidate:
    pixel: tuple[float, float]
    normalised: tuple[float, float]
    area: float
    solidity: float
    score: float


CORNER_ORDER = ("tl", "tr", "bl", "br")
CORNER_ROIS = {
    "tl": (0.0, 0.0, 0.3, 0.3),
    "tr": (0.7, 0.0, 1.0, 0.3),
    "bl": (0.0, 0.7, 0.3, 1.0),
    "br": (0.7, 0.7, 1.0, 1.0),
}


def _l_shape_score(component: NDArray[np.bool_], corner: str) -> float:
    """Score occupancy of the two expected outer arms against the inner quadrant."""

    height, width = component.shape
    band_height = max(1, round(height * 0.3))
    band_width = max(1, round(width * 0.3))
    outer_top = corner in {"tl", "tr"}
    outer_left = corner in {"tl", "bl"}

    horizontal = component[:band_height] if outer_top else component[-band_height:]
    vertical = component[:, :band_width] if outer_left else component[:, -band_width:]
    inner_rows = slice(band_height, None) if outer_top else slice(None, -band_height)
    inner_columns = slice(band_width, None) if outer_left else slice(None, -band_width)
    interior = component[inner_rows, inner_columns]

    arm_fill = min(float(np.mean(horizontal)), float(np.mean(vertical)))
    interior_fill = float(np.mean(interior)) if interior.size else 1.0
    return arm_fill * (1.0 - interior_fill)


def temporal_median(samples: list[NDArray[np.uint8]]) -> NDArray[np.uint8]:
    if not samples:
        raise ValueError("Cannot construct a temporal median without frames")
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def detect_corner_markers(
    frame: NDArray[np.uint8], settings: dict[str, Any]
) -> dict[str, CornerDetection | None]:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = (gray < int(settings["corner_gray_max"])).astype(np.uint8)
    image_corners = {"tl": (0, 0), "tr": (width, 0), "bl": (0, height), "br": (width, height)}
    output: dict[str, CornerDetection | None] = {}
    for name in CORNER_ORDER:
        x0, y0, x1, y1 = CORNER_ROIS[name]
        left, top, right, bottom = (
            int(x0 * width),
            int(y0 * height),
            int(x1 * width),
            int(y1 * height),
        )
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary[top:bottom, left:right]
        )
        candidates: list[tuple[float, int, tuple[int, int, int, int, int, float]]] = []
        for index in range(1, len(stats)):
            x, y, box_width, box_height, area = (int(value) for value in stats[index])
            if not int(settings["corner_area_min"]) <= area <= int(settings["corner_area_max"]):
                continue
            minimum_extent = int(settings["corner_min_extent_px"])
            if box_width < minimum_extent or box_height < minimum_extent:
                continue
            component = labels[y : y + box_height, x : x + box_width] == index
            shape_score = _l_shape_score(component, name)
            if shape_score < float(settings.get("corner_min_l_score", 0.0)):
                continue
            center_x, center_y = centroids[index] + np.asarray([left, top])
            target_x, target_y = image_corners[name]
            distance = float(np.hypot(center_x - target_x, center_y - target_y))
            candidates.append(
                (
                    distance,
                    -area,
                    (x + left, y + top, box_width, box_height, area, shape_score),
                )
            )
        if not candidates:
            output[name] = None
            continue
        _, _, (x, y, box_width, box_height, area, shape_score) = min(candidates)
        points = {
            "tl": (x, y),
            "tr": (x + box_width - 1, y),
            "bl": (x, y + box_height - 1),
            "br": (x + box_width - 1, y + box_height - 1),
        }
        output[name] = CornerDetection(
            point=(float(points[name][0]), float(points[name][1])),
            bbox=(x, y, box_width, box_height),
            area=area,
            l_shape_score=shape_score,
        )
    return output


def corner_reference(
    median_frame: NDArray[np.uint8], settings: dict[str, Any]
) -> dict[str, CornerDetection]:
    detections = detect_corner_markers(median_frame, settings)
    missing = [name for name, detection in detections.items() if detection is None]
    if missing:
        raise ValueError(f"Unable to establish corner marker reference: {missing}")
    return {name: detection for name, detection in detections.items() if detection is not None}


def corner_observation(
    detection: CornerDetection | None,
    reference: CornerDetection,
    tolerance_px: float,
) -> tuple[bool, float, float]:
    if detection is None:
        return False, 0.0, float("nan")
    distance = float(np.linalg.norm(np.asarray(detection.point) - np.asarray(reference.point)))
    if distance > tolerance_px:
        return False, 0.0, distance
    distance_score = max(0.0, 1.0 - distance / tolerance_px)
    area_ratio = min(detection.area, reference.area) / max(detection.area, reference.area)
    return True, float(0.5 * distance_score + 0.5 * area_ratio), distance


def detect_object_candidates(
    frame: NDArray[np.uint8], homography: NDArray[np.float64], settings: dict[str, Any]
) -> list[ObjectCandidate]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = int(settings["red_saturation_min"])
    value = int(settings["red_value_min"])
    low_hue = settings["red_hue_low"]
    high_hue = settings["red_hue_high"]
    mask = cv2.inRange(hsv, (int(low_hue[0]), saturation, value), (int(low_hue[1]), 255, 255))
    mask |= cv2.inRange(hsv, (int(high_hue[0]), saturation, value), (int(high_hue[1]), 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    candidates: list[ObjectCandidate] = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = float(cv2.contourArea(contour))
        if not float(settings["red_area_min"]) <= area <= float(settings["red_area_max"]):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if max(width, height) > int(settings["red_max_extent_px"]):
            continue
        solidity = area / float(width * height)
        if solidity < float(settings["red_min_solidity"]):
            continue
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue
        pixel = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        normalised_array = transform_points([pixel], homography)[0]
        normalised = (float(normalised_array[0]), float(normalised_array[1]))
        if not (-0.05 <= normalised[0] <= 1.05 and -0.05 <= normalised[1] <= 1.05):
            continue
        score = solidity * min(1.0, area / 5000.0)
        candidates.append(
            ObjectCandidate(
                pixel=(float(pixel[0]), float(pixel[1])),
                normalised=normalised,
                area=area,
                solidity=float(solidity),
                score=float(score),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def track_object(
    candidates_by_frame: list[list[ObjectCandidate]],
    start_zone: Rect,
    max_step: float,
    max_gap: int,
) -> tuple[list[ObjectCandidate | None], NDArray[np.float64], list[bool]]:
    selected: list[ObjectCandidate | None] = []
    last: ObjectCandidate | None = None
    gap = 0
    for index, candidates in enumerate(candidates_by_frame):
        if index == 0 or last is None:
            in_start = [
                candidate
                for candidate in candidates
                if start_zone[0] - 0.05 <= candidate.normalised[0] <= start_zone[2] + 0.05
                and start_zone[1] - 0.05 <= candidate.normalised[1] <= start_zone[3] + 0.05
            ]
            choice = max(
                in_start or candidates, key=lambda candidate: candidate.score, default=None
            )
        else:
            allowed_distance = max_step * (gap + 1)
            nearby = [
                candidate
                for candidate in candidates
                if np.linalg.norm(np.asarray(candidate.normalised) - np.asarray(last.normalised))
                <= allowed_distance
            ]
            choice = min(
                nearby,
                key=lambda candidate: (
                    np.linalg.norm(np.asarray(candidate.normalised) - np.asarray(last.normalised)),
                    -candidate.score,
                ),
                default=None,
            )
        selected.append(choice)
        if choice is None:
            gap += 1
        else:
            last = choice
            gap = 0

    points = np.full((len(selected), 2), np.nan, dtype=np.float64)
    for index, candidate in enumerate(selected):
        if candidate is not None:
            points[index] = candidate.normalised
    interpolated = [False] * len(selected)
    valid_indices = np.flatnonzero(np.isfinite(points).all(axis=1))
    for left, right in zip(valid_indices[:-1], valid_indices[1:], strict=True):
        gap_length = int(right - left - 1)
        if 0 < gap_length <= max_gap:
            for offset in range(1, gap_length + 1):
                alpha = offset / (gap_length + 1)
                points[left + offset] = (1.0 - alpha) * points[left] + alpha * points[right]
                interpolated[left + offset] = True
    return selected, points, interpolated


def detect_obstacle(
    median_frame: NDArray[np.uint8], homography: NDArray[np.float64], settings: dict[str, Any]
) -> tuple[Rect, float]:
    hsv = cv2.cvtColor(median_frame, cv2.COLOR_BGR2HSV)
    hue = settings["blue_hue"]
    mask = cv2.inRange(
        hsv,
        (int(hue[0]), int(settings["blue_saturation_min"]), int(settings["blue_value_min"])),
        (int(hue[1]), 255, 255),
    )
    contours = [
        contour
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if cv2.contourArea(contour) > 1000
    ]
    if not contours:
        raise ValueError("Blue obstacle was not detected")
    contour = max(contours, key=cv2.contourArea)
    normalised = transform_points(
        [(float(point[0][0]), float(point[0][1])) for point in contour], homography
    )
    rect: Rect = (
        float(normalised[:, 0].min()),
        float(normalised[:, 1].min()),
        float(normalised[:, 0].max()),
        float(normalised[:, 1].max()),
    )
    x, y, width, height = cv2.boundingRect(contour)
    confidence = min(1.0, float(cv2.contourArea(contour)) / float(width * height))
    return rect, confidence
