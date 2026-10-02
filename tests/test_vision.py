from __future__ import annotations

import cv2
import numpy as np

from phone2panda.pilot_validation.geometry import homography_from_corners
from phone2panda.pilot_validation.vision import (
    ObjectCandidate,
    corner_observation,
    corner_reference,
    detect_object_candidates,
    track_object,
)


def _settings() -> dict[str, object]:
    return {
        "corner_gray_max": 160,
        "corner_area_min": 200,
        "corner_area_max": 5000,
        "corner_min_extent_px": 20,
        "red_hue_low": [0, 18],
        "red_hue_high": [170, 180],
        "red_saturation_min": 130,
        "red_value_min": 60,
        "red_area_min": 100,
        "red_area_max": 5000,
        "red_max_extent_px": 100,
        "red_min_solidity": 0.25,
    }


def _synthetic_frame() -> np.ndarray:
    image = np.full((400, 600, 3), 245, dtype=np.uint8)
    thickness = 12
    # Four connected L components, oriented like the physical pilot markers.
    cv2.line(image, (30, 30), (95, 30), (30, 30, 30), thickness)
    cv2.line(image, (30, 30), (30, 95), (30, 30, 30), thickness)
    cv2.line(image, (505, 30), (570, 30), (30, 30, 30), thickness)
    cv2.line(image, (570, 30), (570, 95), (30, 30, 30), thickness)
    cv2.line(image, (30, 305), (30, 370), (30, 30, 30), thickness)
    cv2.line(image, (30, 370), (95, 370), (30, 30, 30), thickness)
    cv2.line(image, (570, 305), (570, 370), (30, 30, 30), thickness)
    cv2.line(image, (505, 370), (570, 370), (30, 30, 30), thickness)
    cv2.rectangle(image, (260, 290), (315, 345), (0, 0, 220), -1)
    return image


def test_corner_and_red_marker_detection() -> None:
    frame = _synthetic_frame()
    references = corner_reference(frame, _settings())
    assert set(references) == {"tl", "tr", "bl", "br"}
    for reference in references.values():
        direct, confidence, distance = corner_observation(reference, reference, 10.0)
        assert direct
        assert confidence == 1.0
        assert distance == 0.0

    homography = homography_from_corners(
        {name: detection.point for name, detection in references.items()}
    )
    candidates = detect_object_candidates(frame, homography, _settings())
    assert len(candidates) == 1
    assert 0.4 < candidates[0].normalised[0] < 0.6
    assert 0.7 < candidates[0].normalised[1] < 1.0


def test_short_tracking_gap_is_interpolated() -> None:
    first = ObjectCandidate((10, 10), (0.5, 0.8), 1000, 0.8, 0.8)
    last = ObjectCandidate((12, 8), (0.5, 0.6), 1000, 0.8, 0.8)
    selected, points, interpolated = track_object(
        [[first], [], [last]],
        start_zone=(0.4, 0.7, 0.6, 0.9),
        max_step=0.25,
        max_gap=2,
    )
    assert selected[1] is None
    np.testing.assert_allclose(points[1], [0.5, 0.7])
    assert interpolated == [False, True, False]
