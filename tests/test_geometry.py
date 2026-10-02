from __future__ import annotations

import numpy as np
import pytest

from phone2panda.pilot_validation.geometry import (
    classify_route,
    homography_from_corners,
    point_in_rect,
    signed_distance_to_rect,
    transform_points,
)


def test_homography_maps_all_four_marker_corners() -> None:
    corners = {
        "tl": (100.0, 80.0),
        "tr": (900.0, 60.0),
        "bl": (120.0, 700.0),
        "br": (880.0, 720.0),
    }
    homography = homography_from_corners(corners)
    transformed = transform_points(
        [corners["tl"], corners["tr"], corners["br"], corners["bl"]], homography
    )
    np.testing.assert_allclose(
        transformed,
        np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]]),
        atol=1e-6,
    )


def test_route_and_clearance_geometry() -> None:
    obstacle = (0.4, 0.35, 0.6, 0.55)
    left_path = np.asarray([[0.5, 0.8], [0.25, 0.6], [0.25, 0.45], [0.5, 0.2]])
    route, crossing_x, sample_count = classify_route(left_path, obstacle, 0.10)
    assert route == "left"
    assert crossing_x == 0.25
    assert sample_count == 2
    assert signed_distance_to_rect((0.25, 0.45), obstacle) == pytest.approx(0.15)
    assert signed_distance_to_rect((0.50, 0.45), obstacle) < 0


def test_zone_margin_checks_whole_footprint() -> None:
    zone = (0.4, 0.1, 0.6, 0.3)
    assert point_in_rect((0.5, 0.2), zone, margin=0.05)
    assert not point_in_rect((0.42, 0.2), zone, margin=0.05)
