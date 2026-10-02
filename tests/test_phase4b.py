from __future__ import annotations

import numpy as np

from phone2panda.evaluation.phase4b import (
    signed_rectangle_clearance,
    similarity_retarget,
    wilson_interval,
)


def test_similarity_retarget_matches_requested_endpoints() -> None:
    path = np.asarray([[0.0, 0.0], [0.4, 0.2], [1.0, 0.0]])
    start = np.asarray([0.1, -0.2])
    goal = np.asarray([0.2, 0.3])
    output = similarity_retarget(path, start, goal)
    np.testing.assert_allclose(output[0], start)
    np.testing.assert_allclose(output[-1], goal)
    assert np.linalg.norm(output[1] - np.linspace(start, goal, 3)[1]) > 0.01


def test_signed_clearance_accounts_for_object_footprint() -> None:
    points = np.asarray([[0.0, 0.0], [0.2, 0.0]])
    clearance = signed_rectangle_clearance(
        points,
        center=np.asarray([0.0, 0.0]),
        half_size=np.asarray([0.1, 0.05]),
        object_radius=0.02,
    )
    np.testing.assert_allclose(clearance, [-0.07, 0.08])


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(40, 50)
    assert low < 0.8 < high
    assert 0.65 < low < 0.7
    assert 0.88 < high < 0.92
