from __future__ import annotations

import numpy as np

from phone2panda.evaluation.phase4c import ContactAccumulator
from phone2panda.evaluation.phase4d import route_side_staging


def test_route_side_staging_uses_y_and_clamps_workspace() -> None:
    goal = np.asarray([0.22, 0.18])
    left, left_clamped = route_side_staging(goal, "left", 0.09, (0.0, 0.25), (-0.225, 0.225))
    right, right_clamped = route_side_staging(goal, "right", 0.09, (0.0, 0.25), (-0.225, 0.225))
    np.testing.assert_allclose(left, [0.22, 0.225])
    np.testing.assert_allclose(right, [0.22, 0.09])
    assert left_clamped
    assert not right_clamped


def test_contact_counts_are_attributed_to_execution_phase() -> None:
    contacts = ContactAccumulator()
    contacts.begin("seed")
    contacts.observe(1, "cube_g0", "gripper0_right_finger1_collision")
    contacts.observe(3, "robot0_link7_collision", "route_obstacle_geom")
    summary = contacts.phase_summary(
        "seed",
        [{"name": "transport", "steps": 2}, {"name": "placement", "steps": 3}],
    )
    assert summary["transport"]["intended_gripper_cube"]["contact_events"] == 1
    assert summary["placement"]["robot_obstacle"]["contact_events"] == 1
