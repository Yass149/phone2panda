from __future__ import annotations

from phone2panda.evaluation.phase4c import classify_contact_pair


def test_contact_taxonomy() -> None:
    assert (
        classify_contact_pair("cube_g0", "gripper0_right_finger1_pad_collision")
        == "intended_gripper_cube"
    )
    assert (
        classify_contact_pair("robot0_link7_collision", "route_obstacle_geom")
        == "robot_obstacle"
    )
    assert classify_contact_pair("cube_g0", "route_obstacle_geom") == "object_obstacle"
    assert classify_contact_pair("robot0_link7_collision", "table_collision") == "robot_table"
    assert (
        classify_contact_pair("robot0_link6_collision", "gripper0_right_hand_collision")
        == "self_collision"
    )
    assert classify_contact_pair("cube_g0", "table_collision") == "other"
