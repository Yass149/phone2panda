from __future__ import annotations

from pathlib import Path

from phone2panda.evaluation.phase4e import (
    _diagnostic_checks,
    derive_transport_height,
    load_phase4e_config,
)


def test_calibrated_transport_height_lifts_cube_but_keeps_vertical_overlap() -> None:
    config = load_phase4e_config(Path("configs/phase4e.yaml"))
    derived = derive_transport_height(config)
    assert derived["cube_bottom_z_m"] > derived["table_z_m"]
    assert derived["cube_bottom_z_m"] < derived["obstacle_top_z_m"]
    assert derived["cube_top_z_m"] > derived["obstacle_top_z_m"]
    assert abs(derived["eef_transport_z_m"] - 0.847715) < 1e-12


def test_expected_straight_contact_does_not_block_clean_dmp_diagnostic() -> None:
    straight = {
        "object_collision_steps": 1,
        "contacts": {"robot_obstacle": {"contact_events": 1}},
    }
    dmp = {
        "target_placed": True,
        "object_collision_steps": 0,
        "object_remained_grasped": True,
        "unintended_contact": False,
    }

    checks, gate_keys = _diagnostic_checks(straight, dmp)

    assert checks["straight_gripper_contact_recorded"]
    assert "straight_gripper_contact_recorded" not in gate_keys
    assert all(checks[key] for key in gate_keys)
