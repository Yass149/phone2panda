from __future__ import annotations

import pytest

from phone2panda.evaluation.safety import evaluate_object_task, evaluate_safe_task


@pytest.mark.parametrize(
    ("placed", "collision", "dropped", "expected"),
    [
        (True, False, False, True),
        (False, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
    ],
)
def test_object_task_success_truth_table(
    placed: bool, collision: bool, dropped: bool, expected: bool
) -> None:
    outcome = evaluate_object_task(
        placement_succeeded=placed,
        object_collision=collision,
        dropped=dropped,
    )

    assert outcome.success is expected


@pytest.mark.parametrize(
    ("placed", "collision", "unintended", "dropped", "grasped", "expected_reason"),
    [
        (False, False, False, False, True, "final_placement_outside_target"),
        (True, True, False, False, True, "object_obstacle_collision"),
        (True, False, True, False, True, "unintended_robot_contact"),
        (True, False, False, True, True, "object_drop_during_transport"),
        (True, False, False, False, False, "grasp_not_retained"),
    ],
)
def test_safe_task_rejects_each_failure_mode(
    placed: bool,
    collision: bool,
    unintended: bool,
    dropped: bool,
    grasped: bool,
    expected_reason: str,
) -> None:
    outcome = evaluate_safe_task(
        placement_succeeded=placed,
        object_collision=collision,
        unintended_robot_contact=unintended,
        dropped=dropped,
        grasp_retained=grasped,
    )

    assert not outcome.success
    assert expected_reason in outcome.failure_reasons


def test_safe_task_accepts_only_a_clean_retained_placement() -> None:
    outcome = evaluate_safe_task(
        placement_succeeded=True,
        object_collision=False,
        unintended_robot_contact=False,
        dropped=False,
        grasp_retained=True,
    )

    assert outcome.success
    assert outcome.object_task_success
    assert outcome.failure_reasons == ()
