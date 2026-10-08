"""Shared, explicit outcome definitions for simulated manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectTaskOutcome:
    """Placement outcome that intentionally excludes robot-contact safety."""

    success: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SafeTaskOutcome:
    """Placement outcome including manipulation and robot-contact safety."""

    success: bool
    object_task_success: bool
    failure_reasons: tuple[str, ...]


def evaluate_object_task(
    *,
    placement_succeeded: bool,
    object_collision: bool,
    dropped: bool,
) -> ObjectTaskOutcome:
    """Classify the historical object-level task metric."""

    reasons: list[str] = []
    if not placement_succeeded:
        reasons.append("final_placement_outside_target")
    if object_collision:
        reasons.append("object_obstacle_collision")
    if dropped:
        reasons.append("object_drop_during_transport")
    return ObjectTaskOutcome(success=not reasons, failure_reasons=tuple(reasons))


def evaluate_safe_task(
    *,
    placement_succeeded: bool,
    object_collision: bool,
    unintended_robot_contact: bool,
    dropped: bool,
    grasp_retained: bool,
) -> SafeTaskOutcome:
    """Classify the calibrated safety gate from complete rollout signals."""

    object_outcome = evaluate_object_task(
        placement_succeeded=placement_succeeded,
        object_collision=object_collision,
        dropped=dropped,
    )
    reasons = list(object_outcome.failure_reasons)
    if unintended_robot_contact:
        reasons.append("unintended_robot_contact")
    if not grasp_retained:
        reasons.append("grasp_not_retained")
    return SafeTaskOutcome(
        success=not reasons,
        object_task_success=object_outcome.success,
        failure_reasons=tuple(reasons),
    )
