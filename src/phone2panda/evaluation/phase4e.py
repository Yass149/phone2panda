from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from phone2panda.evaluation.phase4b import (
    METHODS,
    Demonstration,
    Phase4BConfig,
    _comparison_markdown,
    _make_environment,
    _plot_comparison,
    _write_json,
    _write_jsonl,
    _write_video,
    build_library,
    execute_rollout,
    load_phase4b_config,
    make_scenario,
    select_method_path,
    wilson_interval,
)
from phone2panda.evaluation.phase4c import (
    UNINTENDED_CATEGORIES,
    ContactAccumulator,
)
from phone2panda.evaluation.phase4d import aggregate_phase_contacts
from phone2panda.evaluation.safety import evaluate_safe_task
from phone2panda.sim.phase4a import Phase4AConfig

FloatArray = NDArray[np.float64]
METHOD = "dmp_route_confidence"


@dataclass(frozen=True)
class Phase4EConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    phase4b: Phase4BConfig

    def project_path(self, key: str) -> Path:
        return self.root / str(self.raw[key])

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def load_phase4e_config(path: Path) -> Phase4EConfig:
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = resolved.parent.parent
    phase4b = load_phase4b_config(root / str(raw["phase4b_config"]))
    calibration = raw["calibration"]
    if float(calibration["obstacle_height_m"]) != 0.04:
        raise ValueError("Phase 4E requires a calibrated 0.04 m obstacle")
    if len(raw["preflight"]["seeds"]) != 5:
        raise ValueError("Phase 4E preflight requires exactly five seeds")
    if int(raw["evaluation"]["episode_count"]) != 50:
        raise ValueError("Phase 4E evaluation requires exactly 50 episodes")
    return Phase4EConfig(path=resolved, root=root, raw=raw, phase4b=phase4b)


def derive_transport_height(config: Phase4EConfig) -> dict[str, float]:
    workspace = config.phase4b.phase4a.raw["workspace"]
    calibration = config.raw["calibration"]
    table_z = float(workspace["table_z"])
    half_height = float(calibration["cube_half_height_max_m"])
    clearance = float(calibration["desired_cube_bottom_clearance_m"])
    grasp_offset = float(calibration["measured_grasp_offset_m"])
    obstacle_height = float(calibration["obstacle_height_m"])
    cube_center_z = table_z + half_height + clearance
    eef_transport_z = cube_center_z + grasp_offset
    cube_bottom_z = cube_center_z - half_height
    cube_top_z = cube_center_z + half_height
    obstacle_top_z = table_z + obstacle_height
    if not table_z < cube_bottom_z < obstacle_top_z < cube_top_z:
        raise ValueError(
            "Derived transport height does not lift the cube while blocking straight travel"
        )
    return {
        "table_z_m": table_z,
        "obstacle_top_z_m": obstacle_top_z,
        "cube_center_z_m": cube_center_z,
        "cube_bottom_z_m": cube_bottom_z,
        "cube_top_z_m": cube_top_z,
        "eef_transport_z_m": eef_transport_z,
        "vertical_overlap_m": obstacle_top_z - cube_bottom_z,
    }


def _execution_config(config: Phase4EConfig) -> tuple[Phase4BConfig, dict[str, float]]:
    derived = derive_transport_height(config)
    phase4b_raw = copy.deepcopy(config.phase4b.raw)
    phase4a_raw = copy.deepcopy(config.phase4b.phase4a.raw)
    phase4b_raw["scenario"]["obstacle_height_m"] = float(
        config.raw["calibration"]["obstacle_height_m"]
    )
    phase4a_raw["workspace"]["transport_z"] = derived["eef_transport_z_m"]
    phase4a_raw["workspace"]["approach_z"] = derived["eef_transport_z_m"]
    execution = Phase4BConfig(
        path=config.phase4b.path,
        root=config.phase4b.root,
        raw=phase4b_raw,
        phase4a=Phase4AConfig(root=config.phase4b.phase4a.root, raw=phase4a_raw),
    )
    return execution, derived


def _cube_dimensions(env: Any) -> FloatArray:
    geom_id = env.sim.model.geom_name2id(env.cube.contact_geoms[0])
    return 2.0 * np.asarray(env.sim.model.geom_size[geom_id, :3], dtype=np.float64)


def _run_rollout(
    config: Phase4EConfig,
    execution_config: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
    method: str,
    seed: int,
    index: int,
    render: bool = False,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]]]:
    np.random.seed(seed)
    scenario = make_scenario(execution_config, demos, geometry, seed, index)
    path, selection = select_method_path(execution_config, demos, scenario, method)
    contacts = ContactAccumulator()
    contacts.begin(f"{method}:{seed}")
    env = _make_environment(execution_config, scenario, render=render)
    cube_dimensions = _cube_dimensions(env)
    try:
        record, frames = execute_rollout(
            execution_config,
            env,
            scenario,
            method,
            path,
            selection,
            capture_video=render,
            contact_callback=contacts.observe,
            monitor_transport_height_drop=False,
        )
    finally:
        env.close()
    rollout_id = f"{method}:{seed}"
    contact_summary = contacts.rollout_summary(rollout_id)
    record["contacts"] = contact_summary
    record["phase_contacts"] = contacts.phase_summary(rollout_id, record["phases"])
    record["cube_dimensions_m"] = cube_dimensions.tolist()
    record["transport_cube_bottom_clearance_m"] = float(
        record["transport_cube_center_z_median_m"]
        - cube_dimensions[2] / 2.0
        - float(execution_config.phase4a.raw["workspace"]["table_z"])
    )
    record["object_remained_grasped"] = bool(
        not record["drop"]
        and record["maximum_transport_grasp_distance_m"]
        <= float(config.raw["calibration"]["maximum_grasp_distance_m"])
    )
    record["unintended_contact"] = any(
        contact_summary[category]["contact_events"] > 0
        for category in UNINTENDED_CATEGORIES
    )
    record["placement_success"] = bool(record["target_placed"])
    outcome = evaluate_safe_task(
        placement_succeeded=record["placement_success"],
        object_collision=record["object_collision_steps"] > 0,
        unintended_robot_contact=record["unintended_contact"],
        dropped=record["drop"],
        grasp_retained=record["object_remained_grasped"],
    )
    record["object_task_success"] = outcome.object_task_success
    record["calibrated_safety_success"] = outcome.success
    record["safety_failure_reasons"] = list(outcome.failure_reasons)
    return record, frames


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _physical_dimensions(config: Phase4EConfig, derived: dict[str, float]) -> dict[str, Any]:
    calibration = config.raw["calibration"]
    return {
        "physical_obstacle_height_m": float(calibration["physical_obstacle_height_m"]),
        "simulated_obstacle_height_m": float(calibration["obstacle_height_m"]),
        "simulated_cube_full_size_range_m": [0.040, 0.044],
        "configured_cube_half_height_max_m": float(calibration["cube_half_height_max_m"]),
        "measured_grasp_offset_m": float(calibration["measured_grasp_offset_m"]),
        "desired_cube_bottom_clearance_m": float(
            calibration["desired_cube_bottom_clearance_m"]
        ),
        **derived,
    }


def _diagnostic_checks(
    straight: dict[str, Any], dmp: dict[str, Any]
) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "straight_expected_obstacle_failure": bool(
            straight["object_collision_steps"] > 0
        ),
        "straight_gripper_contact_recorded": bool(
            straight["contacts"]["robot_obstacle"]["contact_events"] > 0
        ),
        "dmp_successful_placement": bool(dmp["target_placed"]),
        "dmp_no_object_collision": bool(dmp["object_collision_steps"] == 0),
        "dmp_object_remained_grasped": bool(dmp["object_remained_grasped"]),
        "dmp_no_unintended_robot_contact": not bool(dmp["unintended_contact"]),
    }
    gate_keys = [
        "straight_expected_obstacle_failure",
        "dmp_successful_placement",
        "dmp_no_object_collision",
        "dmp_object_remained_grasped",
        "dmp_no_unintended_robot_contact",
    ]
    return checks, gate_keys


def run_diagnostics(config: Phase4EConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    output.mkdir(parents=True, exist_ok=True)
    execution_config, derived = _execution_config(config)
    demos, geometry = build_library(execution_config)
    seed = int(config.raw["diagnostic"]["seed"])
    preflight_seeds = [int(value) for value in config.raw["preflight"]["seeds"]]
    index = preflight_seeds.index(seed)
    straight, _ = _run_rollout(
        config, execution_config, demos, geometry, "straight", seed, index
    )
    dmp, _ = _run_rollout(
        config, execution_config, demos, geometry, METHOD, seed, index
    )
    checks, gate_keys = _diagnostic_checks(straight, dmp)
    report = {
        "schema_version": 1,
        "phase": "4E_diagnostics",
        "config_sha256": config.digest,
        "seed": seed,
        "passed": all(checks[key] for key in gate_keys),
        "gate_keys": gate_keys,
        "checks": checks,
        "dimensions": _physical_dimensions(config, derived),
        "straight": _public_record(straight),
        "dmp_route_confidence": _public_record(dmp),
    }
    _write_json(output / "diagnostics.json", report)
    return report


def reassess_saved_diagnostics(config: Phase4EConfig) -> dict[str, Any]:
    path = config.project_path("output_dir") / "diagnostics.json"
    if not path.is_file():
        raise RuntimeError("No saved Phase 4E diagnostics are available to reassess")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["config_sha256"] != config.digest:
        raise RuntimeError("Saved diagnostics do not match the current configuration")
    checks, gate_keys = _diagnostic_checks(
        report["straight"], report["dmp_route_confidence"]
    )
    report["checks"] = checks
    report["gate_keys"] = gate_keys
    report["passed"] = all(checks[key] for key in gate_keys)
    report["reassessed_without_simulation"] = True
    _write_json(path, report)
    return report


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(row["calibrated_safety_success"] for row in records)
    low, high = wilson_interval(successes, len(records))
    return {
        "episodes": len(records),
        "placement_successes": sum(row["placement_success"] for row in records),
        "calibrated_safety_successes": successes,
        "calibrated_safety_success_rate": successes / len(records),
        "calibrated_safety_success_ci95": [low, high],
        "object_collision_rollouts": sum(row["object_collision_steps"] > 0 for row in records),
        "unintended_contact_rollouts": sum(row["unintended_contact"] for row in records),
        "drop_rollouts": sum(row["drop"] for row in records),
        "grasp_retained_rollouts": sum(row["object_remained_grasped"] for row in records),
        "placement_error_median_m": float(
            np.median([row["placement_error_m"] for row in records])
        ),
        "minimum_obstacle_clearance_median_m": float(
            np.median([row["minimum_obstacle_clearance_m"] for row in records])
        ),
        "transport_height_median_m": float(
            np.median([row["transport_eef_z_median_m"] for row in records])
        ),
        "cube_bottom_clearance_median_m": float(
            np.median([row["transport_cube_bottom_clearance_m"] for row in records])
        ),
        "action_saturation_rate": sum(row["action_saturation_steps"] for row in records)
        / sum(row["steps"] for row in records),
    }


def run_preflight(config: Phase4EConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    diagnostic_path = output / "diagnostics.json"
    if not diagnostic_path.is_file():
        raise RuntimeError("Run the Phase 4E diagnostics before preflight")
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not diagnostics["passed"] or diagnostics["config_sha256"] != config.digest:
        raise RuntimeError("Passing diagnostics for the current configuration are required")
    execution_config, derived = _execution_config(config)
    demos, geometry = build_library(execution_config)
    seeds = [int(value) for value in config.raw["preflight"]["seeds"]]
    records = [
        _run_rollout(config, execution_config, demos, geometry, METHOD, seed, index)[0]
        for index, seed in enumerate(seeds)
    ]
    summary = _summarise(records)
    report = {
        "schema_version": 1,
        "phase": "4E_preflight",
        "config_sha256": config.digest,
        "passed": summary["calibrated_safety_successes"] == 5,
        "seeds": seeds,
        "dimensions": _physical_dimensions(config, derived),
        "aggregate": summary,
        "phase_contacts": aggregate_phase_contacts(records),
    }
    _write_jsonl(output / "preflight_rollouts.jsonl", [_public_record(row) for row in records])
    _write_json(output / "preflight.json", report)
    return report


def run_evaluation(config: Phase4EConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    if not preflight["passed"] or preflight["config_sha256"] != config.digest:
        raise RuntimeError("Passing Phase 4E preflight for this configuration is required")
    execution_config, derived = _execution_config(config)
    demos, geometry = build_library(execution_config)
    evaluation = config.raw["evaluation"]
    seeds = list(
        range(
            int(evaluation["seed_start"]),
            int(evaluation["seed_start"]) + int(evaluation["episode_count"]),
        )
    )
    records = [
        _run_rollout(config, execution_config, demos, geometry, method, seed, index)[0]
        for index, seed in enumerate(seeds)
        for method in METHODS
    ]
    aggregate: dict[str, Any] = {}
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        successes = sum(row["calibrated_safety_success"] for row in rows)
        low, high = wilson_interval(successes, len(rows))
        latencies = np.concatenate(
            [np.asarray(row["_latencies_ms"], dtype=np.float64) for row in rows]
        )
        aggregate[method] = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "success_rate_ci95": [low, high],
            "placement_successes": sum(row["placement_success"] for row in rows),
            "collision_episodes": sum(row["object_collision_steps"] > 0 for row in rows),
            "robot_contact_episodes": sum(row["unintended_contact"] for row in rows),
            "drop_episodes": sum(row["drop"] for row in rows),
            "grasp_retained_episodes": sum(row["object_remained_grasped"] for row in rows),
            "placement_error_median_m": float(
                np.median([row["placement_error_m"] for row in rows])
            ),
            "placement_error_p95_m": float(
                np.percentile([row["placement_error_m"] for row in rows], 95)
            ),
            "steps_median": float(np.median([row["steps"] for row in rows])),
            "path_efficiency_median": float(
                np.median([row["path_efficiency"] for row in rows])
            ),
            "minimum_obstacle_clearance_median_m": float(
                np.median([row["minimum_obstacle_clearance_m"] for row in rows])
            ),
            "minimum_obstacle_clearance_worst_m": float(
                np.min([row["minimum_obstacle_clearance_m"] for row in rows])
            ),
            "cube_bottom_clearance_median_m": float(
                np.median([row["transport_cube_bottom_clearance_m"] for row in rows])
            ),
            "transport_height_median_m": float(
                np.median([row["transport_eef_z_median_m"] for row in rows])
            ),
            "action_saturation_rate": sum(row["action_saturation_steps"] for row in rows)
            / sum(row["steps"] for row in rows),
            "controller_latency_median_ms": float(np.median(latencies)),
            "controller_latency_p95_ms": float(np.percentile(latencies, 95)),
        }
    counts = {method: sum(row["method"] == method for row in records) for method in METHODS}
    if any(count != 50 for count in counts.values()):
        raise RuntimeError(f"Expected exactly 50 calibrated episodes per method, got {counts}")
    _write_jsonl(output / "rollouts.jsonl", [_public_record(row) for row in records])
    contacts_by_method = {}
    for method in METHODS:
        method_records = [row for row in records if row["method"] == method]
        contacts_by_method[method] = {
            "rollouts": len(method_records),
            "by_execution_phase": aggregate_phase_contacts(method_records),
        }
    _write_json(
        output / "contact_diagnostics.json",
        {
            "schema_version": 1,
            "phase": "4E_contacts",
            "rollouts": len(records),
            "method_episode_counts": counts,
            "by_method": contacts_by_method,
        },
    )
    success_choices = [
        row
        for row in records
        if row["method"] == METHOD and row["calibrated_safety_success"]
    ]
    failure_choices = [
        row
        for row in records
        if row["method"] == "straight" and not row["calibrated_safety_success"]
    ]
    if not success_choices or not failure_choices:
        raise RuntimeError("Representative calibrated success and failure are required")
    representatives = [
        ("success", min(success_choices, key=lambda row: row["placement_error_m"])),
        ("failure", min(failure_choices, key=lambda row: row["scenario"]["index"])),
    ]
    media: list[dict[str, Any]] = []
    for label, representative in representatives:
        seed = int(representative["scenario"]["seed"])
        index = int(representative["scenario"]["index"])
        method = str(representative["method"])
        replay, frames = _run_rollout(
            config, execution_config, demos, geometry, method, seed, index, render=True
        )
        if bool(replay["calibrated_safety_success"]) != bool(
            representative["calibrated_safety_success"]
        ):
            raise RuntimeError(f"Representative calibrated {label} did not reproduce")
        filename = f"representative_calibrated_{label}.mp4"
        _write_video(output / filename, execution_config, frames)
        media.append(
            {
                "kind": label,
                "path": filename,
                "method": method,
                "seed": seed,
                "scenario_index": index,
                "frames": len(frames),
            }
        )
    report = {
        "schema_version": 1,
        "phase": "4E",
        "accepted": True,
        "config_sha256": config.digest,
        "seeds": seeds,
        "method_episode_counts": counts,
        "dimensions": _physical_dimensions(config, derived),
        "aggregate": aggregate,
        "representative_artifacts": media,
    }
    _write_json(output / "aggregate.json", report)
    (output / "comparison.md").write_text(
        _comparison_markdown(aggregate, success_label="Calibrated safe success"),
        encoding="utf-8",
    )
    _plot_comparison(
        output / "comparison.png",
        aggregate,
        success_label="Calibrated safe success (%)",
    )
    _write_json(
        output / "run_config.json",
        {"config_sha256": config.digest, "configuration": config.raw},
    )
    return report
