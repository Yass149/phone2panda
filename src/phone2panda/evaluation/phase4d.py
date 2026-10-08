from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from phone2panda.evaluation.phase4b import (
    Demonstration,
    Phase4BConfig,
    _make_environment,
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
    CONTACT_CATEGORIES,
    UNINTENDED_CATEGORIES,
    ContactAccumulator,
)
from phone2panda.provenance import build_run_manifest
from phone2panda.sim.phase4a import Phase4AConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Phase4DConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    phase4b: Phase4BConfig

    def project_path(self, key: str) -> Path:
        return self.root / str(self.raw[key])

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def load_phase4d_config(path: Path) -> Phase4DConfig:
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = resolved.parent.parent
    phase4b = load_phase4b_config(root / str(raw["phase4b_config"]))
    offsets = [float(value) for value in raw["placement"]["lateral_offsets_m"]]
    if offsets != [0.06, 0.09, 0.12]:
        raise ValueError("Phase 4D offset grid must be exactly [0.06, 0.09, 0.12]")
    if len(raw["preflight"]["seeds"]) != 5:
        raise ValueError("Phase 4D preflight requires exactly five seeds")
    if int(raw["evaluation"]["episode_count"]) != 50:
        raise ValueError("Phase 4D evaluation requires exactly 50 episodes")
    return Phase4DConfig(path=resolved, root=root, raw=raw, phase4b=phase4b)


def _execution_config(config: Phase4DConfig) -> Phase4BConfig:
    phase4a_raw = copy.deepcopy(config.phase4b.phase4a.raw)
    placement = config.raw["placement"]
    phase4a_raw["workspace"]["transport_z"] = float(placement["transport_z_m"])
    phase4a_raw["workspace"]["approach_z"] = float(placement["approach_z_m"])
    return Phase4BConfig(
        path=config.phase4b.path,
        root=config.phase4b.root,
        raw=copy.deepcopy(config.phase4b.raw),
        phase4a=Phase4AConfig(root=config.phase4b.phase4a.root, raw=phase4a_raw),
    )


def route_side_staging(
    goal_xy: FloatArray,
    route: str,
    offset_m: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[FloatArray, bool]:
    if route not in {"left", "right"}:
        raise ValueError(f"Unknown route side: {route}")
    sign = 1.0 if route == "left" else -1.0
    requested = np.asarray([goal_xy[0], goal_xy[1] + sign * offset_m], dtype=np.float64)
    clamped = np.asarray(
        [
            np.clip(requested[0], *x_range),
            np.clip(requested[1], *y_range),
        ],
        dtype=np.float64,
    )
    return clamped, bool(np.any(np.abs(clamped - requested) > 1e-12))


def _run_rollout(
    config: Phase4DConfig,
    execution_config: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
    seed: int,
    index: int,
    offset_m: float,
    render: bool = False,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]]]:
    scenario = make_scenario(execution_config, demos, geometry, seed, index)
    path, selection = select_method_path(
        execution_config, demos, scenario, str(config.raw["method"])
    )
    route = str(selection["source_route"])
    workspace = execution_config.phase4a.raw["workspace"]
    staging_xy, was_clamped = route_side_staging(
        scenario.goal_xy,
        route,
        offset_m,
        tuple(float(value) for value in workspace["robot_x_range"]),
        tuple(float(value) for value in workspace["robot_y_range"]),
    )
    contacts = ContactAccumulator()
    contacts.begin(str(seed))
    env = _make_environment(execution_config, scenario, render=render)
    try:
        record, frames = execute_rollout(
            execution_config,
            env,
            scenario,
            str(config.raw["method"]),
            path,
            selection,
            capture_video=render,
            contact_callback=contacts.observe,
            lateral_staging_xy=staging_xy,
        )
    finally:
        env.close()
    contact_summary = contacts.rollout_summary(str(seed))
    unintended = any(
        contact_summary[category]["contact_events"] > 0
        for category in UNINTENDED_CATEGORIES
    )
    record["lateral_offset_m"] = offset_m
    record["lateral_staging_xy_m"] = staging_xy.tolist()
    record["lateral_staging_clamped"] = was_clamped
    record["contacts"] = contact_summary
    record["phase_contacts"] = contacts.phase_summary(str(seed), record["phases"])
    record["placement_success"] = bool(record["task_success"])
    record["unintended_contact"] = unintended
    record["safety_success"] = bool(record["placement_success"] and not unintended)
    return record, frames


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def aggregate_phase_contacts(records: list[dict[str, Any]]) -> dict[str, Any]:
    events: dict[str, Counter[tuple[str, tuple[str, str]]]] = defaultdict(Counter)
    steps: dict[str, Counter[tuple[str, tuple[str, str]]]] = defaultdict(Counter)
    rollouts: dict[str, dict[tuple[str, tuple[str, str]], set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for record in records:
        seed = int(record["scenario"]["seed"])
        for phase, categories in record["phase_contacts"].items():
            for category, details in categories.items():
                for pair in details["pairs"]:
                    key = (category, tuple(pair["geometries"]))
                    events[phase][key] += int(pair["contact_events"])
                    steps[phase][key] += int(pair["contact_steps"])
                    rollouts[phase][key].add(seed)
    result: dict[str, Any] = {}
    for phase in events:
        result[phase] = {}
        for category in CONTACT_CATEGORIES:
            keys = [key for key in events[phase] if key[0] == category]
            result[phase][category] = {
                "contact_events": sum(events[phase][key] for key in keys),
                "contact_steps": sum(steps[phase][key] for key in keys),
                "rollouts": len(
                    set().union(*(rollouts[phase][key] for key in keys)) if keys else set()
                ),
                "pairs": [
                    {
                        "geometries": list(key[1]),
                        "contact_events": events[phase][key],
                        "contact_steps": steps[phase][key],
                        "rollouts": len(rollouts[phase][key]),
                    }
                    for key in sorted(keys)
                ],
            }
    return result


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    safety_successes = sum(row["safety_success"] for row in records)
    low, high = wilson_interval(safety_successes, len(records))
    return {
        "episodes": len(records),
        "placement_successes": sum(row["placement_success"] for row in records),
        "safety_successes": safety_successes,
        "safety_success_rate": safety_successes / len(records),
        "safety_success_rate_ci95": [low, high],
        "unintended_contact_rollouts": sum(row["unintended_contact"] for row in records),
        "robot_obstacle_contact_rollouts": sum(
            row["contacts"]["robot_obstacle"]["contact_events"] > 0 for row in records
        ),
        "robot_table_contact_rollouts": sum(
            row["contacts"]["robot_table"]["contact_events"] > 0 for row in records
        ),
        "self_collision_rollouts": sum(
            row["contacts"]["self_collision"]["contact_events"] > 0 for row in records
        ),
        "placement_error_median_m": float(
            np.median([row["placement_error_m"] for row in records])
        ),
        "minimum_obstacle_clearance_median_m": float(
            np.median([row["minimum_obstacle_clearance_m"] for row in records])
        ),
        "action_saturation_rate": sum(row["action_saturation_steps"] for row in records)
        / sum(row["steps"] for row in records),
        "staging_clamped_rollouts": sum(row["lateral_staging_clamped"] for row in records),
        "phase_contacts": aggregate_phase_contacts(records),
    }


def run_preflight_grid(config: Phase4DConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    output.mkdir(parents=True, exist_ok=True)
    execution_config = _execution_config(config)
    demos, geometry = build_library(execution_config)
    seeds = [int(value) for value in config.raw["preflight"]["seeds"]]
    attempts: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    selected_offset: float | None = None
    for offset in (float(value) for value in config.raw["placement"]["lateral_offsets_m"]):
        records = [
            _run_rollout(
                config, execution_config, demos, geometry, seed, index, offset
            )[0]
            for index, seed in enumerate(seeds)
        ]
        all_records.extend(records)
        summary = _summarise(records)
        passed = summary["placement_successes"] == 5 and summary[
            "unintended_contact_rollouts"
        ] == 0
        attempts.append({"offset_m": offset, "passed": passed, **summary})
        if passed:
            selected_offset = offset
            break
    report = {
        "schema_version": 1,
        "phase": "4D_preflight",
        "config_sha256": config.digest,
        "seeds": seeds,
        "passed": selected_offset is not None,
        "selected_offset_m": selected_offset,
        "attempts": attempts,
    }
    _write_jsonl(output / "preflight_rollouts.jsonl", [_public_record(row) for row in all_records])
    _write_json(output / "preflight_grid.json", report)
    return report


def run_evaluation(config: Phase4DConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    preflight = json.loads((output / "preflight_grid.json").read_text(encoding="utf-8"))
    if not preflight["passed"] or preflight["config_sha256"] != config.digest:
        raise RuntimeError("A passing Phase 4D preflight for this configuration is required")
    offset = float(preflight["selected_offset_m"])
    execution_config = _execution_config(config)
    demos, geometry = build_library(execution_config)
    evaluation = config.raw["evaluation"]
    seeds = list(
        range(
            int(evaluation["seed_start"]),
            int(evaluation["seed_start"]) + int(evaluation["episode_count"]),
        )
    )
    records = [
        _run_rollout(
            config, execution_config, demos, geometry, seed, index, offset
        )[0]
        for index, seed in enumerate(seeds)
    ]
    summary = _summarise(records)
    _write_jsonl(output / "rollouts.jsonl", [_public_record(row) for row in records])
    _write_json(
        output / "phase_contact_diagnostics.json",
        {
            "schema_version": 1,
            "phase": "4D_contacts",
            "rollouts": 50,
            "selected_offset_m": offset,
            "by_execution_phase": summary["phase_contacts"],
        },
    )
    choices = [row for row in records if row["safety_success"]]
    if not choices:
        raise RuntimeError("No safe corrected rollout is available for the representative video")
    representative = min(choices, key=lambda row: row["placement_error_m"])
    seed = int(representative["scenario"]["seed"])
    index = int(representative["scenario"]["index"])
    replay, frames = _run_rollout(
        config, execution_config, demos, geometry, seed, index, offset, render=True
    )
    if not replay["safety_success"]:
        raise RuntimeError("Representative video replay did not reproduce safely")
    _write_video(output / "representative_corrected.mp4", execution_config, frames)
    report = {
        "schema_version": 1,
        "phase": "4D",
        "accepted": summary["safety_successes"] == 50,
        "config_sha256": config.digest,
        "selected_offset_m": offset,
        "seeds": seeds,
        "aggregate": {key: value for key, value in summary.items() if key != "phase_contacts"},
        "representative": {
            "path": "representative_corrected.mp4",
            "seed": seed,
            "scenario_index": index,
            "frames": len(frames),
        },
    }
    _write_json(output / "aggregate.json", report)
    _write_json(
        output / "run_config.json",
        build_run_manifest(config.root, config_sha256=config.digest, configuration=config.raw),
    )
    return report
