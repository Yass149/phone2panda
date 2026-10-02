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

from phone2panda.evaluation.phase4b import (
    Demonstration,
    Phase4BConfig,
    Scenario,
    _make_environment,
    _write_json,
    _write_jsonl,
    build_library,
    execute_rollout,
    load_phase4b_config,
    make_scenario,
    select_method_path,
    signed_rectangle_clearance,
    wilson_interval,
)
from phone2panda.sim.phase4a import Phase4AConfig

CONTACT_CATEGORIES = (
    "intended_gripper_cube",
    "object_obstacle",
    "robot_obstacle",
    "robot_table",
    "self_collision",
    "other",
)
UNINTENDED_CATEGORIES = ("robot_obstacle", "robot_table", "self_collision")


@dataclass(frozen=True)
class Phase4CConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    phase4b: Phase4BConfig

    def project_path(self, key: str) -> Path:
        return self.root / str(self.raw[key])

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def load_phase4c_config(path: Path) -> Phase4CConfig:
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = resolved.parent.parent
    phase4b = load_phase4b_config(root / str(raw["phase4b_config"]))
    if str(raw["method"]) != "dmp_route_confidence":
        raise ValueError("Phase 4C audits only the winning route/confidence method")
    if len(raw["preflight"]["seeds"]) != 5:
        raise ValueError("Phase 4C preflight requires exactly five seeds")
    if int(raw["evaluation"]["episode_count"]) != 50:
        raise ValueError("Phase 4C evaluation requires exactly 50 episodes")
    return Phase4CConfig(path=resolved, root=root, raw=raw, phase4b=phase4b)


def _is_robot(name: str) -> bool:
    return name.startswith(("robot0_", "gripper0_"))


def _is_cube(name: str) -> bool:
    return name.startswith("cube_")


def _is_gripper(name: str) -> bool:
    return name.startswith("gripper0_")


def classify_contact_pair(first: str, second: str) -> str:
    names = (first, second)
    if any(_is_cube(name) for name in names) and any(_is_gripper(name) for name in names):
        return "intended_gripper_cube"
    if "route_obstacle_geom" in names and any(_is_cube(name) for name in names):
        return "object_obstacle"
    if "route_obstacle_geom" in names and any(_is_robot(name) for name in names):
        return "robot_obstacle"
    if "table_collision" in names and any(_is_robot(name) for name in names):
        return "robot_table"
    if all(_is_robot(name) for name in names):
        return "self_collision"
    return "other"


class ContactAccumulator:
    def __init__(self) -> None:
        self.rollout_id = ""
        self.events: Counter[tuple[str, tuple[str, str]]] = Counter()
        self.steps: dict[tuple[str, tuple[str, str]], set[tuple[str, int]]] = defaultdict(set)
        self.rollouts: dict[tuple[str, tuple[str, str]], set[str]] = defaultdict(set)
        self.per_rollout_events: dict[str, Counter[tuple[str, tuple[str, str]]]] = {}
        self.per_rollout_steps: dict[
            str, dict[tuple[str, tuple[str, str]], set[int]]
        ] = {}
        self.per_rollout_step_events: dict[
            str, Counter[tuple[int, str, tuple[str, str]]]
        ] = {}

    def begin(self, rollout_id: str) -> None:
        self.rollout_id = rollout_id
        self.per_rollout_events[rollout_id] = Counter()
        self.per_rollout_steps[rollout_id] = defaultdict(set)
        self.per_rollout_step_events[rollout_id] = Counter()

    def observe(self, step_index: int, first: str, second: str) -> None:
        if not self.rollout_id:
            raise RuntimeError("Call begin() before observing contacts")
        pair = tuple(sorted((first, second)))
        category = classify_contact_pair(*pair)
        key = (category, pair)
        self.events[key] += 1
        self.steps[key].add((self.rollout_id, step_index))
        self.rollouts[key].add(self.rollout_id)
        self.per_rollout_events[self.rollout_id][key] += 1
        self.per_rollout_steps[self.rollout_id][key].add(step_index)
        self.per_rollout_step_events[self.rollout_id][(step_index, category, pair)] += 1

    def rollout_summary(self, rollout_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        events = self.per_rollout_events[rollout_id]
        steps = self.per_rollout_steps[rollout_id]
        for category in CONTACT_CATEGORIES:
            keys = [key for key in events if key[0] == category]
            payload[category] = {
                "contact_events": sum(events[key] for key in keys),
                "contact_steps": len(set().union(*(steps[key] for key in keys))) if keys else 0,
                "pairs": [
                    {
                        "geometries": list(key[1]),
                        "contact_events": events[key],
                        "contact_steps": len(steps[key]),
                    }
                    for key in sorted(keys)
                ],
            }
        return payload

    def phase_summary(
        self, rollout_id: str, phases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        boundaries: list[tuple[int, int, str]] = []
        start = 0
        for phase in phases:
            end = start + int(phase["steps"])
            boundaries.append((start, end, str(phase["name"])))
            start = end
        phase_events: dict[str, Counter[tuple[str, tuple[str, str]]]] = defaultdict(Counter)
        phase_steps: dict[
            str, dict[tuple[str, tuple[str, str]], set[int]]
        ] = defaultdict(lambda: defaultdict(set))
        for (step, category, pair), count in self.per_rollout_step_events[rollout_id].items():
            phase_name = next(
                (name for low, high, name in boundaries if low <= step < high),
                "unattributed",
            )
            key = (category, pair)
            phase_events[phase_name][key] += count
            phase_steps[phase_name][key].add(step)
        payload: dict[str, Any] = {}
        for _, _, phase_name in boundaries:
            events = phase_events[phase_name]
            steps = phase_steps[phase_name]
            payload[phase_name] = {}
            for category in CONTACT_CATEGORIES:
                keys = [key for key in events if key[0] == category]
                payload[phase_name][category] = {
                    "contact_events": sum(events[key] for key in keys),
                    "contact_steps": len(
                        set().union(*(steps[key] for key in keys)) if keys else set()
                    ),
                    "pairs": [
                        {
                            "geometries": list(key[1]),
                            "contact_events": events[key],
                            "contact_steps": len(steps[key]),
                        }
                        for key in sorted(keys)
                    ],
                }
        return payload

    def aggregate(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for category in CONTACT_CATEGORIES:
            keys = [key for key in self.events if key[0] == category]
            category_steps = set().union(*(self.steps[key] for key in keys)) if keys else set()
            category_rollouts = (
                set().union(*(self.rollouts[key] for key in keys)) if keys else set()
            )
            payload[category] = {
                "contact_events": sum(self.events[key] for key in keys),
                "contact_steps": len(category_steps),
                "rollouts": len(category_rollouts),
                "pairs": [
                    {
                        "geometries": list(key[1]),
                        "contact_events": self.events[key],
                        "contact_steps": len(self.steps[key]),
                        "rollouts": len(self.rollouts[key]),
                    }
                    for key in sorted(keys)
                ],
            }
        return payload


def _phase4b_winner_records(config: Phase4CConfig) -> list[dict[str, Any]]:
    path = config.phase4b.project_path("output_dir") / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    winner = [row for row in rows if row["method"] == config.raw["method"]]
    if len(winner) != 50:
        raise ValueError(f"Expected 50 original winning-method records, found {len(winner)}")
    return winner


def audit_original_contacts(config: Phase4CConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    output.mkdir(parents=True, exist_ok=True)
    original = _phase4b_winner_records(config)
    demos, geometry = build_library(config.phase4b)
    contacts = ContactAccumulator()
    replays: list[dict[str, Any]] = []
    for expected in original:
        scenario_data = expected["scenario"]
        scenario = make_scenario(
            config.phase4b,
            demos,
            geometry,
            int(scenario_data["seed"]),
            int(scenario_data["index"]),
        )
        path, selection = select_method_path(
            config.phase4b, demos, scenario, str(config.raw["method"])
        )
        rollout_id = str(scenario.seed)
        contacts.begin(rollout_id)
        env = _make_environment(config.phase4b, scenario, render=False)
        try:
            replay, _ = execute_rollout(
                config.phase4b,
                env,
                scenario,
                str(config.raw["method"]),
                path,
                selection,
                contact_callback=contacts.observe,
            )
        finally:
            env.close()
        if replay["robot_obstacle_contact_steps"] != expected["robot_obstacle_contact_steps"]:
            raise RuntimeError(f"Contact replay mismatch for seed {scenario.seed}")
        replays.append(
            {
                "seed": scenario.seed,
                "scenario_index": scenario.index,
                "legacy_robot_obstacle_contact_steps": replay[
                    "robot_obstacle_contact_steps"
                ],
                "contacts": contacts.rollout_summary(rollout_id),
            }
        )
    aggregate = contacts.aggregate()
    unintended = {
        category: aggregate[category] for category in UNINTENDED_CATEGORIES
    }
    report = {
        "schema_version": 1,
        "phase": "4C_original_contact_audit",
        "source_rollouts": "../phase4b/rollouts.jsonl",
        "rollouts": 50,
        "existing_log_robot_contact_rollouts": sum(
            bool(row["robot_obstacle_contact"]) for row in original
        ),
        "existing_log_robot_contact_steps": sum(
            int(row["robot_obstacle_contact_steps"]) for row in original
        ),
        "deterministic_replay_matched_existing_logs": True,
        "contains_unintended_contacts": any(
            details["contact_events"] > 0 for details in unintended.values()
        ),
        "aggregate": aggregate,
        "per_rollout": replays,
    }
    _write_json(output / "original_contact_diagnostics.json", report)
    return report


def _safety_config(config: Phase4CConfig) -> Phase4BConfig:
    phase4b_raw = copy.deepcopy(config.phase4b.raw)
    phase4a_raw = copy.deepcopy(config.phase4b.phase4a.raw)
    safety = config.raw["safety"]
    phase4a_raw["workspace"]["transport_z"] = float(safety["transport_z_m"])
    phase4a_raw["workspace"]["approach_z"] = float(safety["approach_z_m"])
    phase4b_raw["scenario"]["object_radius_m"] = float(
        safety["robot_clearance_radius_m"]
    )
    phase4a = Phase4AConfig(root=config.phase4b.phase4a.root, raw=phase4a_raw)
    return Phase4BConfig(
        path=config.phase4b.path,
        root=config.phase4b.root,
        raw=phase4b_raw,
        phase4a=phase4a,
    )


def _safety_path(
    config: Phase4CConfig,
    safe_config: Phase4BConfig,
    demos: list[Demonstration],
    scenario: Scenario,
) -> tuple[np.ndarray, dict[str, Any]]:
    path, selection = select_method_path(
        safe_config, demos, scenario, str(config.raw["method"])
    )
    safety_radius = float(config.raw["safety"]["robot_clearance_radius_m"])
    selection["predicted_robot_geometry_clearance_m"] = float(
        np.min(
            signed_rectangle_clearance(
                path,
                scenario.obstacle_xy,
                scenario.obstacle_half_size,
                safety_radius,
            )
        )
    )
    return path, selection


def _run_safe_rollout(
    config: Phase4CConfig,
    safe_config: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
    seed: int,
    index: int,
    render: bool = False,
    contacts: ContactAccumulator | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    scenario = make_scenario(safe_config, demos, geometry, seed, index)
    path, selection = _safety_path(config, safe_config, demos, scenario)
    active_contacts = contacts or ContactAccumulator()
    active_contacts.begin(str(seed))
    env = _make_environment(safe_config, scenario, render=render)
    try:
        record, frames = execute_rollout(
            safe_config,
            env,
            scenario,
            str(config.raw["method"]),
            path,
            selection,
            capture_video=render,
            contact_callback=active_contacts.observe,
            pre_lower_yaw_steps=int(config.raw["safety"]["pre_lower_yaw_steps"]),
            pre_lower_yaw_command=float(
                config.raw["safety"]["pre_lower_yaw_command"]
            ),
            goal_side_lower_offset_m=float(
                config.raw["safety"]["goal_side_lower_offset_m"]
            ),
        )
    finally:
        env.close()
    contact_summary = active_contacts.rollout_summary(str(seed))
    unintended = {
        category: contact_summary[category] for category in UNINTENDED_CATEGORIES
    }
    record["contacts"] = contact_summary
    record["unintended_contact"] = any(
        details["contact_events"] > 0 for details in unintended.values()
    )
    record["object_task_success"] = bool(record["task_success"])
    record["safety_success"] = bool(record["task_success"] and not record["unintended_contact"])
    return record, frames


def run_safety_preflight(config: Phase4CConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    audit_path = output / "original_contact_diagnostics.json"
    if not audit_path.is_file():
        raise RuntimeError("Run the original contact audit before the safety preflight")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["contains_unintended_contacts"]:
        raise RuntimeError("Original audit found no unintended contacts; correction is unnecessary")
    safe_config = _safety_config(config)
    demos, geometry = build_library(safe_config)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, value in enumerate(config.raw["preflight"]["seeds"]):
        seed = int(value)
        try:
            record, _ = _run_safe_rollout(
                config, safe_config, demos, geometry, seed, index
            )
            records.append(record)
            if not record["safety_success"]:
                failures.append(
                    {
                        "seed": seed,
                        "failure_reasons": record["failure_reasons"],
                        "unintended_contact": record["unintended_contact"],
                    }
                )
        except Exception as error:  # noqa: BLE001 - persisted audit outcome
            failures.append({"seed": seed, "error": f"{type(error).__name__}: {error}"})
            break
    report = {
        "schema_version": 1,
        "phase": "4C_preflight",
        "config_sha256": config.digest,
        "passed": len(records) == 5 and not failures,
        "rollouts": len(records),
        "safety_successes": sum(row["safety_success"] for row in records),
        "unintended_contact_rollouts": sum(row["unintended_contact"] for row in records),
        "failures": failures,
    }
    _write_jsonl(output / "preflight_rollouts.jsonl", [_public_record(row) for row in records])
    _write_json(output / "preflight.json", report)
    return report


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _corrected_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    safety_successes = sum(row["safety_success"] for row in records)
    low, high = wilson_interval(safety_successes, len(records))
    return {
        "episodes": len(records),
        "object_task_successes": sum(row["object_task_success"] for row in records),
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
    }


def _comparison(
    config: Phase4CConfig, corrected: dict[str, Any], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    original = json.loads(
        (config.phase4b.project_path("output_dir") / "aggregate.json").read_text(
            encoding="utf-8"
        )
    )["aggregate"]["dmp_route_confidence"]
    return {
        "original_phase4b": {
            "episodes": original["episodes"],
            "successes": original["successes"],
            "success_rate": original["success_rate"],
            "unintended_contact_rollouts": diagnostics["aggregate"]["robot_obstacle"][
                "rollouts"
            ]
            + diagnostics["aggregate"]["robot_table"]["rollouts"]
            + diagnostics["aggregate"]["self_collision"]["rollouts"],
            "placement_error_median_m": original["placement_error_median_m"],
            "minimum_obstacle_clearance_median_m": original[
                "minimum_obstacle_clearance_median_m"
            ],
            "action_saturation_rate": original["action_saturation_rate"],
        },
        "corrected_phase4c": corrected,
    }


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    original = comparison["original_phase4b"]
    corrected = comparison["corrected_phase4c"]
    return (
        "| Result | Success | Unintended-contact rollouts | Placement median | "
        "Clearance median | Saturation |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| Original Phase 4B | {original['successes']}/{original['episodes']} | "
        f"{original['unintended_contact_rollouts']} | "
        f"{1000 * original['placement_error_median_m']:.1f} mm | "
        f"{1000 * original['minimum_obstacle_clearance_median_m']:.1f} mm | "
        f"{100 * original['action_saturation_rate']:.2f}% |\n"
        f"| Corrected Phase 4C | {corrected['safety_successes']}/{corrected['episodes']} | "
        f"{corrected['unintended_contact_rollouts']} | "
        f"{1000 * corrected['placement_error_median_m']:.1f} mm | "
        f"{1000 * corrected['minimum_obstacle_clearance_median_m']:.1f} mm | "
        f"{100 * corrected['action_saturation_rate']:.2f}% |\n"
    )


def run_safety_evaluation(config: Phase4CConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    if not preflight["passed"] or preflight["config_sha256"] != config.digest:
        raise RuntimeError("A passing Phase 4C preflight for this configuration is required")
    diagnostics = json.loads(
        (output / "original_contact_diagnostics.json").read_text(encoding="utf-8")
    )
    safe_config = _safety_config(config)
    demos, geometry = build_library(safe_config)
    evaluation = config.raw["evaluation"]
    seeds = list(
        range(
            int(evaluation["seed_start"]),
            int(evaluation["seed_start"]) + int(evaluation["episode_count"]),
        )
    )
    corrected_contacts = ContactAccumulator()
    records = [
        _run_safe_rollout(
            config,
            safe_config,
            demos,
            geometry,
            seed,
            index,
            contacts=corrected_contacts,
        )[0]
        for index, seed in enumerate(seeds)
    ]
    if len(records) != 50:
        raise RuntimeError(f"Expected 50 corrected rollouts, found {len(records)}")
    corrected = _corrected_aggregate(records)
    if corrected["safety_successes"] != 50:
        raise RuntimeError(
            f"Corrected safety gate failed: {corrected['safety_successes']}/50 successes"
        )
    _write_jsonl(output / "rollouts.jsonl", [_public_record(row) for row in records])
    contact_report = {
        "schema_version": 1,
        "phase": "4C_corrected_contact_diagnostics",
        "rollouts": 50,
        "aggregate": corrected_contacts.aggregate(),
        "per_rollout": [
            {"seed": row["scenario"]["seed"], "contacts": row["contacts"]}
            for row in records
        ],
    }
    _write_json(output / "contact_diagnostics.json", contact_report)
    comparison = _comparison(config, corrected, diagnostics)
    _write_json(output / "comparison.json", comparison)
    (output / "comparison.md").write_text(
        _comparison_markdown(comparison), encoding="utf-8"
    )
    representative = min(records, key=lambda row: row["placement_error_m"])
    seed = int(representative["scenario"]["seed"])
    index = int(representative["scenario"]["index"])
    replay, frames = _run_safe_rollout(
        config, safe_config, demos, geometry, seed, index, render=True
    )
    if not replay["safety_success"]:
        raise RuntimeError("Representative corrected replay did not reproduce safely")
    from phone2panda.evaluation.phase4b import _write_video

    _write_video(output / "representative_corrected.mp4", safe_config, frames)
    report = {
        "schema_version": 1,
        "phase": "4C",
        "accepted": True,
        "config_sha256": config.digest,
        "seeds": seeds,
        "aggregate": corrected,
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
        {"config_sha256": config.digest, "configuration": config.raw},
    )
    return report
