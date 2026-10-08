from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml
from numpy.typing import NDArray

from phone2panda.evaluation.phase4b import (
    Demonstration,
    Phase4BConfig,
    Scenario,
    _make_environment,
    _write_json,
    _write_jsonl,
    _write_video,
    build_library,
    execute_rollout,
    make_scenario,
    select_method_path,
)
from phone2panda.evaluation.phase4c import UNINTENDED_CATEGORIES, ContactAccumulator
from phone2panda.evaluation.phase4e import (
    Phase4EConfig,
    _execution_config,
    load_phase4e_config,
)
from phone2panda.evaluation.safety import evaluate_safe_task
from phone2panda.policy.gru import GRUPolicy, Normalizer, pad_sequences
from phone2panda.retarget.mapping import bounded_action

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

FloatArray = NDArray[np.float64]
ActionProvider = Callable[[FloatArray, float, FloatArray, FloatArray], FloatArray]
METHOD = "dmp_route_confidence"
FEATURE_NAMES = (
    "eef_x_m",
    "eef_y_m",
    "eef_z_m",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "active_target_x_m",
    "active_target_y_m",
    "active_target_z_m",
    "target_delta_x_m",
    "target_delta_y_m",
    "target_delta_z_m",
    "task_goal_x_m",
    "task_goal_y_m",
    "obstacle_x_m",
    "obstacle_y_m",
    "obstacle_half_x_m",
    "obstacle_half_y_m",
    "route_left",
    "route_right",
    "gripper_state",
    "desired_gripper",
)
ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "gripper",
)


@dataclass(frozen=True)
class Phase5AConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    phase4e: Phase4EConfig

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def project_path(self, key: str) -> Path:
        return self.root / str(self.raw[key])


@dataclass
class TeacherEpisode:
    seed: int
    scenario_index: int
    features: FloatArray
    actions: FloatArray
    record: dict[str, Any]


def load_phase5a_config(path: Path) -> Phase5AConfig:
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = resolved.parent.parent
    phase4e = load_phase4e_config(root / str(raw["phase4e_config"]))
    seeds = [int(value) for value in raw["teacher"]["seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Phase 5A requires exactly five unique teacher seeds")
    if str(raw["teacher"]["method"]) != METHOD:
        raise ValueError(f"Phase 5A teacher must be {METHOD}")
    if int(raw["training"]["maximum_epochs"]) > 500:
        raise ValueError("Phase 5A training cap must not exceed 500 epochs")
    return Phase5AConfig(resolved, root, raw, phase4e)


def policy_features(
    eef: FloatArray,
    cube: FloatArray,
    target: FloatArray,
    scenario: Scenario,
    route: str,
    gripper_state: float,
    desired_gripper: float,
) -> FloatArray:
    if route not in {"left", "right"}:
        raise ValueError(f"Unsupported route: {route}")
    return np.concatenate(
        [
            np.asarray(eef, dtype=np.float64),
            np.asarray(cube, dtype=np.float64),
            np.asarray(target, dtype=np.float64),
            np.asarray(target, dtype=np.float64) - np.asarray(eef, dtype=np.float64),
            scenario.goal_xy,
            scenario.obstacle_xy,
            scenario.obstacle_half_size,
            np.asarray([route == "left", route == "right"], dtype=np.float64),
            np.asarray([gripper_state, desired_gripper], dtype=np.float64),
        ]
    )


def _scenario_and_path(
    execution: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
    seed: int,
    index: int,
) -> tuple[Scenario, FloatArray, dict[str, Any]]:
    scenario = make_scenario(execution, demos, geometry, seed, index)
    path, selection = select_method_path(execution, demos, scenario, METHOD)
    return scenario, path, selection


def _complete_record(
    config: Phase5AConfig,
    record: dict[str, Any],
    contacts: ContactAccumulator,
    rollout_id: str,
) -> dict[str, Any]:
    summary = contacts.rollout_summary(rollout_id)
    record["contacts"] = summary
    record["phase_contacts"] = contacts.phase_summary(rollout_id, record["phases"])
    record["object_remained_grasped"] = bool(
        not record["drop"]
        and record["maximum_transport_grasp_distance_m"]
        <= float(config.phase4e.raw["calibration"]["maximum_grasp_distance_m"])
    )
    record["unintended_contact"] = any(
        summary[category]["contact_events"] > 0 for category in UNINTENDED_CATEGORIES
    )
    placement_limit = float(config.raw["gate"]["maximum_placement_error_m"])
    record["acceptable_placement"] = bool(
        record["target_placed"] and record["placement_error_m"] <= placement_limit
    )
    outcome = evaluate_safe_task(
        placement_succeeded=record["acceptable_placement"],
        object_collision=record["object_collision_steps"] > 0,
        unintended_robot_contact=record["unintended_contact"],
        dropped=record["drop"],
        grasp_retained=record["object_remained_grasped"],
    )
    record["phase5a_gate_success"] = outcome.success
    record["safety_failure_reasons"] = list(outcome.failure_reasons)
    return record


def _run_episode(
    config: Phase5AConfig,
    execution: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
    seed: int,
    index: int,
    provider_factory: Callable[[Scenario, dict[str, Any]], ActionProvider],
    *,
    render: bool = False,
    method_name: str = "gru_policy",
) -> tuple[dict[str, Any], list[NDArray[np.uint8]]]:
    np.random.seed(seed)
    scenario, path, selection = _scenario_and_path(execution, demos, geometry, seed, index)
    provider = provider_factory(scenario, selection)
    contacts = ContactAccumulator()
    rollout_id = f"phase5a:{seed}"
    contacts.begin(rollout_id)
    env = _make_environment(execution, scenario, render=render)
    try:
        record, frames = execute_rollout(
            execution,
            env,
            scenario,
            method_name,
            path,
            selection,
            capture_video=render,
            contact_callback=contacts.observe,
            monitor_transport_height_drop=False,
            action_provider=provider,
        )
    finally:
        env.close()
    return _complete_record(config, record, contacts, rollout_id), frames


def collect_teacher_episodes(
    config: Phase5AConfig,
    execution: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
) -> list[TeacherEpisode]:
    episodes: list[TeacherEpisode] = []
    controller = execution.phase4a.raw["controller"]
    seeds = [int(value) for value in config.raw["teacher"]["seeds"]]
    for index, seed in enumerate(seeds):
        feature_rows: list[FloatArray] = []
        action_rows: list[FloatArray] = []

        def factory(
            scenario: Scenario,
            selection: dict[str, Any],
            feature_rows: list[FloatArray] = feature_rows,
            action_rows: list[FloatArray] = action_rows,
        ) -> ActionProvider:
            route = str(selection["source_route"])
            gripper_state = -1.0

            def provider(
                target: FloatArray,
                desired_gripper: float,
                eef: FloatArray,
                cube: FloatArray,
            ) -> FloatArray:
                nonlocal gripper_state
                features = policy_features(
                    eef,
                    cube,
                    target,
                    scenario,
                    route,
                    gripper_state,
                    desired_gripper,
                )
                action, _ = bounded_action(
                    target - eef,
                    float(controller["position_output_limit"]),
                    desired_gripper,
                )
                feature_rows.append(features)
                action_rows.append(action)
                gripper_state = desired_gripper
                return action

            return provider

        record, _ = _run_episode(
            config, execution, demos, geometry, seed, index, factory
        )
        if not record["phase5a_gate_success"]:
            raise RuntimeError(f"Teacher rollout seed {seed} did not pass the Phase 5A gate")
        episodes.append(
            TeacherEpisode(
                seed=seed,
                scenario_index=index,
                features=np.asarray(feature_rows, dtype=np.float64),
                actions=np.asarray(action_rows, dtype=np.float64),
                record=record,
            )
        )
    if len(episodes) != 5:
        raise RuntimeError("Exactly five successful teacher episodes are required")
    return episodes


def _save_teacher_dataset(path: Path, episodes: list[TeacherEpisode]) -> None:
    batch = pad_sequences(
        [episode.features for episode in episodes],
        [episode.actions for episode in episodes],
    )
    provenance = [
        {
            "seed": episode.seed,
            "scenario_index": episode.scenario_index,
            "source_episode": episode.record["source_episode"],
            "source_route": episode.record["source_route"],
            "source_confidence": episode.record["source_confidence"],
        }
        for episode in episodes
    ]
    np.savez_compressed(
        path,
        states_and_context=batch.inputs,
        teacher_actions=batch.targets,
        valid_mask=batch.mask,
        episode_lengths=batch.lengths,
        seeds=np.asarray([episode.seed for episode in episodes], dtype=np.int64),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def _save_training_curve(
    csv_path: Path, plot_path: Path, history: list[dict[str, float | int]]
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "loss", "gradient_norm"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(history)
    figure, axis = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    axis.plot([int(row["epoch"]) for row in history], [float(row["loss"]) for row in history])
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized action MSE")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    figure.savefig(plot_path, dpi=160, metadata={"Software": "phone2panda"})
    plt.close(figure)


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def run_phase5a(config: Phase5AConfig) -> dict[str, Any]:
    execution, dimensions = _execution_config(config.phase4e)
    demos, geometry = build_library(execution)
    teacher = collect_teacher_episodes(config, execution, demos, geometry)
    feature_values = np.concatenate([episode.features for episode in teacher])
    action_values = np.concatenate([episode.actions for episode in teacher])
    input_normalizer = Normalizer.fit(feature_values, minimum_scale=1e-5)
    output_normalizer = Normalizer.fit(action_values, minimum_scale=1e-3)
    normalized_inputs = [input_normalizer.transform(episode.features) for episode in teacher]
    normalized_actions = [output_normalizer.transform(episode.actions) for episode in teacher]
    batch = pad_sequences(normalized_inputs, normalized_actions)

    policy_config = config.raw["policy"]
    policy = GRUPolicy(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=int(policy_config["hidden_size"]),
        output_dim=len(ACTION_NAMES),
        seed=int(policy_config["seed"]),
    )
    training = config.raw["training"]
    started = time.perf_counter()
    history = policy.fit(
        batch,
        maximum_epochs=int(training["maximum_epochs"]),
        learning_rate=float(training["learning_rate"]),
        patience=int(training["patience"]),
        minimum_improvement=float(training["minimum_improvement"]),
        target_loss=float(training["target_loss"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    training_seconds = time.perf_counter() - started

    def policy_factory(
        scenario: Scenario, selection: dict[str, Any]
    ) -> ActionProvider:
        route = str(selection["source_route"])
        hidden = policy.initial_state()
        gripper_state = -1.0

        def provider(
            target: FloatArray,
            desired_gripper: float,
            eef: FloatArray,
            cube: FloatArray,
        ) -> FloatArray:
            nonlocal hidden, gripper_state
            features = policy_features(
                eef,
                cube,
                target,
                scenario,
                route,
                gripper_state,
                desired_gripper,
            )
            normalized = input_normalizer.transform(features)
            prediction, hidden = policy.step(normalized, hidden)
            action = output_normalizer.inverse(prediction[0])
            gripper_state = desired_gripper
            return np.clip(action, -1.0, 1.0)

        return provider

    seeds = [int(value) for value in config.raw["teacher"]["seeds"]]
    evaluation_records = [
        _run_episode(
            config,
            execution,
            demos,
            geometry,
            seed,
            index,
            policy_factory,
        )[0]
        for index, seed in enumerate(seeds)
    ]
    passed = all(record["phase5a_gate_success"] for record in evaluation_records)
    summary = {
        "episodes": 5,
        "successes": sum(record["phase5a_gate_success"] for record in evaluation_records),
        "placements": sum(record["acceptable_placement"] for record in evaluation_records),
        "object_collision_rollouts": sum(
            record["object_collision_steps"] > 0 for record in evaluation_records
        ),
        "unintended_contact_rollouts": sum(
            record["unintended_contact"] for record in evaluation_records
        ),
        "drop_rollouts": sum(record["drop"] for record in evaluation_records),
        "grasp_retained_rollouts": sum(
            record["object_remained_grasped"] for record in evaluation_records
        ),
        "placement_errors_m": [record["placement_error_m"] for record in evaluation_records],
        "placement_error_max_m": max(
            record["placement_error_m"] for record in evaluation_records
        ),
        "controller_latency_median_ms": float(
            np.median(
                np.concatenate(
                    [np.asarray(record["_latencies_ms"]) for record in evaluation_records]
                )
            )
        ),
        "controller_latency_p95_ms": float(
            np.percentile(
                np.concatenate(
                    [np.asarray(record["_latencies_ms"]) for record in evaluation_records]
                ),
                95,
            )
        ),
    }
    report = {
        "schema_version": 1,
        "phase": "5A",
        "passed": passed,
        "config_sha256": config.digest,
        "teacher": {
            "episodes": 5,
            "seeds": seeds,
            "all_successful": True,
            "method": METHOD,
            "phase4e_config_sha256": config.phase4e.digest,
        },
        "policy": {
            "type": "normalized_gru",
            "selection_reason": (
                "Reproducible compact recurrent policy for modest hardware and controlled "
                "latency evaluation"
            ),
            "hidden_size": policy.hidden_dim,
            "parameter_count": policy.parameter_count,
        },
        "training": {
            "seconds": training_seconds,
            "epochs": len(history),
            "maximum_epochs": int(training["maximum_epochs"]),
            "final_recorded_loss": float(history[-1]["loss"]),
            "best_recorded_loss": min(float(row["loss"]) for row in history),
        },
        "gate": {
            "maximum_placement_error_m": float(
                config.raw["gate"]["maximum_placement_error_m"]
            ),
            **summary,
        },
        "dimensions": dimensions,
    }
    if not passed:
        return report

    output = config.project_path("output_dir")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "gru_policy.npz"
    metadata = {
        "schema_version": 1,
        "config_sha256": config.digest,
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "teacher_seeds": seeds,
        "teacher_method": METHOD,
        "phase4e_config_sha256": config.phase4e.digest,
    }
    policy.save(checkpoint, input_normalizer, output_normalizer, metadata)
    report["policy"]["checkpoint_bytes"] = checkpoint.stat().st_size
    _save_teacher_dataset(output / "teacher_dataset.npz", teacher)
    schema = {
        "schema_version": 1,
        "episode_axis": 0,
        "time_axis": 1,
        "episode_boundaries": "episode_lengths and valid_mask",
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "teacher_method": METHOD,
        "provenance": [
            {
                "seed": episode.seed,
                "scenario_index": episode.scenario_index,
                "source_episode": episode.record["source_episode"],
                "source_route": episode.record["source_route"],
                "source_confidence": episode.record["source_confidence"],
                "phase4e_config_sha256": config.phase4e.digest,
            }
            for episode in teacher
        ],
    }
    _write_json(output / "dataset_schema.json", schema)
    _save_training_curve(output / "training_curve.csv", output / "training_curve.png", history)
    _write_jsonl(
        output / "evaluation_rollouts.jsonl",
        [_public_record(record) for record in evaluation_records],
    )
    representative_seed = int(config.raw["video"]["representative_seed"])
    representative_index = seeds.index(representative_seed)
    replay, frames = _run_episode(
        config,
        execution,
        demos,
        geometry,
        representative_seed,
        representative_index,
        policy_factory,
        render=True,
    )
    if not replay["phase5a_gate_success"]:
        raise RuntimeError("Representative policy rollout did not reproduce the passing gate")
    _write_video(output / "representative_success.mp4", execution, frames)
    report["artifacts"] = {
        "checkpoint": "gru_policy.npz",
        "teacher_dataset": "teacher_dataset.npz",
        "dataset_schema": "dataset_schema.json",
        "training_curve_csv": "training_curve.csv",
        "training_curve_plot": "training_curve.png",
        "evaluation_rollouts": "evaluation_rollouts.jsonl",
        "representative_video": "representative_success.mp4",
    }
    _write_json(output / "evaluation.json", report)
    _write_json(
        output / "run_config.json",
        {"config_sha256": config.digest, "configuration": config.raw},
    )
    return report
