from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
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
    _write_json,
    _write_jsonl,
    _write_video,
    build_library,
    make_scenario,
    select_method_path,
)
from phone2panda.evaluation.phase4e import _execution_config
from phone2panda.evaluation.phase5a import (
    ACTION_NAMES,
    FEATURE_NAMES,
    ActionProvider,
    Phase5AConfig,
    TeacherEpisode,
    _public_record,
    _run_episode,
    load_phase5a_config,
    policy_features,
)
from phone2panda.policy.gru import GRUPolicy, Normalizer, pad_sequences
from phone2panda.retarget.mapping import bounded_action

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

FloatArray = NDArray[np.float64]
METHOD = "dmp_route_confidence"
START_IDS = ("s1", "s2", "s3")


@dataclass(frozen=True)
class Phase5BConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    phase5a: Phase5AConfig

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def project_path(self, key: str) -> Path:
        return self.root / str(self.raw[key])


def load_phase5b_config(path: Path) -> Phase5BConfig:
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = resolved.parent.parent
    phase5a = load_phase5a_config(root / str(raw["phase5a_config"]))
    expected = {"training": 120, "validation": 20, "testing": 50}
    for split, total in expected.items():
        counts = raw["splits"][split]["start_counts"]
        if tuple(counts) != START_IDS or sum(int(value) for value in counts.values()) != total:
            raise ValueError(f"Phase 5B {split} split must contain {total} balanced scenarios")
    seed_bases = [int(raw["splits"][split]["seed_base"]) for split in expected]
    if len(set(seed_bases)) != 3:
        raise ValueError("Phase 5B split seed bases must be distinct")
    if int(raw["policy"]["hidden_size"]) != 24:
        raise ValueError("Phase 5B reuses the validated 24-unit GRU")
    if int(raw["gate"]["minimum_safe_successes"]) != 45:
        raise ValueError("Phase 5B requires at least 45/50 safe successes")
    return Phase5BConfig(resolved, root, raw, phase5a)


def _route_counts_for_start(start_id: str, total: int) -> dict[str, int]:
    if start_id == "s1":
        return {"left": total, "right": 0}
    if start_id == "s3":
        return {"left": 0, "right": total}
    if total % 2:
        raise ValueError("The s2 quota must be even to balance routes")
    return {"left": total // 2, "right": total // 2}


def build_balanced_manifest(
    config: Phase5BConfig,
    execution: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    splits: dict[str, list[dict[str, Any]]] = {}
    all_seeds: set[int] = set()
    for split_name, split_config in config.raw["splits"].items():
        base = int(split_config["seed_base"])
        records: list[dict[str, Any]] = []
        for start_index, start_id in enumerate(START_IDS):
            total = int(split_config["start_counts"][start_id])
            route_quotas = _route_counts_for_start(start_id, total)
            candidate = base + start_index * 1000
            start_ordinal = 0
            for required_route, required_count in route_quotas.items():
                found = 0
                while found < required_count:
                    if candidate in all_seeds:
                        raise RuntimeError("Split seed collision")
                    scenario_index = start_index + 3 * start_ordinal
                    scenario = make_scenario(
                        execution, demos, geometry, candidate, scenario_index
                    )
                    _, selection = select_method_path(
                        execution, demos, scenario, METHOD
                    )
                    if selection["source_route"] == required_route:
                        records.append(
                            {
                                "split": split_name,
                                "seed": candidate,
                                "scenario_index": scenario_index,
                                "start_id": start_id,
                                "selected_route": required_route,
                                "source_episode": selection["source_episode"],
                                "source_confidence": selection["source_confidence"],
                                "start_xy_m": scenario.start_xy.tolist(),
                                "goal_xy_m": scenario.goal_xy.tolist(),
                                "obstacle_xy_m": scenario.obstacle_xy.tolist(),
                                "obstacle_half_size_m": scenario.obstacle_half_size.tolist(),
                            }
                        )
                        all_seeds.add(candidate)
                        found += 1
                        start_ordinal += 1
                    candidate += 1
                    if candidate >= base + start_index * 1000 + 1000:
                        raise RuntimeError(f"Could not fill {split_name} {start_id} route quotas")
        records.sort(key=lambda row: int(row["scenario_index"]))
        splits[split_name] = records
    seed_sets = {
        name: {int(row["seed"]) for row in records} for name, records in splits.items()
    }
    if any(
        seed_sets[first].intersection(seed_sets[second])
        for first, second in (
            ("training", "validation"),
            ("training", "testing"),
            ("validation", "testing"),
        )
    ):
        raise RuntimeError("Phase 5B split seeds are not disjoint")
    summary = {}
    for name, records in splits.items():
        summary[name] = {
            "scenarios": len(records),
            "start_counts": dict(Counter(str(row["start_id"]) for row in records)),
            "route_counts": dict(Counter(str(row["selected_route"]) for row in records)),
            "seed_sha256": hashlib.sha256(
                json.dumps([row["seed"] for row in records]).encode()
            ).hexdigest(),
        }
    return {
        "schema_version": 1,
        "fixed_before_rollouts": True,
        "split_seed_disjoint": True,
        "summary": summary,
        "splits": splits,
    }


def _teacher_factory(
    execution: Phase4BConfig,
    feature_rows: list[FloatArray] | None = None,
    action_rows: list[FloatArray] | None = None,
) -> Callable[[Scenario, dict[str, Any]], ActionProvider]:
    controller = execution.phase4a.raw["controller"]

    def factory(scenario: Scenario, selection: dict[str, Any]) -> ActionProvider:
        route = str(selection["source_route"])
        gripper_state = -1.0

        def provider(
            target: FloatArray,
            desired_gripper: float,
            eef: FloatArray,
            cube: FloatArray,
        ) -> FloatArray:
            nonlocal gripper_state
            action, _ = bounded_action(
                target - eef,
                float(controller["position_output_limit"]),
                desired_gripper,
            )
            if feature_rows is not None and action_rows is not None:
                feature_rows.append(
                    policy_features(
                        eef,
                        cube,
                        target,
                        scenario,
                        route,
                        gripper_state,
                        desired_gripper,
                    )
                )
                action_rows.append(action)
            gripper_state = desired_gripper
            return action

        return provider

    return factory


def collect_teacher_split(
    config: Phase5BConfig,
    execution: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
    specifications: list[dict[str, Any]],
) -> list[TeacherEpisode]:
    episodes: list[TeacherEpisode] = []
    for specification in specifications:
        features: list[FloatArray] = []
        actions: list[FloatArray] = []
        record, _ = _run_episode(
            config.phase5a,
            execution,
            demos,
            geometry,
            int(specification["seed"]),
            int(specification["scenario_index"]),
            _teacher_factory(execution, features, actions),
            method_name=METHOD,
        )
        if not record["phase5a_gate_success"]:
            raise RuntimeError(
                f"Unsafe teacher rollout in {specification['split']} seed {specification['seed']}"
            )
        episodes.append(
            TeacherEpisode(
                seed=int(specification["seed"]),
                scenario_index=int(specification["scenario_index"]),
                features=np.asarray(features, dtype=np.float64),
                actions=np.asarray(actions, dtype=np.float64),
                record=record,
            )
        )
    return episodes


def _policy_factory(
    policy: GRUPolicy,
    input_normalizer: Normalizer,
    output_normalizer: Normalizer,
) -> Callable[[Scenario, dict[str, Any]], ActionProvider]:
    def factory(scenario: Scenario, selection: dict[str, Any]) -> ActionProvider:
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
            prediction, hidden = policy.step(input_normalizer.transform(features), hidden)
            action = output_normalizer.inverse(prediction[0])
            gripper_state = desired_gripper
            return np.clip(action, -1.0, 1.0)

        return provider

    return factory


def _save_dataset(
    path: Path,
    training: list[TeacherEpisode],
    validation: list[TeacherEpisode],
) -> None:
    train_batch = pad_sequences(
        [episode.features for episode in training],
        [episode.actions for episode in training],
    )
    validation_batch = pad_sequences(
        [episode.features for episode in validation],
        [episode.actions for episode in validation],
    )
    np.savez_compressed(
        path,
        training_states_and_context=train_batch.inputs,
        training_teacher_actions=train_batch.targets,
        training_valid_mask=train_batch.mask,
        training_episode_lengths=train_batch.lengths,
        training_seeds=np.asarray([episode.seed for episode in training], dtype=np.int64),
        validation_states_and_context=validation_batch.inputs,
        validation_teacher_actions=validation_batch.targets,
        validation_valid_mask=validation_batch.mask,
        validation_episode_lengths=validation_batch.lengths,
        validation_seeds=np.asarray(
            [episode.seed for episode in validation], dtype=np.int64
        ),
    )


def _save_curve(
    csv_path: Path, plot_path: Path, history: list[dict[str, float | int]]
) -> None:
    fields = ["epoch", "train_loss", "validation_loss", "gradient_norm"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in history:
            writer.writerow({field: row[field] for field in fields})
    figure, axis = plt.subplots(figsize=(5.7, 3.7), constrained_layout=True)
    epochs = [int(row["epoch"]) for row in history]
    axis.plot(epochs, [float(row["train_loss"]) for row in history], label="training")
    axis.plot(
        epochs,
        [float(row["validation_loss"]) for row in history],
        label="episode-disjoint validation",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized action MSE")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(plot_path, dpi=160, metadata={"Software": "phone2panda"})
    plt.close(figure)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = np.concatenate(
        [np.asarray(record["_latencies_ms"], dtype=np.float64) for record in records]
    )
    return {
        "episodes": len(records),
        "safe_successes": sum(record["phase5a_gate_success"] for record in records),
        "object_collision_rollouts": sum(
            record["object_collision_steps"] > 0 for record in records
        ),
        "unintended_contact_rollouts": sum(
            record["unintended_contact"] for record in records
        ),
        "drop_rollouts": sum(record["drop"] for record in records),
        "placement_error_median_m": float(
            np.median([record["placement_error_m"] for record in records])
        ),
        "placement_error_p95_m": float(
            np.percentile([record["placement_error_m"] for record in records], 95)
        ),
        "minimum_clearance_median_m": float(
            np.median([record["minimum_obstacle_clearance_m"] for record in records])
        ),
        "minimum_clearance_worst_m": float(
            np.min([record["minimum_obstacle_clearance_m"] for record in records])
        ),
        "controller_latency_median_ms": float(np.median(latencies)),
        "controller_latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def _provenance(
    config: Phase5BConfig,
    split: str,
    episodes: list[TeacherEpisode],
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "seed": episode.seed,
            "scenario_index": episode.scenario_index,
            "source_episode": episode.record["source_episode"],
            "source_route": episode.record["source_route"],
            "source_confidence": episode.record["source_confidence"],
            "teacher_method": METHOD,
            "human_trajectory_provenance": (
                f"accepted human recording {episode.record['source_episode']} -> smoothed DMP "
                "-> scenario retargeting -> bounded Panda action"
            ),
            "phase4e_config_sha256": config.phase5a.phase4e.digest,
        }
        for episode in episodes
    ]


def run_phase5b(config: Phase5BConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    output.mkdir(parents=True, exist_ok=True)
    execution, dimensions = _execution_config(config.phase5a.phase4e)
    demos, geometry = build_library(execution)
    manifest = build_balanced_manifest(config, execution, demos, geometry)
    _write_json(output / "split_manifest.json", manifest)

    training = collect_teacher_split(
        config, execution, demos, geometry, manifest["splits"]["training"]
    )
    validation = collect_teacher_split(
        config, execution, demos, geometry, manifest["splits"]["validation"]
    )
    if len(training) != 120 or len(validation) != 20:
        raise RuntimeError("Phase 5B teacher split sizes changed")
    _save_dataset(output / "teacher_dataset.npz", training, validation)
    schema = {
        "schema_version": 1,
        "episode_boundaries": "per-split episode_lengths and valid_mask",
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "teacher_method": METHOD,
        "training_provenance": _provenance(config, "training", training),
        "validation_provenance": _provenance(config, "validation", validation),
        "evaluation_seeds_excluded": True,
        "split_manifest": "split_manifest.json",
    }
    _write_json(output / "dataset_schema.json", schema)

    train_features = np.concatenate([episode.features for episode in training])
    train_actions = np.concatenate([episode.actions for episode in training])
    input_normalizer = Normalizer.fit(train_features, minimum_scale=1e-5)
    output_normalizer = Normalizer.fit(train_actions, minimum_scale=1e-3)
    train_batch = pad_sequences(
        [input_normalizer.transform(episode.features) for episode in training],
        [output_normalizer.transform(episode.actions) for episode in training],
    )
    validation_batch = pad_sequences(
        [input_normalizer.transform(episode.features) for episode in validation],
        [output_normalizer.transform(episode.actions) for episode in validation],
    )
    policy = GRUPolicy(
        len(FEATURE_NAMES),
        int(config.raw["policy"]["hidden_size"]),
        len(ACTION_NAMES),
        int(config.raw["policy"]["seed"]),
    )
    training_config = config.raw["training"]
    started = time.perf_counter()
    history = policy.fit(
        train_batch,
        maximum_epochs=int(training_config["maximum_epochs"]),
        learning_rate=float(training_config["learning_rate"]),
        patience=int(training_config["patience"]),
        minimum_improvement=float(training_config["minimum_improvement"]),
        target_loss=float(training_config["target_validation_loss"]),
        gradient_clip_norm=float(training_config["gradient_clip_norm"]),
        validation_batch=validation_batch,
    )
    training_seconds = time.perf_counter() - started
    checkpoint = output / "gru_policy.npz"
    policy.save(
        checkpoint,
        input_normalizer,
        output_normalizer,
        {
            "schema_version": 1,
            "config_sha256": config.digest,
            "feature_names": FEATURE_NAMES,
            "action_names": ACTION_NAMES,
            "split_seed_hashes": {
                name: details["seed_sha256"]
                for name, details in manifest["summary"].items()
            },
            "teacher_method": METHOD,
            "phase4e_config_sha256": config.phase5a.phase4e.digest,
        },
    )
    _save_curve(output / "training_curve.csv", output / "training_curve.png", history)

    testing = manifest["splits"]["testing"]
    dmp_records = [
        _run_episode(
            config.phase5a,
            execution,
            demos,
            geometry,
            int(specification["seed"]),
            int(specification["scenario_index"]),
            _teacher_factory(execution),
            method_name=METHOD,
        )[0]
        for specification in testing
    ]
    gru_factory = _policy_factory(policy, input_normalizer, output_normalizer)
    gru_records = [
        _run_episode(
            config.phase5a,
            execution,
            demos,
            geometry,
            int(specification["seed"]),
            int(specification["scenario_index"]),
            gru_factory,
            method_name="gru_policy",
        )[0]
        for specification in testing
    ]
    if len(dmp_records) != 50 or len(gru_records) != 50:
        raise RuntimeError("Phase 5B requires exactly 50 held-out rollouts per method")
    dmp_aggregate = _aggregate(dmp_records)
    gru_aggregate = _aggregate(gru_records)
    passed = gru_aggregate["safe_successes"] >= int(
        config.raw["gate"]["minimum_safe_successes"]
    )
    media: list[dict[str, Any]] = []
    success = next(
        (record for record in gru_records if record["phase5a_gate_success"]), None
    )
    failure = next(
        (record for record in gru_records if not record["phase5a_gate_success"]), None
    )
    for label, selected in (("success", success), ("failure", failure)):
        if selected is None:
            continue
        seed = int(selected["scenario"]["seed"])
        specification = next(row for row in testing if int(row["seed"]) == seed)
        replay, frames = _run_episode(
            config.phase5a,
            execution,
            demos,
            geometry,
            seed,
            int(specification["scenario_index"]),
            gru_factory,
            render=True,
            method_name="gru_policy",
        )
        if bool(replay["phase5a_gate_success"]) != bool(
            selected["phase5a_gate_success"]
        ):
            raise RuntimeError(f"Representative {label} did not reproduce")
        filename = f"representative_{label}.mp4"
        _write_video(output / filename, execution, frames)
        media.append({"kind": label, "path": filename, "seed": seed})
    records = [
        {"method": "dmp_route_confidence", **_public_record(record)}
        for record in dmp_records
    ] + [{"method": "gru_policy", **_public_record(record)} for record in gru_records]
    _write_jsonl(output / "evaluation_rollouts.jsonl", records)
    report = {
        "schema_version": 1,
        "phase": "5B",
        "passed": passed,
        "config_sha256": config.digest,
        "environment": dimensions,
        "splits": manifest["summary"],
        "policy": {
            "type": "normalized_gru",
            "hidden_size": policy.hidden_dim,
            "parameter_count": policy.parameter_count,
            "checkpoint_bytes": checkpoint.stat().st_size,
        },
        "training": {
            "seconds": training_seconds,
            "epochs": len(history),
            "maximum_epochs": int(training_config["maximum_epochs"]),
            "best_validation_loss": min(
                float(row["validation_loss"]) for row in history
            ),
            "final_validation_loss": float(history[-1]["validation_loss"]),
        },
        "held_out_evaluation": {
            "seeds_fixed_before_training": True,
            "scenarios": 50,
            "gate_minimum_safe_successes": int(
                config.raw["gate"]["minimum_safe_successes"]
            ),
            "dmp_route_confidence": dmp_aggregate,
            "gru_policy": gru_aggregate,
        },
        "representative_artifacts": media,
    }
    _write_json(output / "evaluation.json", report)
    _write_json(
        output / "run_config.json",
        {"config_sha256": config.digest, "configuration": config.raw},
    )
    return report
