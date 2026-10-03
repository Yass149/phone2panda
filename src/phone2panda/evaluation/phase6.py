from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
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
    _load_raw_path,
    _make_environment,
    _resample,
    _write_json,
    _write_jsonl,
    build_library,
    execute_rollout,
    select_method_path,
)
from phone2panda.evaluation.phase4c import ContactAccumulator
from phone2panda.evaluation.phase4e import _execution_config
from phone2panda.evaluation.phase5a import ActionProvider, _complete_record
from phone2panda.evaluation.phase5b import (
    Phase5BConfig,
    _aggregate,
    _policy_factory,
    load_phase5b_config,
)
from phone2panda.pilot_validation.geometry import homography_from_corners, transform_points
from phone2panda.policy.gru import GRUPolicy, Normalizer
from phone2panda.retarget.mapping import WorkspaceMap
from phone2panda.trajectories.dmp import fit_dmp, rollout_dmp
from phone2panda.trajectories.processing import load_processed_path, smooth_and_resample

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

FloatArray = NDArray[np.float64]
BASELINE_ALIASES = (
    "homography_calibrated",
    "smoothing_enabled",
    "filtering_enabled",
    "human_derived_path",
)


@dataclass(frozen=True)
class Phase6Config:
    path: Path
    root: Path
    raw: dict[str, Any]
    phase5b: Phase5BConfig

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def project_path(self, key: str) -> Path:
        return self.root / str(self.raw[key])


def load_phase6_config(path: Path) -> Phase6Config:
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = resolved.parent.parent
    phase5b = load_phase5b_config(root / str(raw["phase5b_config"]))
    if [int(value) for value in raw["demonstration_counts"]] != [5, 15, 30]:
        raise ValueError("Phase 6 demonstration counts must be exactly 5, 15, and 30")
    return Phase6Config(resolved, root, raw, phase5b)


def select_balanced_demonstration_subsets(
    config: Phase6Config,
) -> dict[str, list[str]]:
    with config.phase5b.phase5a.phase4e.phase4b.project_path("metadata_csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        cells[(str(row["start_id"]), str(row["route"]))].append(str(row["episode_id"]))
    order = (
        ("s1", "left"),
        ("s2", "right"),
        ("s3", "left"),
        ("s1", "right"),
        ("s2", "left"),
        ("s3", "right"),
    )
    ordered: list[str] = []
    maximum = max(len(cells[cell]) for cell in order)
    for position in range(maximum):
        for cell in order:
            if position < len(cells[cell]):
                ordered.append(sorted(cells[cell])[position])
    subsets = {
        str(count): ordered[:count] for count in config.raw["demonstration_counts"]
    }
    if any(len(set(values)) != int(count) for count, values in subsets.items()):
        raise RuntimeError("Demonstration subset selection produced duplicate episodes")
    return subsets


def _naive_pixel_scale(
    points: FloatArray, episode: dict[str, Any]
) -> FloatArray:
    corners = {
        key: tuple(float(value) for value in episode["calibration"]["opening_markers"][key])
        for key in ("tl", "tr", "bl", "br")
    }
    inverse = np.linalg.inv(homography_from_corners(corners))
    pixels = transform_points(
        [(float(point[0]), float(point[1])) for point in points], inverse
    )
    width = float(episode["video"]["width"])
    height = float(episode["video"]["height"])
    return np.clip(pixels / np.asarray([width, height]), 0.0, 1.0)


def build_ablation_library(
    config: Phase6Config,
    execution: Phase4BConfig,
    episode_ids: list[str],
    *,
    mapping_mode: str,
    smoothing_enabled: bool,
) -> list[Demonstration]:
    phase4b = config.phase5b.phase5a.phase4e.phase4b
    report = json.loads(phase4b.project_path("dataset_report").read_text(encoding="utf-8"))
    report_by_id = {str(item["episode_id"]): item for item in report["episodes"]}
    with phase4b.project_path("metadata_csv").open(newline="", encoding="utf-8") as handle:
        metadata = {
            str(row["episode_id"]): row for row in csv.DictReader(handle)
        }
    workspace = execution.phase4a.raw["workspace"]
    trajectory = execution.phase4a.raw["trajectory"]
    mapping = WorkspaceMap(
        tuple(float(value) for value in workspace["robot_x_range"]),
        tuple(float(value) for value in workspace["robot_y_range"]),
    )
    samples = int(trajectory["transport_samples"])
    demonstrations: list[Demonstration] = []
    for episode_id in episode_ids:
        episode = report_by_id[episode_id]
        row = metadata[episode_id]
        trajectory_path = config.root / str(episode["processed_trajectory"])
        raw = _resample(_load_raw_path(trajectory_path), samples)
        if smoothing_enabled:
            source = smooth_and_resample(
                load_processed_path(trajectory_path),
                samples,
                int(trajectory["savgol_window"]),
                int(trajectory["savgol_order"]),
            )
        else:
            source = raw
        dmp = rollout_dmp(
            fit_dmp(source, int(trajectory["dmp_basis_functions"])), samples
        )
        if mapping_mode == "naive_pixel":
            raw = _naive_pixel_scale(raw, episode)
            dmp = _naive_pixel_scale(dmp, episode)
        elif mapping_mode != "calibrated_homography":
            raise ValueError(f"Unsupported mapping mode: {mapping_mode}")
        demonstrations.append(
            Demonstration(
                episode_id=episode_id,
                start_id=str(row["start_id"]),
                route=str(row["route"]),
                split=str(row["split"]),
                coverage=float(episode["tracking"]["detected_coverage"]),
                confidence=float(episode["tracking"]["mean_confidence"]),
                raw_robot=mapping.map_points(raw),
                dmp_robot=mapping.map_points(dmp),
            )
        )
    return demonstrations


def _fixed_scenarios(config: Phase6Config) -> list[Scenario]:
    manifest_path = config.phase5b.project_path("output_dir") / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["splits"]["testing"]
    if len(records) != 50:
        raise RuntimeError("Phase 6 requires the fixed 50-scenario Phase 5B test manifest")
    return [
        Scenario(
            index=int(row["scenario_index"]),
            seed=int(row["seed"]),
            start_id=str(row["start_id"]),
            start_xy=np.asarray(row["start_xy_m"], dtype=np.float64),
            goal_xy=np.asarray(row["goal_xy_m"], dtype=np.float64),
            obstacle_xy=np.asarray(row["obstacle_xy_m"], dtype=np.float64),
            obstacle_half_size=np.asarray(row["obstacle_half_size_m"], dtype=np.float64),
        )
        for row in records
    ]


def _run_path(
    config: Phase6Config,
    execution: Phase4BConfig,
    scenario: Scenario,
    path: FloatArray,
    selection: dict[str, Any],
    condition: str,
    action_provider: ActionProvider | None = None,
) -> dict[str, Any]:
    np.random.seed(scenario.seed)
    contacts = ContactAccumulator()
    rollout_id = f"phase6:{condition}:{scenario.seed}"
    contacts.begin(rollout_id)
    env = _make_environment(execution, scenario, render=False)
    try:
        record, _ = execute_rollout(
            execution,
            env,
            scenario,
            condition,
            path,
            selection,
            contact_callback=contacts.observe,
            monitor_transport_height_drop=False,
            action_provider=action_provider,
        )
    finally:
        env.close()
    return _complete_record(config.phase5b.phase5a, record, contacts, rollout_id)


def _run_condition(
    config: Phase6Config,
    execution: Phase4BConfig,
    scenarios: list[Scenario],
    demonstrations: list[Demonstration],
    condition: str,
    method: str,
) -> list[dict[str, Any]]:
    records = []
    samples = int(execution.phase4a.raw["trajectory"]["transport_samples"])
    for scenario in scenarios:
        if method == "straight":
            path = np.linspace(scenario.start_xy, scenario.goal_xy, samples)
            selection = {
                "source_episode": None,
                "source_route": None,
                "source_confidence": None,
                "eligible_demonstrations": 0,
            }
        else:
            path, selection = select_method_path(
                execution, demonstrations, scenario, method
            )
        records.append(
            _run_path(config, execution, scenario, path, selection, condition)
        )
    if len(records) != 50:
        raise RuntimeError(f"Ablation {condition} did not produce exactly 50 episodes")
    return records


def _aggregate_public(records: list[dict[str, Any]]) -> dict[str, Any]:
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
        "controller_latency_median_ms": float(
            np.median([record["controller_latency_median_ms"] for record in records])
        ),
        "controller_latency_p95_ms": float(
            np.percentile([record["controller_latency_p95_ms"] for record in records], 95)
        ),
    }


def _compact_record(
    condition: str, record: dict[str, Any], reused_from: str | None = None
) -> dict[str, Any]:
    contacts = record["contacts"]
    payload = {
        "condition": condition,
        "scenario": record["scenario"],
        "source_episode": record.get("source_episode"),
        "source_route": record.get("source_route"),
        "source_confidence": record.get("source_confidence"),
        "safe_success": record["phase5a_gate_success"],
        "object_collision_steps": record["object_collision_steps"],
        "unintended_contact": record["unintended_contact"],
        "robot_obstacle_contact_events": contacts["robot_obstacle"]["contact_events"],
        "robot_table_contact_events": contacts["robot_table"]["contact_events"],
        "self_collision_events": contacts["self_collision"]["contact_events"],
        "drop": record["drop"],
        "placement_error_m": record["placement_error_m"],
        "minimum_obstacle_clearance_m": record["minimum_obstacle_clearance_m"],
        "controller_latency_median_ms": record["controller_latency_median_ms"],
        "controller_latency_p95_ms": record["controller_latency_p95_ms"],
    }
    if reused_from is not None:
        payload["reused_from"] = reused_from
    return payload


def _phase5b_baseline(config: Phase6Config) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = config.phase5b.project_path("output_dir")
    rows = [
        json.loads(line)
        for line in (output / "evaluation_rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    baseline = [row for row in rows if row["method"] == "dmp_route_confidence"]
    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    if len(baseline) != 50:
        raise RuntimeError("Expected 50 reusable Phase 5B DMP teacher records")
    return baseline, report["held_out_evaluation"]["dmp_route_confidence"]


def _ablation_markdown(summary: dict[str, Any]) -> str:
    header = (
        "| Factor | Level | Safe success | Object collisions | Unintended contacts | "
        "Drops | Placement median | Clearance median |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for row in summary["table"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['factor']} | {row['level']} | {metrics['safe_successes']}/50 | "
            f"{metrics['object_collision_rollouts']} | "
            f"{metrics['unintended_contact_rollouts']} | {metrics['drop_rollouts']} | "
            f"{1000 * metrics['placement_error_median_m']:.2f} mm | "
            f"{1000 * metrics['minimum_clearance_median_m']:.2f} mm |"
        )
    return header + "\n".join(lines) + "\n"


def _plot_ablation(path: Path, table: list[dict[str, Any]]) -> None:
    labels = [f"{row['factor']}: {row['level']}" for row in table]
    success = [row["metrics"]["safe_successes"] * 2.0 for row in table]
    clearance = [
        1000.0 * row["metrics"]["minimum_clearance_median_m"] for row in table
    ]
    x = np.arange(len(table))
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), constrained_layout=True)
    axes[0].bar(x, success, color="#4c78a8")
    axes[0].axhline(90.0, color="#b22222", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Safe success (%)")
    axes[0].set_ylim(0, 105)
    axes[1].bar(x, clearance, color="#54a24b")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Median clearance (mm)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=160, metadata={"Software": "phone2panda"})
    plt.close(figure)


def _float32_policy(
    source_checkpoint: Path, destination: Path, config: Phase6Config
) -> tuple[
    GRUPolicy,
    Normalizer,
    Normalizer,
    GRUPolicy,
    Normalizer,
    Normalizer,
]:
    policy64, input64, output64, metadata = GRUPolicy.load(source_checkpoint)
    policy32, input32, output32, _ = GRUPolicy.load(source_checkpoint, dtype=np.float32)
    metadata = {
        **metadata,
        "precision": "float32",
        "phase6_config_sha256": config.digest,
        "source_checkpoint": "../phase5b/gru_policy.npz",
    }
    policy32.save(destination, input32, output32, metadata)
    return policy64, input64, output64, policy32, input32, output32


def _prediction_difference(
    dataset_path: Path,
    policy64: GRUPolicy,
    input64: Normalizer,
    output64: Normalizer,
    policy32: GRUPolicy,
    input32: Normalizer,
    output32: Normalizer,
) -> dict[str, float | int]:
    differences: list[FloatArray] = []
    compared = 0
    with np.load(dataset_path, allow_pickle=False) as data:
        features = data["validation_states_and_context"]
        lengths = data["validation_episode_lengths"]
        for episode, length in zip(features, lengths, strict=True):
            hidden64 = policy64.initial_state()
            hidden32 = policy32.initial_state()
            for row in episode[: int(length)]:
                prediction64, hidden64 = policy64.step(input64.transform(row), hidden64)
                prediction32, hidden32 = policy32.step(input32.transform(row), hidden32)
                action64 = output64.inverse(prediction64[0])
                action32 = output32.inverse(prediction32[0]).astype(np.float64)
                differences.append(np.abs(action64 - action32))
                compared += 1
    values = np.asarray(differences, dtype=np.float64)
    return {
        "compared_actions": compared,
        "maximum_absolute_output_difference": float(np.max(values)),
        "mean_absolute_output_difference": float(np.mean(values)),
    }


def _benchmark_policy(
    policy: GRUPolicy,
    input_normalizer: Normalizer,
    output_normalizer: Normalizer,
    features: FloatArray,
    *,
    warmup: int,
    trials: int,
    reset_interval: int,
) -> dict[str, Any]:
    hidden = policy.initial_state()
    for index in range(warmup):
        row = features[index % len(features)]
        prediction, hidden = policy.step(input_normalizer.transform(row), hidden)
        output_normalizer.inverse(prediction[0])
        if (index + 1) % reset_interval == 0:
            hidden = policy.initial_state()
    latencies = np.empty(trials, dtype=np.float64)
    hidden = policy.initial_state()
    for index in range(trials):
        row = features[index % len(features)]
        started = time.perf_counter_ns()
        prediction, hidden = policy.step(input_normalizer.transform(row), hidden)
        output_normalizer.inverse(prediction[0])
        latencies[index] = (time.perf_counter_ns() - started) / 1_000_000.0
        if (index + 1) % reset_interval == 0:
            hidden = policy.initial_state()
    return {
        "warmup_trials": warmup,
        "measured_trials": trials,
        "input_shape": [1, policy.input_dim],
        "hidden_shape": [1, policy.hidden_dim],
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }


def _hardware_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "logical_cpu_count": os.cpu_count(),
        "python_implementation": sys.implementation.name,
    }


def run_phase6(config: Phase6Config) -> dict[str, Any]:
    output = config.project_path("output_dir")
    output.mkdir(parents=True, exist_ok=True)
    execution, dimensions = _execution_config(config.phase5b.phase5a.phase4e)
    all_demos, _ = build_library(execution)
    all_ids = [demo.episode_id for demo in all_demos]
    subsets = select_balanced_demonstration_subsets(config)
    subset_report = {
        "schema_version": 1,
        "selected_before_ablation_rollouts": True,
        "selection_order": "round_robin_start_route_cells_then_episode_id",
        "subsets": subsets,
    }
    _write_json(output / "demonstration_subsets.json", subset_report)
    scenarios = _fixed_scenarios(config)
    baseline_records, baseline_metrics = _phase5b_baseline(config)
    if [record["scenario"]["seed"] for record in baseline_records] != [
        scenario.seed for scenario in scenarios
    ]:
        raise RuntimeError("Reusable Phase 5B baseline does not match fixed scenarios")

    libraries = {
        f"demo_{count}": build_ablation_library(
            config,
            execution,
            subsets[str(count)],
            mapping_mode="calibrated_homography",
            smoothing_enabled=True,
        )
        for count in (5, 15, 30)
    }
    libraries["naive_pixel_scaling"] = build_ablation_library(
        config,
        execution,
        all_ids,
        mapping_mode="naive_pixel",
        smoothing_enabled=True,
    )
    libraries["smoothing_disabled"] = build_ablation_library(
        config,
        execution,
        all_ids,
        mapping_mode="calibrated_homography",
        smoothing_enabled=False,
    )
    libraries["filtering_disabled"] = all_demos
    libraries["straight_line"] = all_demos
    methods = {
        "demo_5": "dmp_route_confidence",
        "demo_15": "dmp_route_confidence",
        "demo_30": "dmp_route_confidence",
        "naive_pixel_scaling": "dmp_route_confidence",
        "smoothing_disabled": "dmp_route_confidence",
        "filtering_disabled": "dmp",
        "straight_line": "straight",
    }
    unique_records = {
        condition: _run_condition(
            config,
            execution,
            scenarios,
            libraries[condition],
            condition,
            method,
        )
        for condition, method in methods.items()
    }
    metrics = {
        condition: _aggregate(records) for condition, records in unique_records.items()
    }
    for alias in BASELINE_ALIASES:
        metrics[alias] = baseline_metrics

    rows: list[dict[str, Any]] = []
    table_specification = (
        ("demonstrations", "5", "demo_5"),
        ("demonstrations", "15", "demo_15"),
        ("demonstrations", "30", "demo_30"),
        ("homography", "calibrated", "homography_calibrated"),
        ("homography", "naive pixel scaling", "naive_pixel_scaling"),
        ("smoothing", "enabled", "smoothing_enabled"),
        ("smoothing", "disabled", "smoothing_disabled"),
        ("confidence/route filtering", "enabled", "filtering_enabled"),
        ("confidence/route filtering", "disabled", "filtering_disabled"),
        ("path", "human-derived", "human_derived_path"),
        ("path", "straight line", "straight_line"),
    )
    for factor, level, condition in table_specification:
        rows.append(
            {
                "factor": factor,
                "level": level,
                "condition": condition,
                "episodes": 50,
                "reused_phase5b_baseline": condition in BASELINE_ALIASES,
                "metrics": metrics[condition],
            }
        )
    raw_rows: list[dict[str, Any]] = []
    for condition, records in unique_records.items():
        raw_rows.extend(_compact_record(condition, record) for record in records)
    for alias in BASELINE_ALIASES:
        raw_rows.extend(
            _compact_record(
                alias,
                record,
                reused_from="results/phase5b/evaluation_rollouts.jsonl",
            )
            for record in baseline_records
        )
    if any(
        sum(row["condition"] == condition for row in raw_rows) != 50
        for _, _, condition in table_specification
    ):
        raise RuntimeError("Every ablation condition must have exactly 50 records")
    _write_jsonl(output / "ablation_rollouts.jsonl", raw_rows)
    ablation_report = {
        "schema_version": 1,
        "phase": "6_ablations",
        "environment": dimensions,
        "fixed_scenario_seed_sha256": json.loads(
            (config.phase5b.project_path("output_dir") / "split_manifest.json").read_text(
                encoding="utf-8"
            )
        )["summary"]["testing"]["seed_sha256"],
        "unique_simulated_conditions": list(unique_records),
        "reused_baseline_conditions": list(BASELINE_ALIASES),
        "table": rows,
    }
    _write_json(output / "ablation_summary.json", ablation_report)
    (output / "ablation_table.md").write_text(
        _ablation_markdown(ablation_report), encoding="utf-8"
    )
    _plot_ablation(output / "ablation_plot.png", rows)

    phase5b_output = config.phase5b.project_path("output_dir")
    source_checkpoint = phase5b_output / "gru_policy.npz"
    checkpoint32 = output / "gru_policy_float32.npz"
    policy64, input64, output64, policy32, input32, output32 = _float32_policy(
        source_checkpoint, checkpoint32, config
    )
    dataset_path = phase5b_output / "teacher_dataset.npz"
    difference = _prediction_difference(
        dataset_path, policy64, input64, output64, policy32, input32, output32
    )
    benchmark_config = config.raw["precision_benchmark"]
    with np.load(dataset_path, allow_pickle=False) as dataset:
        benchmark_features = dataset["validation_states_and_context"].reshape(
            -1, policy64.input_dim
        )
    benchmark64 = _benchmark_policy(
        policy64,
        input64,
        output64,
        benchmark_features,
        warmup=int(benchmark_config["warmup_trials"]),
        trials=int(benchmark_config["measured_trials"]),
        reset_interval=int(benchmark_config["reset_interval"]),
    )
    benchmark32 = _benchmark_policy(
        policy32,
        input32,
        output32,
        benchmark_features,
        warmup=int(benchmark_config["warmup_trials"]),
        trials=int(benchmark_config["measured_trials"]),
        reset_interval=int(benchmark_config["reset_interval"]),
    )
    predictions_differ = difference["maximum_absolute_output_difference"] > 0.0
    float32_records: list[dict[str, Any]] = []
    if predictions_differ and bool(config.raw["float32"]["rerun_if_predictions_differ"]):
        factory32 = _policy_factory(policy32, input32, output32)
        for scenario in scenarios:
            path, selection = select_method_path(
                execution,
                all_demos,
                scenario,
                "dmp_route_confidence",
            )
            float32_records.append(
                _run_path(
                    config,
                    execution,
                    scenario,
                    path,
                    selection,
                    "gru_float32",
                    action_provider=factory32(scenario, selection),
                )
            )
        if len(float32_records) != 50:
            raise RuntimeError("Float32 rerun must contain exactly 50 episodes")
        _write_jsonl(
            output / "float32_rollouts.jsonl",
            [_compact_record("precision_float32", record) for record in float32_records],
        )
    precision_report = {
        "schema_version": 1,
        "phase": "6_precision",
        "hardware": _hardware_metadata(),
        "benchmark_protocol": {
            "includes": "normalization, one GRU step, output denormalization",
            "warmup_trials": int(benchmark_config["warmup_trials"]),
            "measured_trials": int(benchmark_config["measured_trials"]),
            "fixed_input_shape": [1, policy64.input_dim],
            "fixed_hidden_shape": [1, policy64.hidden_dim],
        },
        "float64": {
            **benchmark64,
            "checkpoint_bytes": source_checkpoint.stat().st_size,
            "held_out_closed_loop": json.loads(
                (phase5b_output / "evaluation.json").read_text(encoding="utf-8")
            )["held_out_evaluation"]["gru_policy"],
            "closed_loop_reused": True,
        },
        "float32": {
            **benchmark32,
            "checkpoint_bytes": checkpoint32.stat().st_size,
            "held_out_closed_loop": (
                _aggregate(float32_records) if float32_records else None
            ),
            "closed_loop_rerun": bool(float32_records),
        },
        "numerical_difference": difference,
        "predictions_differ": predictions_differ,
    }
    _write_json(output / "precision_benchmark.json", precision_report)
    summary = {
        "schema_version": 1,
        "phase": "6",
        "config_sha256": config.digest,
        "ablation_summary": "ablation_summary.json",
        "precision_benchmark": "precision_benchmark.json",
        "videos_generated": 0,
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "run_config.json",
        {"config_sha256": config.digest, "configuration": config.raw},
    )
    return summary
