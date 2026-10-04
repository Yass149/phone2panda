from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

os.environ.setdefault("MUJOCO_GL", "cgl" if platform.system() == "Darwin" else "egl")

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np
import robosuite
import yaml
from numpy.typing import NDArray
from robosuite.controllers import load_composite_controller_config

from phone2panda.retarget.mapping import WorkspaceMap, bounded_action
from phone2panda.sim.phase4a import (
    HumanPathPickPlace,
    Phase4AConfig,
    load_phase4a_config,
    reset_signature,
)
from phone2panda.trajectories.dmp import fit_dmp, rollout_dmp
from phone2panda.trajectories.processing import load_processed_path, smooth_and_resample
from phone2panda.trajectories.public_data import resolve_processed_path

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

FloatArray = NDArray[np.float64]
METHODS = ("straight", "raw_replay", "dmp", "dmp_route_confidence")


@dataclass(frozen=True)
class Phase4BConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    phase4a: Phase4AConfig

    def project_path(self, key: str) -> Path:
        return self.root / str(self.raw[key])

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Demonstration:
    episode_id: str
    start_id: str
    route: str
    split: str
    coverage: float
    confidence: float
    raw_robot: FloatArray
    dmp_robot: FloatArray


@dataclass(frozen=True)
class Scenario:
    index: int
    seed: int
    start_id: str
    start_xy: FloatArray
    goal_xy: FloatArray
    obstacle_xy: FloatArray
    obstacle_half_size: FloatArray

    def serialise(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "start_id": self.start_id,
            "start_xy_m": self.start_xy.tolist(),
            "goal_xy_m": self.goal_xy.tolist(),
            "obstacle_xy_m": self.obstacle_xy.tolist(),
            "obstacle_half_size_m": self.obstacle_half_size.tolist(),
        }


def load_phase4b_config(path: Path) -> Phase4BConfig:
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = resolved.parent.parent
    phase4a = load_phase4a_config(root / str(raw["phase4a_config"]))
    methods = tuple(str(value) for value in raw["methods"])
    if methods != METHODS:
        raise ValueError(f"Phase 4B requires methods in fixed order: {METHODS}")
    if len(raw["preflight"]["seeds"]) != 5:
        raise ValueError("Phase 4B preflight requires exactly five seeds")
    if int(raw["evaluation"]["episode_count"]) != 50:
        raise ValueError("Phase 4B evaluation requires exactly 50 episodes per method")
    return Phase4BConfig(path=resolved, root=root, raw=raw, phase4a=phase4a)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _resample(points: FloatArray, samples: int) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    keep = np.concatenate([[True], np.diff(distance) > 1e-12])
    values = values[keep]
    distance = distance[keep]
    if len(values) < 2 or distance[-1] <= 0.0:
        raise ValueError("Trajectory has zero length")
    sample_distance = np.linspace(0.0, distance[-1], samples)
    output = np.column_stack(
        [np.interp(sample_distance, distance, values[:, axis]) for axis in range(2)]
    )
    output[0] = values[0]
    output[-1] = values[-1]
    return output


def _load_raw_path(path: Path) -> FloatArray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["valid"] == "1"]
    points = np.asarray(
        [[float(row["object_x"]), float(row["object_y"])] for row in rows],
        dtype=np.float64,
    )
    if len(points) < 5:
        raise ValueError(f"Not enough valid raw points in {path}")
    return points


def build_library(
    config: Phase4BConfig, *, public_only: bool = False
) -> tuple[list[Demonstration], dict[str, Any]]:
    report = json.loads(config.project_path("dataset_report").read_text(encoding="utf-8"))
    report_by_id = {str(item["episode_id"]): item for item in report["episodes"]}
    with config.project_path("metadata_csv").open(newline="", encoding="utf-8") as handle:
        metadata = list(csv.DictReader(handle))
    workspace = config.phase4a.raw["workspace"]
    trajectory = config.phase4a.raw["trajectory"]
    mapping = WorkspaceMap(
        tuple(float(value) for value in workspace["robot_x_range"]),
        tuple(float(value) for value in workspace["robot_y_range"]),
    )
    samples = int(trajectory["transport_samples"])
    required_split = str(config.raw["demonstrations"]["split"])
    demos: list[Demonstration] = []
    obstacle_centers: list[FloatArray] = []
    obstacle_half_sizes: list[FloatArray] = []
    for row in metadata:
        episode_id = str(row["episode_id"])
        episode = report_by_id[episode_id]
        if not episode["accepted"] or (required_split != "all" and row["split"] != required_split):
            continue
        trajectory_path = resolve_processed_path(
            config.root, str(episode["processed_trajectory"]), public_only=public_only
        )
        raw = _resample(_load_raw_path(trajectory_path), samples)
        smoothed = smooth_and_resample(
            load_processed_path(trajectory_path),
            samples,
            int(trajectory["savgol_window"]),
            int(trajectory["savgol_order"]),
        )
        dmp = rollout_dmp(
            fit_dmp(smoothed, int(trajectory["dmp_basis_functions"])), samples
        )
        demos.append(
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
        center, half_size = mapping.map_rect(
            tuple(float(value) for value in episode["task"]["obstacle_rect"])
        )
        obstacle_centers.append(center)
        obstacle_half_sizes.append(half_size)
    if not demos:
        raise ValueError("No accepted demonstrations match the configured split")
    geometry = {
        "obstacle_xy": np.median(np.asarray(obstacle_centers), axis=0),
        "obstacle_half_size": np.median(np.asarray(obstacle_half_sizes), axis=0),
    }
    return demos, geometry


def similarity_retarget(path: FloatArray, start: FloatArray, goal: FloatArray) -> FloatArray:
    values = np.asarray(path, dtype=np.float64)
    source = values[-1] - values[0]
    target = np.asarray(goal, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-9 or target_norm <= 1e-9:
        raise ValueError("Cannot retarget a zero-length trajectory")
    angle = math.atan2(target[1], target[0]) - math.atan2(source[1], source[0])
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    output = np.asarray(start) + (values - values[0]) @ rotation.T * (target_norm / source_norm)
    output[0] = start
    output[-1] = goal
    return output


def signed_rectangle_clearance(
    points: FloatArray,
    center: FloatArray,
    half_size: FloatArray,
    object_radius: float,
) -> FloatArray:
    delta = np.abs(np.asarray(points, dtype=np.float64) - center) - half_size
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    inside = np.minimum(np.maximum(delta[:, 0], delta[:, 1]), 0.0)
    return outside + inside - float(object_radius)


def make_scenario(
    config: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
    seed: int,
    index: int,
) -> Scenario:
    rng = np.random.default_rng(seed)
    start_id = ("s1", "s2", "s3")[index % 3]
    starts = np.asarray([demo.raw_robot[0] for demo in demos if demo.start_id == start_id])
    if not len(starts):
        raise ValueError(f"No demonstration starts at {start_id}")
    goals = np.asarray([demo.raw_robot[-1] for demo in demos])
    scenario_cfg = config.raw["scenario"]
    return Scenario(
        index=index,
        seed=seed,
        start_id=start_id,
        start_xy=np.mean(starts, axis=0)
        + rng.normal(0.0, float(scenario_cfg["start_jitter_m"]), size=2),
        goal_xy=np.mean(goals, axis=0)
        + rng.normal(0.0, float(scenario_cfg["goal_jitter_m"]), size=2),
        obstacle_xy=np.asarray(geometry["obstacle_xy"])
        + rng.normal(0.0, float(scenario_cfg["obstacle_jitter_m"]), size=2),
        obstacle_half_size=np.asarray(geometry["obstacle_half_size"]),
    )


def _endpoint_score(path: FloatArray, scenario: Scenario) -> float:
    return float(
        np.linalg.norm(path[0] - scenario.start_xy)
        + np.linalg.norm(path[-1] - scenario.goal_xy)
    )


def select_method_path(
    config: Phase4BConfig,
    demos: list[Demonstration],
    scenario: Scenario,
    method: str,
) -> tuple[FloatArray, dict[str, Any]]:
    samples = int(config.phase4a.raw["trajectory"]["transport_samples"])
    if method == "straight":
        return np.linspace(scenario.start_xy, scenario.goal_xy, samples), {
            "source_episode": None,
            "source_route": None,
            "source_confidence": None,
            "eligible_demonstrations": 0,
        }
    attribute = "raw_robot" if method == "raw_replay" else "dmp_robot"
    if method != "dmp_route_confidence":
        selected = min(demos, key=lambda item: _endpoint_score(getattr(item, attribute), scenario))
        path = similarity_retarget(
            getattr(selected, attribute), scenario.start_xy, scenario.goal_xy
        )
        return path, {
            "source_episode": selected.episode_id,
            "source_route": selected.route,
            "source_confidence": selected.confidence,
            "eligible_demonstrations": len(demos),
        }

    demo_cfg = config.raw["demonstrations"]
    eligible = [
        demo
        for demo in demos
        if demo.coverage >= float(demo_cfg["minimum_tracking_coverage"])
        and demo.confidence >= float(demo_cfg["minimum_mean_confidence"])
    ]
    candidates: list[tuple[float, Demonstration, FloatArray]] = []
    for route in ("left", "right"):
        route_demos = [demo for demo in eligible if demo.route == route]
        if not route_demos:
            continue
        selected = min(
            route_demos,
            key=lambda item: _endpoint_score(item.dmp_robot, scenario),
        )
        path = similarity_retarget(selected.dmp_robot, scenario.start_xy, scenario.goal_xy)
        clearance = float(
            np.min(
                signed_rectangle_clearance(
                    path,
                    scenario.obstacle_xy,
                    scenario.obstacle_half_size,
                    float(config.raw["scenario"]["object_radius_m"]),
                )
            )
        )
        candidates.append((clearance, selected, path))
    if not candidates:
        raise ValueError("Confidence filtering removed all route candidates")
    clearance, selected, path = max(candidates, key=lambda item: (item[0], item[1].confidence))
    return path, {
        "source_episode": selected.episode_id,
        "source_route": selected.route,
        "source_confidence": selected.confidence,
        "eligible_demonstrations": len(eligible),
        "predicted_minimum_clearance_m": clearance,
    }


def _make_environment(
    config: Phase4BConfig, scenario: Scenario, render: bool
) -> HumanPathPickPlace:
    phase4a = config.phase4a.raw
    workspace = phase4a["workspace"]
    controller = phase4a["controller"]
    video = phase4a["video"]
    return HumanPathPickPlace(
        robots="Panda",
        controller_configs=load_composite_controller_config(controller="BASIC", robot="Panda"),
        start_xy=scenario.start_xy,
        target_xy=scenario.goal_xy,
        target_radius=float(workspace["target_radius"]),
        obstacle_xy=scenario.obstacle_xy,
        obstacle_half_size=scenario.obstacle_half_size,
        obstacle_height=float(config.raw["scenario"]["obstacle_height_m"]),
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=render,
        use_camera_obs=render,
        use_object_obs=True,
        camera_names=str(video["camera"]),
        camera_widths=int(video["width"]),
        camera_heights=int(video["height"]),
        control_freq=int(controller["control_frequency"]),
        horizon=1000,
        ignore_done=True,
        hard_reset=False,
    )


def _reset_gripper_state(env: HumanPathPickPlace) -> None:
    grippers = env.robots[0].gripper
    if isinstance(grippers, dict):
        for gripper in grippers.values():
            gripper.current_action = np.zeros(gripper.dof)


ContactCallback = Callable[[int, str, str], None]
ActionProvider = Callable[[FloatArray, float, FloatArray, FloatArray], FloatArray]


def _contact_types(
    env: HumanPathPickPlace,
    step_index: int,
    contact_callback: ContactCallback | None = None,
) -> tuple[bool, bool]:
    cube_geoms = {env.sim.model.geom_name2id(name) for name in env.cube.contact_geoms}
    object_collision = False
    robot_collision = False
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        name1 = env.sim.model.geom_id2name(int(contact.geom1)) or f"geom_{int(contact.geom1)}"
        name2 = env.sim.model.geom_id2name(int(contact.geom2)) or f"geom_{int(contact.geom2)}"
        if contact_callback is not None:
            contact_callback(step_index, name1, name2)
        pair = {int(contact.geom1), int(contact.geom2)}
        if env.obstacle_geom_id not in pair:
            continue
        other = next(iter(pair - {env.obstacle_geom_id}), -1)
        if other in cube_geoms:
            object_collision = True
        else:
            name = env.sim.model.geom_id2name(other) or ""
            robot_collision |= name.startswith("robot0_")
    return object_collision, robot_collision


def execute_rollout(
    config: Phase4BConfig,
    env: HumanPathPickPlace,
    scenario: Scenario,
    method: str,
    path: FloatArray,
    selection: dict[str, Any],
    capture_video: bool = False,
    contact_callback: ContactCallback | None = None,
    pre_lower_yaw_steps: int = 0,
    pre_lower_yaw_command: float = 0.0,
    goal_side_lower_offset_m: float = 0.0,
    lateral_staging_xy: FloatArray | None = None,
    minimum_transport_cube_z_m: float | None = None,
    monitor_transport_height_drop: bool = True,
    action_provider: ActionProvider | None = None,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]]]:
    np.random.seed(scenario.seed)
    observation = env.reset()
    _reset_gripper_state(env)
    controller = config.phase4a.raw["controller"]
    workspace = config.phase4a.raw["workspace"]
    video_cfg = config.phase4a.raw["video"]
    eef_site = env.robots[0].eef_site_id["right"]
    table_z = float(workspace["table_z"])
    frames: list[NDArray[np.uint8]] = []
    actions: list[FloatArray] = []
    cube_path: list[FloatArray] = []
    eef_path: list[FloatArray] = []
    latencies_ms: list[float] = []
    phases: list[dict[str, Any]] = []
    saturation_steps = 0
    object_collision_steps = 0
    robot_collision_steps = 0
    dropped = False
    transport_start = 0
    transport_end = 0
    initial_eef = np.asarray(env.sim.data.site_xpos[eef_site]).copy()
    initial_cube = np.asarray(env.sim.data.body_xpos[env.cube_body_id]).copy()

    def step_toward(
        target: FloatArray,
        gripper: float,
        monitor_drop: bool = False,
        yaw_command: float = 0.0,
    ) -> float:
        nonlocal observation, saturation_steps, object_collision_steps
        nonlocal robot_collision_steps, dropped
        eef = np.asarray(env.sim.data.site_xpos[eef_site])
        cube_before = np.asarray(env.sim.data.body_xpos[env.cube_body_id])
        started = perf_counter_ns()
        if action_provider is None:
            action, saturated = bounded_action(
                target - eef,
                float(controller["position_output_limit"]),
                gripper,
            )
        else:
            raw_action = np.asarray(
                action_provider(target.copy(), gripper, eef.copy(), cube_before.copy()),
                dtype=np.float64,
            )
            if raw_action.shape != (7,) or not np.all(np.isfinite(raw_action)):
                raise ValueError("Action provider must return one finite seven-value action")
            saturated = bool(np.any(np.abs(raw_action) > 1.0))
            action = np.clip(raw_action, -1.0, 1.0)
        action[5] = float(np.clip(yaw_command, -1.0, 1.0))
        latencies_ms.append((perf_counter_ns() - started) / 1_000_000.0)
        saturation_steps += int(saturated)
        observation, _, _, _ = env.step(action)
        object_collision, robot_collision = _contact_types(
            env, len(actions), contact_callback
        )
        object_collision_steps += int(object_collision)
        robot_collision_steps += int(robot_collision)
        cube = np.asarray(env.sim.data.body_xpos[env.cube_body_id]).copy()
        eef_after = np.asarray(env.sim.data.site_xpos[eef_site]).copy()
        if monitor_drop:
            height_threshold = (
                minimum_transport_cube_z_m
                if minimum_transport_cube_z_m is not None
                else table_z + 0.08
            )
            height_drop = monitor_transport_height_drop and cube[2] < height_threshold
            dropped |= bool(height_drop or np.linalg.norm(cube - eef_after) > 0.08)
        actions.append(action.copy())
        cube_path.append(cube)
        eef_path.append(eef_after)
        if capture_video and len(actions) % int(config.raw["video"]["frame_stride"]) == 0:
            frame = observation[f"{video_cfg['camera']}_image"]
            frames.append(np.asarray(frame, dtype=np.uint8))
        return float(np.linalg.norm(target - eef_after))

    def converge(name: str, target: FloatArray, gripper: float) -> None:
        start_step = len(actions)
        error = float("inf")
        for _ in range(int(controller["phase_timeout_steps"])):
            error = step_toward(target, gripper)
            if error <= float(controller["position_tolerance"]):
                break
        phases.append(
            {"name": name, "steps": len(actions) - start_step, "final_position_error_m": error}
        )

    start_xy, goal_xy = np.asarray(path[0]), np.asarray(path[-1])
    approach = np.asarray([*start_xy, float(workspace["approach_z"])])
    grasp = np.asarray([*start_xy, float(workspace["grasp_z"])])
    lift = np.asarray([*start_xy, float(workspace["transport_z"])])
    converge("approach", approach, -1.0)
    converge("grasp", grasp, -1.0)
    start_step = len(actions)
    for _ in range(int(controller["grasp_steps"])):
        step_toward(grasp, 1.0)
    phases.append({"name": "close", "steps": len(actions) - start_step})
    converge("lift", lift, 1.0)
    transport_start = len(actions)
    for xy in path:
        target = np.asarray([xy[0], xy[1], float(workspace["transport_z"])])
        for _ in range(int(controller["waypoint_steps"])):
            step_toward(target, 1.0, monitor_drop=True)
    transport_end = len(actions)
    phases.append(
        {
            "name": f"transport_{method}",
            "steps": transport_end - transport_start,
            "waypoints": len(path),
        }
    )
    if pre_lower_yaw_steps:
        posture_target = np.asarray([*goal_xy, float(workspace["transport_z"])])
        start_step = len(actions)
        for _ in range(pre_lower_yaw_steps):
            step_toward(
                posture_target,
                1.0,
                monitor_drop=True,
                yaw_command=pre_lower_yaw_command,
            )
        phases.append(
            {"name": "pre_lower_safe_posture", "steps": len(actions) - start_step}
        )
    lower = np.asarray([*goal_xy, float(workspace["grasp_z"])])
    staging_high: FloatArray | None = None
    staging_low: FloatArray | None = None
    if lateral_staging_xy is not None:
        staging_xy = np.asarray(lateral_staging_xy, dtype=np.float64)
        staging_high = np.asarray([*staging_xy, float(workspace["transport_z"])])
        staging_low = np.asarray([*staging_xy, float(workspace["grasp_z"])])
        converge("route_side_staging", staging_high, 1.0)
        converge("lower_route_side", staging_low, 1.0)
        converge("place_from_route_side", lower, 1.0)
    elif goal_side_lower_offset_m > 0.0:
        goal_side = goal_xy - scenario.obstacle_xy
        goal_side /= max(float(np.linalg.norm(goal_side)), 1e-12)
        staging_xy = goal_xy + goal_side * goal_side_lower_offset_m
        staging_high = np.asarray([*staging_xy, float(workspace["transport_z"])])
        staging_low = np.asarray([*staging_xy, float(workspace["grasp_z"])])
        converge("goal_side_staging", staging_high, 1.0)
        converge("lower_goal_side", staging_low, 1.0)
        converge("place_from_goal_side", lower, 1.0)
    else:
        converge("lower", lower, 1.0)
    start_step = len(actions)
    for _ in range(int(controller["release_steps"])):
        step_toward(lower, -1.0)
    phases.append({"name": "release", "steps": len(actions) - start_step})
    if lateral_staging_xy is not None:
        assert staging_low is not None and staging_high is not None
        converge("retreat_route_side_low", staging_low, -1.0)
        converge("retreat_route_side_high", staging_high, -1.0)
        retreat = staging_high
    else:
        retreat = np.asarray([*goal_xy, float(workspace["approach_z"])])
        converge("retreat", retreat, -1.0)
    start_step = len(actions)
    for _ in range(int(controller["settle_steps"])):
        step_toward(retreat, -1.0)
    phases.append({"name": "settle", "steps": len(actions) - start_step})

    action_values = np.asarray(actions)
    cube_values = np.asarray(cube_path)
    eef_values = np.asarray(eef_path)
    transport_cube = cube_values[transport_start:transport_end, :2]
    transport_grasp_distance = np.linalg.norm(
        cube_values[transport_start:transport_end]
        - eef_values[transport_start:transport_end],
        axis=1,
    )
    movement = float(np.sum(np.linalg.norm(np.diff(transport_cube, axis=0), axis=1)))
    direct = float(np.linalg.norm(scenario.goal_xy - scenario.start_xy))
    path_efficiency = min(1.0, direct / movement) if movement > 1e-9 else 0.0
    clearances = signed_rectangle_clearance(
        transport_cube,
        scenario.obstacle_xy,
        scenario.obstacle_half_size,
        float(config.raw["scenario"]["object_radius_m"]),
    )
    cube_final = cube_values[-1]
    placement_error = float(np.linalg.norm(cube_final[:2] - scenario.goal_xy))
    collision_steps = object_collision_steps
    target_placed = bool(env._check_success())
    success = bool(target_placed and collision_steps == 0 and not dropped)
    reasons: list[str] = []
    if not target_placed:
        reasons.append("final_placement_outside_target")
    if object_collision_steps:
        reasons.append("object_obstacle_collision")
    if dropped:
        reasons.append("object_drop_during_transport")
    result = {
        "schema_version": 1,
        "method": method,
        "scenario": scenario.serialise(),
        **selection,
        "task_success": success,
        "target_placed": target_placed,
        "failure_reasons": reasons,
        "collision": bool(collision_steps),
        "collision_steps": collision_steps,
        "object_collision_steps": object_collision_steps,
        "robot_obstacle_contact": bool(robot_collision_steps),
        "robot_obstacle_contact_steps": robot_collision_steps,
        "drop": dropped,
        "maximum_transport_grasp_distance_m": float(np.max(transport_grasp_distance)),
        "transport_cube_center_z_median_m": float(
            np.median(cube_values[transport_start:transport_end, 2])
        ),
        "transport_eef_z_median_m": float(
            np.median(eef_values[transport_start:transport_end, 2])
        ),
        "transport_grasp_offset_z_median_m": float(
            np.median(
                eef_values[transport_start:transport_end, 2]
                - cube_values[transport_start:transport_end, 2]
            )
        ),
        "placement_error_m": placement_error,
        "steps": len(actions),
        "path_efficiency": path_efficiency,
        "minimum_obstacle_clearance_m": float(np.min(clearances)),
        "action_saturation_steps": saturation_steps,
        "action_saturation_rate": saturation_steps / len(actions),
        "action_bound_violations": int(np.sum(np.abs(action_values) > 1.0 + 1e-12)),
        "maximum_absolute_action": float(np.max(np.abs(action_values))),
        "controller_latency_median_ms": float(np.median(latencies_ms)),
        "controller_latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "phases": phases,
        "_latencies_ms": latencies_ms,
        "_actions": action_values,
        "_initial_eef": initial_eef,
        "_initial_cube": initial_cube,
        "_eef_path": eef_values,
        "_cube_path": cube_values,
    }
    return result, frames


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _check_record(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if int(record["action_bound_violations"]) != 0:
        failures.append("action_bound_violation")
    numeric = (
        "placement_error_m",
        "path_efficiency",
        "minimum_obstacle_clearance_m",
        "action_saturation_rate",
        "controller_latency_median_ms",
        "controller_latency_p95_ms",
    )
    if not all(np.isfinite(float(record[key])) for key in numeric):
        failures.append("non_finite_metric")
    return failures


def _deterministic_reset_error(env: HumanPathPickPlace, seed: int) -> float:
    np.random.seed(seed)
    env.reset()
    _reset_gripper_state(env)
    first = reset_signature(env)
    np.random.seed(seed)
    env.reset()
    _reset_gripper_state(env)
    return float(np.max(np.abs(reset_signature(env) - first)))


def _run_scenarios(
    config: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
    seeds: list[int],
    stop_on_infrastructure_failure: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    infrastructure_failures: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        env: HumanPathPickPlace | None = None
        try:
            scenario = make_scenario(config, demos, geometry, seed, index)
            env = _make_environment(config, scenario, render=False)
            reset_error = _deterministic_reset_error(env, seed)
            if reset_error > 1e-9:
                raise RuntimeError(f"deterministic reset error {reset_error:.3e}")
            for method in METHODS:
                path, selection = select_method_path(config, demos, scenario, method)
                if path.shape != (int(config.phase4a.raw["trajectory"]["transport_samples"]), 2):
                    raise RuntimeError(f"invalid path shape for {method}: {path.shape}")
                if not np.all(np.isfinite(path)):
                    raise RuntimeError(f"non-finite path for {method}")
                record, _ = execute_rollout(config, env, scenario, method, path, selection)
                record["deterministic_reset_max_error"] = reset_error
                problems = _check_record(record)
                if problems:
                    raise RuntimeError(f"{method}: {', '.join(problems)}")
                records.append(record)
        except Exception as error:  # noqa: BLE001 - persisted as an infrastructure result
            infrastructure_failures.append(
                {"scenario_index": index, "seed": seed, "error": f"{type(error).__name__}: {error}"}
            )
            if stop_on_infrastructure_failure:
                break
        finally:
            if env is not None:
                env.close()
    return records, infrastructure_failures


def run_preflight(config: Phase4BConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    output.mkdir(parents=True, exist_ok=True)
    demos, geometry = build_library(config)
    seeds = [int(value) for value in config.raw["preflight"]["seeds"]]
    records, failures = _run_scenarios(config, demos, geometry, seeds, True)
    method_counts = {method: sum(row["method"] == method for row in records) for method in METHODS}
    passed = not failures and all(count == 5 for count in method_counts.values())
    report = {
        "schema_version": 1,
        "phase": "4B_preflight",
        "passed": passed,
        "config_sha256": config.digest,
        "seeds": seeds,
        "method_counts": method_counts,
        "infrastructure_failures": failures,
        "task_failures": sum(not row["task_success"] for row in records),
    }
    _write_jsonl(output / "preflight_rollouts.jsonl", [_public_record(row) for row in records])
    _write_json(output / "preflight.json", report)
    return report


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - margin, center + margin


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        successes = sum(bool(row["task_success"]) for row in rows)
        low, high = wilson_interval(successes, len(rows))
        latencies = np.concatenate(
            [np.asarray(row["_latencies_ms"], dtype=np.float64) for row in rows]
        )
        aggregate[method] = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "success_rate_ci95": [low, high],
            "collision_episodes": sum(bool(row["collision"]) for row in rows),
            "robot_contact_episodes": sum(
                bool(row["robot_obstacle_contact"]) for row in rows
            ),
            "drop_episodes": sum(bool(row["drop"]) for row in rows),
            "placement_error_median_m": float(
                np.median([row["placement_error_m"] for row in rows])
            ),
            "placement_error_p95_m": float(
                np.percentile([row["placement_error_m"] for row in rows], 95)
            ),
            "steps_median": float(np.median([row["steps"] for row in rows])),
            "path_efficiency_median": float(np.median([row["path_efficiency"] for row in rows])),
            "minimum_obstacle_clearance_median_m": float(
                np.median([row["minimum_obstacle_clearance_m"] for row in rows])
            ),
            "minimum_obstacle_clearance_worst_m": float(
                np.min([row["minimum_obstacle_clearance_m"] for row in rows])
            ),
            "action_saturation_rate": sum(row["action_saturation_steps"] for row in rows)
            / sum(row["steps"] for row in rows),
            "controller_latency_median_ms": float(np.median(latencies)),
            "controller_latency_p95_ms": float(np.percentile(latencies, 95)),
        }
    return aggregate


def _comparison_markdown(
    aggregate: dict[str, Any], success_label: str = "Success"
) -> str:
    header = (
        f"| Method | {success_label} (95% CI) | Object collisions | Robot contacts | Drops | "
        "Placement median | Steps median | "
        "Path efficiency | Min clearance median | Saturation | Latency med/p95 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    labels = {
        "straight": "Straight line",
        "raw_replay": "Nearest raw replay",
        "dmp": "DMP retargeting",
        "dmp_route_confidence": "DMP + route/confidence",
    }
    lines = []
    for method in METHODS:
        row = aggregate[method]
        low, high = row["success_rate_ci95"]
        lines.append(
            f"| {labels[method]} | {row['successes']}/{row['episodes']} "
            f"({100 * low:.1f}–{100 * high:.1f}%) | {row['collision_episodes']} | "
            f"{row['robot_contact_episodes']} | {row['drop_episodes']} | "
            f"{1000 * row['placement_error_median_m']:.1f} mm | "
            f"{row['steps_median']:.0f} | {row['path_efficiency_median']:.3f} | "
            f"{1000 * row['minimum_obstacle_clearance_median_m']:.1f} mm | "
            f"{100 * row['action_saturation_rate']:.2f}% | "
            f"{row['controller_latency_median_ms']:.4f}/{row['controller_latency_p95_ms']:.4f} ms |"
        )
    return header + "\n".join(lines) + "\n"


def _plot_comparison(
    path: Path, aggregate: dict[str, Any], success_label: str = "Object-level task success"
) -> None:
    methods = ["dmp_route_confidence", "dmp", "raw_replay", "straight"]
    labels = [
        "Route/confidence DMP",
        "DMP retargeting",
        "Nearest raw replay",
        "Straight line",
    ]
    colors = ["#0072B2", "#738694", "#A7B2B9", "#D55E00"]
    successes = np.asarray([aggregate[method]["successes"] for method in methods])
    episodes = np.asarray([aggregate[method]["episodes"] for method in methods])
    failures = episodes - successes
    clearance = np.asarray(
        [aggregate[method]["minimum_obstacle_clearance_median_m"] for method in methods]
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.5),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    y = np.arange(len(methods))
    safe_scope = success_label.startswith("Calibrated safe")
    outcome_word = "safe" if safe_scope else "successful"
    axes[0].barh(y, successes, color="#0072B2", height=0.58)
    axes[0].barh(
        y,
        failures,
        left=successes,
        color="#D55E00",
        edgecolor="#5C310E",
        linewidth=0.3,
        hatch="///",
        height=0.58,
    )
    for index, (safe, failed) in enumerate(zip(successes, failures, strict=True)):
        axes[0].text(
            52.0,
            index,
            f"{safe} {outcome_word}  /  {failed} failed",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#202124",
        )
    axes[0].axvline(50, color="#8A939F", linewidth=0.8)
    axes[0].set_xlim(0, 69)
    axes[0].set_xticks([0, 10, 20, 30, 40, 50])
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Rollouts (50 fixed scenarios)")
    axes[0].set_title(
        "Observed safety outcomes" if safe_scope else "Object-level task outcomes",
        loc="left",
        fontweight="bold",
    )

    clearance_mm = 1000.0 * clearance
    bars = axes[1].barh(y, clearance_mm, color=colors, height=0.58)
    for bar, value in zip(bars, clearance_mm, strict=True):
        negative = value < 0
        axes[1].text(
            value / 2.0 if negative else value + 1.2,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.1f} mm",
            ha="center" if negative else "left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#202124",
        )
    axes[1].axvline(0.0, color="#202124", linewidth=0.9)
    axes[1].set_xlim(-58, 32)
    axes[1].set_xlabel("Median footprint-adjusted clearance (mm)")
    axes[1].set_title("Median obstacle clearance", loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(axis="x", color="#D7DCE2", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#B8C0CA")
        axis.spines["bottom"].set_color("#B8C0CA")
    figure.savefig(path, dpi=160, metadata={"Software": "phone2panda"})
    plt.close(figure)


def _write_video(path: Path, config: Phase4BConfig, frames: list[NDArray[np.uint8]]) -> None:
    writer = imageio.get_writer(
        path,
        fps=int(config.phase4a.raw["video"]["fps"]),
        codec="libx264",
        quality=7,
        macro_block_size=None,
        ffmpeg_log_level="error",
        output_params=["-an", "-map_metadata", "-1"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def _representative(
    records: list[dict[str, Any]], success: bool, preferred_method: str
) -> dict[str, Any] | None:
    choices = [row for row in records if bool(row["task_success"]) is success]
    preferred = [row for row in choices if row["method"] == preferred_method]
    pool = preferred or choices
    if not pool:
        return None
    return min(
        pool,
        key=lambda row: row["placement_error_m"] if success else row["scenario"]["index"],
    )


def _render_representatives(
    config: Phase4BConfig,
    demos: list[Demonstration],
    geometry: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = config.project_path("output_dir")
    video_cfg = config.raw["video"]
    selections = [
        ("success", _representative(records, True, str(video_cfg["success_method_preference"]))),
        ("failure", _representative(records, False, str(video_cfg["failure_method_preference"]))),
    ]
    artifacts: list[dict[str, Any]] = []
    for label, selected in selections:
        if selected is None:
            continue
        scenario_data = selected["scenario"]
        scenario = make_scenario(
            config,
            demos,
            geometry,
            int(scenario_data["seed"]),
            int(scenario_data["index"]),
        )
        method = str(selected["method"])
        path, selection = select_method_path(config, demos, scenario, method)
        env = _make_environment(config, scenario, render=True)
        try:
            replay, frames = execute_rollout(
                config, env, scenario, method, path, selection, capture_video=True
            )
        finally:
            env.close()
        filename = f"representative_{label}.mp4"
        _write_video(output / filename, config, frames)
        artifacts.append(
            {
                "kind": label,
                "path": filename,
                "method": method,
                "seed": scenario.seed,
                "scenario_index": scenario.index,
                "evaluation_success": bool(selected["task_success"]),
                "replay_success": bool(replay["task_success"]),
                "frames": len(frames),
            }
        )
    return artifacts


def run_evaluation(config: Phase4BConfig) -> dict[str, Any]:
    output = config.project_path("output_dir")
    preflight_path = output / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("Run the five-seed preflight before evaluation")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight["passed"] or preflight["config_sha256"] != config.digest:
        raise RuntimeError("A passing preflight for the current configuration is required")
    demos, geometry = build_library(config)
    evaluation_cfg = config.raw["evaluation"]
    seeds = list(
        range(
            int(evaluation_cfg["seed_start"]),
            int(evaluation_cfg["seed_start"]) + int(evaluation_cfg["episode_count"]),
        )
    )
    records, failures = _run_scenarios(config, demos, geometry, seeds, True)
    if failures:
        _write_json(output / "evaluation_infrastructure_failure.json", failures)
        raise RuntimeError(f"Evaluation infrastructure failure: {failures[0]['error']}")
    counts = {method: sum(row["method"] == method for row in records) for method in METHODS}
    if any(count != 50 for count in counts.values()):
        raise RuntimeError(f"Expected exactly 50 evaluation episodes per method, got {counts}")
    aggregate = aggregate_records(records)
    public_records = [_public_record(row) for row in records]
    _write_jsonl(output / "rollouts.jsonl", public_records)
    artifacts = _render_representatives(config, demos, geometry, records)
    report = {
        "schema_version": 1,
        "phase": "4B",
        "accepted": True,
        "config_sha256": config.digest,
        "fixed_seeds": seeds,
        "method_episode_counts": counts,
        "aggregate": aggregate,
        "representative_artifacts": artifacts,
        "versions": {
            "python": platform.python_version(),
            "robosuite": robosuite.__version__,
            "mujoco": mujoco.__version__,
        },
        "generated_at_unix": int(time.time()),
    }
    _write_json(output / "aggregate.json", report)
    (output / "comparison.md").write_text(_comparison_markdown(aggregate), encoding="utf-8")
    _plot_comparison(output / "comparison.png", aggregate)
    _write_json(
        output / "run_config.json",
        {"config_sha256": config.digest, "configuration": config.raw},
    )
    return report
