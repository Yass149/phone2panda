from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "cgl" if platform.system() == "Darwin" else "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import robosuite
import yaml
from numpy.typing import NDArray
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.utils.mjcf_utils import new_body, new_geom
from robosuite.utils.placement_samplers import UniformRandomSampler

from phone2panda.retarget.mapping import WorkspaceMap, bounded_action
from phone2panda.trajectories.dmp import fit_dmp, rollout_dmp
from phone2panda.trajectories.processing import load_processed_path, smooth_and_resample
from phone2panda.trajectories.public_data import resolve_processed_path

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Phase4AConfig:
    root: Path
    raw: dict[str, Any]

    def path(self, key: str) -> Path:
        return self.root / str(self.raw[key])


def load_phase4a_config(path: Path) -> Phase4AConfig:
    resolved = path.resolve()
    return Phase4AConfig(
        root=resolved.parent.parent,
        raw=yaml.safe_load(resolved.read_text(encoding="utf-8")),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class HumanPathPickPlace(Lift):
    def __init__(
        self,
        *args: Any,
        start_xy: FloatArray,
        target_xy: FloatArray,
        target_radius: float,
        obstacle_xy: FloatArray,
        obstacle_half_size: FloatArray,
        obstacle_height: float,
        **kwargs: Any,
    ) -> None:
        self.start_xy = np.asarray(start_xy, dtype=np.float64)
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self.target_radius = float(target_radius)
        self.obstacle_xy = np.asarray(obstacle_xy, dtype=np.float64)
        self.obstacle_half_size = np.asarray(obstacle_half_size, dtype=np.float64)
        self.obstacle_height = float(obstacle_height)
        super().__init__(*args, **kwargs)

    def _load_model(self) -> None:
        super()._load_model()
        self.placement_initializer = UniformRandomSampler(
            name="DeterministicCubeSampler",
            mujoco_objects=self.cube,
            x_range=(float(self.start_xy[0]), float(self.start_xy[0])),
            y_range=(float(self.start_xy[1]), float(self.start_xy[1])),
            rotation=0.0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )
        obstacle = new_body(
            name="route_obstacle",
            pos=[
                float(self.obstacle_xy[0]),
                float(self.obstacle_xy[1]),
                float(self.table_offset[2] + self.obstacle_height / 2.0),
            ],
        )
        obstacle.append(
            new_geom(
                name="route_obstacle_geom",
                type="box",
                size=[
                    float(self.obstacle_half_size[0]),
                    float(self.obstacle_half_size[1]),
                    float(self.obstacle_height / 2.0),
                ],
                rgba=[0.1, 0.25, 0.8, 1.0],
                friction=[1.0, 0.005, 0.0001],
                contype="1",
                conaffinity="1",
            )
        )
        target = new_body(
            name="target_marker",
            pos=[float(self.target_xy[0]), float(self.target_xy[1]), self.table_offset[2] + 0.002],
        )
        target.append(
            new_geom(
                name="target_marker_geom",
                type="cylinder",
                size=[self.target_radius, 0.002],
                rgba=[0.1, 0.8, 0.2, 0.45],
                group="1",
                contype="0",
                conaffinity="0",
            )
        )
        self.model.worldbody.extend([obstacle, target])

    def _setup_references(self) -> None:
        super()._setup_references()
        self.obstacle_geom_id = self.sim.model.geom_name2id("route_obstacle_geom")

    def _check_success(self) -> bool:
        cube = np.asarray(self.sim.data.body_xpos[self.cube_body_id])
        horizontal_error = float(np.linalg.norm(cube[:2] - self.target_xy))
        return horizontal_error <= self.target_radius and cube[2] <= self.table_offset[2] + 0.05

    def obstacle_collision(self) -> bool:
        cube_geoms = {self.sim.model.geom_name2id(name) for name in self.cube.contact_geoms}
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.obstacle_geom_id in pair and pair.intersection(cube_geoms):
                return True
        return False


def _episode_report(config: Phase4AConfig) -> dict[str, Any]:
    report = json.loads(config.path("dataset_report").read_text(encoding="utf-8"))
    episode_id = str(config.raw["episode_id"])
    return next(item for item in report["episodes"] if item["episode_id"] == episode_id)


def prepare_human_transport(config: Phase4AConfig) -> dict[str, Any]:
    trajectory = config.raw["trajectory"]
    workspace = config.raw["workspace"]
    raw_points = load_processed_path(
        resolve_processed_path(config.root, config.raw["trajectory_csv"])
    )
    smoothed = smooth_and_resample(
        raw_points,
        int(trajectory["transport_samples"]),
        int(trajectory["savgol_window"]),
        int(trajectory["savgol_order"]),
    )
    model = fit_dmp(smoothed, int(trajectory["dmp_basis_functions"]))
    reproduced = rollout_dmp(model, int(trajectory["transport_samples"]))
    mapping = WorkspaceMap(
        tuple(float(value) for value in workspace["robot_x_range"]),
        tuple(float(value) for value in workspace["robot_y_range"]),
    )
    robot_path = mapping.map_points(reproduced)
    episode = _episode_report(config)
    obstacle_rect = tuple(float(value) for value in episode["task"]["obstacle_rect"])
    obstacle_xy, obstacle_half_size = mapping.map_rect(obstacle_rect)
    return {
        "raw": raw_points,
        "smoothed": smoothed,
        "dmp": reproduced,
        "robot": robot_path,
        "mapping": mapping,
        "obstacle_xy": obstacle_xy,
        "obstacle_half_size": obstacle_half_size,
        "episode": episode,
    }


def make_environment(
    config: Phase4AConfig, prepared: dict[str, Any], render: bool
) -> HumanPathPickPlace:
    raw = config.raw
    workspace = raw["workspace"]
    video = raw["video"]
    path = prepared["robot"]
    return HumanPathPickPlace(
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        start_xy=path[0],
        target_xy=path[-1],
        target_radius=float(workspace["target_radius"]),
        obstacle_xy=prepared["obstacle_xy"],
        obstacle_half_size=prepared["obstacle_half_size"],
        obstacle_height=float(workspace["obstacle_height"]),
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=render,
        use_camera_obs=render,
        use_object_obs=True,
        camera_names=str(video["camera"]),
        camera_widths=int(video["width"]),
        camera_heights=int(video["height"]),
        control_freq=int(raw["controller"]["control_frequency"]),
        horizon=1000,
        ignore_done=True,
        hard_reset=False,
    )


def reset_signature(env: HumanPathPickPlace) -> FloatArray:
    eef = np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    cube = np.asarray(env.sim.data.body_xpos[env.cube_body_id])
    return np.concatenate([eef, cube])


@dataclass
class _Phase4AExecution:
    config: Phase4AConfig
    env: HumanPathPickPlace
    observation: dict[str, Any]
    controller: dict[str, Any] = field(init=False)
    eef_site: int = field(init=False)
    frames: list[NDArray[np.uint8]] = field(default_factory=list, init=False)
    actions: list[list[float]] = field(default_factory=list, init=False)
    cube_path: list[list[float]] = field(default_factory=list, init=False)
    eef_path: list[list[float]] = field(default_factory=list, init=False)
    phases: list[dict[str, Any]] = field(default_factory=list, init=False)
    saturation_count: int = field(default=0, init=False)
    obstacle_collision_steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.controller = self.config.raw["controller"]
        self.eef_site = self.env.robots[0].eef_site_id["right"]

    def step_toward(self, target: FloatArray, gripper: float) -> float:
        eef = np.asarray(self.env.sim.data.site_xpos[self.eef_site])
        action, saturated = bounded_action(
            target - eef,
            float(self.controller["position_output_limit"]),
            gripper,
        )
        self.saturation_count += int(saturated)
        self.observation, _, _, _ = self.env.step(action)
        self.obstacle_collision_steps += int(self.env.obstacle_collision())
        self.actions.append(action.tolist())
        self.cube_path.append(
            np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_id]).tolist()
        )
        self.eef_path.append(
            np.asarray(self.env.sim.data.site_xpos[self.eef_site]).tolist()
        )
        camera = self.config.raw["video"]["camera"]
        self.frames.append(np.asarray(self.observation[f"{camera}_image"], dtype=np.uint8))
        eef_after = np.asarray(self.env.sim.data.site_xpos[self.eef_site])
        return float(np.linalg.norm(target - eef_after))

    def converge(self, name: str, target: FloatArray, gripper: float) -> None:
        start = len(self.actions)
        error = float("inf")
        for _ in range(int(self.controller["phase_timeout_steps"])):
            error = self.step_toward(target, gripper)
            if error <= float(self.controller["position_tolerance"]):
                break
        self.phases.append(
            {
                "name": name,
                "steps": len(self.actions) - start,
                "final_position_error": error,
            }
        )

    def hold(self, name: str, target: FloatArray, gripper: float, steps: int) -> None:
        start = len(self.actions)
        for _ in range(steps):
            self.step_toward(target, gripper)
        self.phases.append(
            {"name": name, "steps": len(self.actions) - start, "final_position_error": 0.0}
        )

    def transport(self, path: FloatArray, transport_z: float) -> None:
        start = len(self.actions)
        for xy in path:
            target = np.asarray([xy[0], xy[1], transport_z])
            for _ in range(int(self.controller["waypoint_steps"])):
                self.step_toward(target, 1.0)
        self.phases.append(
            {
                "name": "transport_human_dmp",
                "steps": len(self.actions) - start,
                "waypoints": len(path),
            }
        )


def run_phase4a(config: Phase4AConfig) -> dict[str, Any]:
    seed = int(config.raw["seed"])
    np.random.seed(seed)
    prepared = prepare_human_transport(config)
    env = make_environment(config, prepared, render=True)
    output_dir = config.path("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = config.raw["controller"]
    workspace = config.raw["workspace"]
    env.reset()
    initial_signature = reset_signature(env)
    env.reset()
    deterministic_reset_error = float(np.max(np.abs(reset_signature(env) - initial_signature)))
    execution = _Phase4AExecution(config, env, env.reset())

    path = np.asarray(prepared["robot"])
    start_xy, goal_xy = path[0], path[-1]
    approach = np.asarray([*start_xy, float(workspace["approach_z"])])
    grasp = np.asarray([*start_xy, float(workspace["grasp_z"])])
    lift = np.asarray([*start_xy, float(workspace["transport_z"])])
    execution.converge("approach", approach, -1.0)
    execution.converge("grasp", grasp, -1.0)
    execution.hold("close", grasp, 1.0, int(controller["grasp_steps"]))
    execution.converge("lift", lift, 1.0)
    execution.transport(path, float(workspace["transport_z"]))
    lower = np.asarray([*goal_xy, float(workspace["grasp_z"])])
    execution.converge("lower", lower, 1.0)
    execution.hold("release", lower, -1.0, int(controller["release_steps"]))
    retreat = np.asarray([*goal_xy, float(workspace["approach_z"])])
    execution.converge("retreat", retreat, -1.0)
    execution.hold("settle", retreat, -1.0, int(controller["settle_steps"]))

    cube_final = np.asarray(env.sim.data.body_xpos[env.cube_body_id])
    placement_error = float(np.linalg.norm(cube_final[:2] - goal_xy))
    success = bool(env._check_success() and execution.obstacle_collision_steps == 0)
    obstacle_low = prepared["obstacle_xy"] - prepared["obstacle_half_size"]
    obstacle_high = prepared["obstacle_xy"] + prepared["obstacle_half_size"]
    cube_xy = np.asarray(execution.cube_path)[:, :2]
    obstacle_dx = np.maximum(
        np.maximum(obstacle_low[0] - cube_xy[:, 0], cube_xy[:, 0] - obstacle_high[0]),
        0.0,
    )
    obstacle_dy = np.maximum(
        np.maximum(obstacle_low[1] - cube_xy[:, 1], cube_xy[:, 1] - obstacle_high[1]),
        0.0,
    )
    minimum_obstacle_center_clearance = float(np.min(np.hypot(obstacle_dx, obstacle_dy)))
    robot_transport = np.asarray(execution.eef_path)[
        sum(phase["steps"] for phase in execution.phases[:4]) : sum(
            phase["steps"] for phase in execution.phases[:5]
        ),
        :2,
    ]
    straight_start, straight_goal = path[0], path[-1]
    line = straight_goal - straight_start
    line_norm = float(np.linalg.norm(line))
    deviation = np.abs(
        np.cross(line, robot_transport - straight_start) / max(line_norm, 1e-12)
    )
    video_path = output_dir / "phase4a_rollout.mp4"
    writer = imageio.get_writer(
        video_path,
        fps=int(config.raw["video"]["fps"]),
        codec="libx264",
        quality=7,
        macro_block_size=None,
        ffmpeg_log_level="error",
        output_params=["-an", "-map_metadata", "-1"],
    )
    for frame in execution.frames:
        writer.append_data(frame)
    writer.close()
    rollout_path = output_dir / "rollout.npz"
    np.savez_compressed(
        rollout_path,
        actions=np.asarray(execution.actions),
        cube_positions=np.asarray(execution.cube_path),
        eef_positions=np.asarray(execution.eef_path),
        human_normalised=np.asarray(prepared["smoothed"]),
        dmp_normalised=np.asarray(prepared["dmp"]),
        retargeted_waypoints=path,
    )
    metrics = {
        "schema_version": 1,
        "accepted": success,
        "seed": seed,
        "source_episode": config.raw["episode_id"],
        "source_route": prepared["episode"]["expected_route"],
        "controller": "OSC_POSE_fixed_impedance",
        "steps": len(execution.actions),
        "duration_seconds": len(execution.actions) / float(controller["control_frequency"]),
        "placement_error_m": placement_error,
        "target_radius_m": float(workspace["target_radius"]),
        "obstacle_collision_steps": execution.obstacle_collision_steps,
        "minimum_obstacle_center_clearance_m": minimum_obstacle_center_clearance,
        "controller_saturation_steps": execution.saturation_count,
        "maximum_absolute_action": float(np.max(np.abs(execution.actions))),
        "action_bound_violations": int(np.sum(np.abs(execution.actions) > 1.0)),
        "maximum_transport_deviation_from_straight_m": float(np.max(deviation)),
        "human_path_materially_controls_transport": bool(np.max(deviation) >= 0.04),
        "deterministic_reset_max_error": deterministic_reset_error,
        "phases": execution.phases,
        "versions": {
            "robosuite": robosuite.__version__,
            "mujoco": mujoco.__version__,
            "python": platform.python_version(),
        },
        "artifacts": {
            "video": "phase4a_rollout.mp4",
            "rollout": "rollout.npz",
            "config": "../../configs/phase4a.yaml",
        },
        "generated_at_unix": int(time.time()),
    }
    _write_json(output_dir / "rollout_metrics.json", metrics)
    env.close()
    return metrics
