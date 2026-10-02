from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WorkspaceMap:
    robot_x_range: tuple[float, float]
    robot_y_range: tuple[float, float]

    def map_points(self, normalised: FloatArray) -> FloatArray:
        points = np.asarray(normalised, dtype=np.float64)
        if points.shape[-1] != 2:
            raise ValueError("Normalised points must have an x-y coordinate pair")
        human_x = np.clip(points[..., 0], 0.0, 1.0)
        human_y = np.clip(points[..., 1], 0.0, 1.0)
        robot_x = self.robot_x_range[0] + (1.0 - human_y) * (
            self.robot_x_range[1] - self.robot_x_range[0]
        )
        robot_y = self.robot_y_range[0] + (1.0 - human_x) * (
            self.robot_y_range[1] - self.robot_y_range[0]
        )
        return np.stack([robot_x, robot_y], axis=-1)

    def map_rect(self, rect: tuple[float, float, float, float]) -> tuple[FloatArray, FloatArray]:
        corners = self.map_points(
            np.asarray([[rect[0], rect[1]], [rect[2], rect[3]]], dtype=np.float64)
        )
        low = np.minimum(corners[0], corners[1])
        high = np.maximum(corners[0], corners[1])
        return (low + high) / 2.0, (high - low) / 2.0


def bounded_action(
    position_error: FloatArray,
    position_output_limit: float,
    gripper_command: float,
) -> tuple[FloatArray, bool]:
    raw = np.asarray(position_error, dtype=np.float64) / position_output_limit
    clipped = np.clip(raw, -1.0, 1.0)
    action = np.zeros(7, dtype=np.float64)
    action[:3] = clipped
    action[-1] = float(np.clip(gripper_command, -1.0, 1.0))
    return action, bool(np.any(np.abs(raw) > 1.0))
