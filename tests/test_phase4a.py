from pathlib import Path

import numpy as np

from phone2panda.retarget.mapping import WorkspaceMap, bounded_action
from phone2panda.sim.phase4a import (
    load_phase4a_config,
    make_environment,
    reset_signature,
)


def test_coordinate_mapping_preserves_canvas_orientation() -> None:
    mapping = WorkspaceMap((0.0, 0.25), (-0.225, 0.225))
    mapped = mapping.map_points(np.asarray([[0.0, 1.0], [1.0, 0.0]]))
    np.testing.assert_allclose(mapped[0], [0.0, 0.225])
    np.testing.assert_allclose(mapped[1], [0.25, -0.225])


def test_actions_are_bounded() -> None:
    action, saturated = bounded_action(np.asarray([0.5, -0.5, 0.01]), 0.05, 2.0)
    assert saturated
    assert action.shape == (7,)
    assert np.max(np.abs(action)) <= 1.0
    np.testing.assert_allclose(action[:3], [1.0, -1.0, 0.2])
    assert action[-1] == 1.0


def test_deterministic_reset() -> None:
    config = load_phase4a_config(Path("configs/phase4a.yaml"))
    prepared = {
        "robot": np.asarray([[0.04, -0.18], [0.22, 0.14]], dtype=np.float64),
        "obstacle_xy": np.asarray([0.13, 0.0], dtype=np.float64),
        "obstacle_half_size": np.asarray([0.035, 0.055], dtype=np.float64),
    }
    env = make_environment(config, prepared, render=False)
    try:
        np.random.seed(int(config.raw["seed"]))
        env.reset()
        first = reset_signature(env)
        env.reset()
        second = reset_signature(env)
        np.testing.assert_allclose(first, second, atol=1e-10)
    finally:
        env.close()
