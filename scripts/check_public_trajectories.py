#!/usr/bin/env python3
"""Audit the numerical release and rebuild its DMPs against the saved bundle."""

from pathlib import Path

import numpy as np

from phone2panda.evaluation.phase4b import build_library
from phone2panda.evaluation.phase4e import _execution_config
from phone2panda.evaluation.phase5b import load_phase5b_config
from phone2panda.public_demo import BUNDLE_DIRECTORY, load_motion_priors
from phone2panda.trajectories.public_data import PUBLIC_DIRECTORY, validate_public_release


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_public_release(root / PUBLIC_DIRECTORY)
    config = load_phase5b_config(root / "configs/phase5b.yaml")
    execution, _ = _execution_config(config.phase5a.phase4e)
    rebuilt, geometry = build_library(execution, public_only=True)
    saved, expected_geometry = load_motion_priors(root / BUNDLE_DIRECTORY)
    assert [row.episode_id for row in rebuilt] == [row.episode_id for row in saved]
    for actual, expected in zip(rebuilt, saved, strict=True):
        np.testing.assert_allclose(actual.dmp_robot, expected.dmp_robot, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            actual.raw_robot[[0, -1]], expected.raw_robot, rtol=0, atol=1e-12,
        )
    for key in geometry:
        np.testing.assert_allclose(geometry[key], expected_geometry[key], rtol=0, atol=1e-12)
    print(
        f"Public data audit passed: {result['episodes']} episodes, "
        f"{result['rows']} rows; 36 DMPs rebuilt"
    )


if __name__ == "__main__":
    main()
