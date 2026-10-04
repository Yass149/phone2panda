from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from phone2panda.evaluation.phase4b import make_scenario, select_method_path
from phone2panda.evaluation.phase4e import _execution_config
from phone2panda.evaluation.phase5a import policy_features
from phone2panda.evaluation.phase5b import load_phase5b_config
from phone2panda.policy.gru import GRUPolicy
from phone2panda.public_demo import (
    BUNDLE_DIRECTORY,
    CHECKPOINT,
    SPLIT_MANIFEST,
    load_motion_priors,
    representative_scenarios,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def test_public_bundle_reproduces_all_saved_test_scenes_and_routes():
    demos, geometry = load_motion_priors(ROOT / BUNDLE_DIRECTORY)
    config = load_phase5b_config(ROOT / "configs/phase5b.yaml")
    execution, _ = _execution_config(config.phase5a.phase4e)
    rows = json.loads((ROOT / SPLIT_MANIFEST).read_text())["splits"]["testing"]
    assert len(demos) == 36
    assert len(rows) == 50
    for row in rows:
        scene = make_scenario(execution, demos, geometry, row["seed"], row["scenario_index"])
        _, selection = select_method_path(execution, demos, scene, "dmp_route_confidence")
        np.testing.assert_allclose(scene.start_xy, row["start_xy_m"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(scene.goal_xy, row["goal_xy_m"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(scene.obstacle_xy, row["obstacle_xy_m"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            scene.obstacle_half_size, row["obstacle_half_size_m"], rtol=0, atol=1e-12
        )
        assert selection["source_episode"] == row["source_episode"]
        assert selection["source_route"] == row["selected_route"]
    chosen = representative_scenarios(rows)
    assert len(chosen) == 5
    assert {(row["start_id"], row["selected_route"]) for row in chosen} == {
        ("s1", "left"), ("s2", "left"), ("s2", "right"), ("s3", "right"),
    }


def test_public_bundle_rejects_corrupt_and_nonfinite_arrays(tmp_path):
    shutil.copytree(ROOT / BUNDLE_DIRECTORY, tmp_path / "bundle")
    directory = tmp_path / "bundle"
    archive = directory / "routes.npz"
    original = archive.read_bytes()
    archive.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_motion_priors(directory)
    archive.write_bytes(original)
    with np.load(archive, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["dmp_robot"][0, 0, 0] = np.nan
    np.savez_compressed(archive, **arrays)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["archive_sha256"] = sha256(archive)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="non-finite"):
        load_motion_priors(directory)


def test_saved_checkpoint_runs_with_public_route_features():
    demos, geometry = load_motion_priors(ROOT / BUNDLE_DIRECTORY)
    config = load_phase5b_config(ROOT / "configs/phase5b.yaml")
    execution, _ = _execution_config(config.phase5a.phase4e)
    row = json.loads((ROOT / SPLIT_MANIFEST).read_text())["splits"]["testing"][0]
    scene = make_scenario(execution, demos, geometry, row["seed"], row["scenario_index"])
    path, selection = select_method_path(execution, demos, scene, "dmp_route_confidence")
    policy, inputs, outputs, _ = GRUPolicy.load(ROOT / CHECKPOINT)
    eef = np.r_[scene.start_xy, 0.847715]
    cube = np.r_[scene.start_xy, 0.82]
    target = np.r_[path[1], 0.847715]
    features = policy_features(eef, cube, target, scene, selection["source_route"], 1, 1)
    action, hidden = policy.step(inputs.transform(features), policy.initial_state())
    assert policy.parameter_count == 3559
    assert np.isfinite(outputs.inverse(action[0])).all()
    assert np.isfinite(hidden).all()
