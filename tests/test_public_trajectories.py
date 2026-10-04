from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from phone2panda.evaluation.phase4b import build_library
from phone2panda.evaluation.phase4e import _execution_config
from phone2panda.evaluation.phase5b import load_phase5b_config
from phone2panda.evaluation.phase6 import build_ablation_library, load_phase6_config
from phone2panda.public_demo import BUNDLE_DIRECTORY, load_motion_priors
from phone2panda.sim.phase4a import load_phase4a_config, prepare_human_transport
from phone2panda.trajectories.public_data import (
    PRIVATE_DIRECTORY,
    PUBLIC_DIRECTORY,
    PUBLIC_FIELDS,
    digest,
    resolve_processed_path,
    validate_numeric_csv,
    validate_public_release,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def public_clone(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / PUBLIC_DIRECTORY, tmp_path / PUBLIC_DIRECTORY)
    for relative in ("data/metadata.csv", "results/dataset_quality/quality_report.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    assert not (tmp_path / PRIVATE_DIRECTORY).exists()
    return tmp_path


def test_public_release_and_rebuilt_priors_do_not_need_private_files(public_clone):
    report = validate_public_release(public_clone / PUBLIC_DIRECTORY)
    assert report == {"passed": True, "episodes": 36, "rows": 7963}
    config = load_phase5b_config(public_clone / "configs/phase5b.yaml")
    execution, _ = _execution_config(config.phase5a.phase4e)
    rebuilt, geometry = build_library(execution)
    saved, expected_geometry = load_motion_priors(ROOT / BUNDLE_DIRECTORY)
    assert [row.episode_id for row in rebuilt] == [row.episode_id for row in saved]
    for actual, expected in zip(rebuilt, saved, strict=True):
        np.testing.assert_allclose(actual.dmp_robot, expected.dmp_robot, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            actual.raw_robot[[0, -1]], expected.raw_robot, rtol=0, atol=1e-12,
        )
        assert actual.raw_robot.shape == (80, 2)
    for key in geometry:
        np.testing.assert_allclose(geometry[key], expected_geometry[key], rtol=0, atol=1e-12)


@pytest.mark.parametrize("smoothing,mapping", [
    (True, "calibrated_homography"), (False, "calibrated_homography"), (True, "naive_pixel"),
])
def test_ablation_inputs_and_phase4a_rebuild_from_public_data(public_clone, smoothing, mapping):
    config = load_phase6_config(public_clone / "configs/phase6.yaml")
    execution, _ = _execution_config(config.phase5b.phase5a.phase4e)
    demos = build_ablation_library(
        config, execution, ["ep_001", "ep_002", "ep_003"],
        mapping_mode=mapping, smoothing_enabled=smoothing,
    )
    assert len(demos) == 3
    assert all(np.isfinite(demo.dmp_robot).all() for demo in demos)
    prepared = prepare_human_transport(load_phase4a_config(public_clone / "configs/phase4a.yaml"))
    assert prepared["robot"].shape == (80, 2)


def test_public_release_rejects_corruption_and_extra_metadata(public_clone):
    directory = public_clone / PUBLIC_DIRECTORY
    path = directory / "trajectories/ep_001.csv"
    original = path.read_bytes()
    path.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_public_release(directory)
    with pytest.raises(ValueError, match="checksum"):
        resolve_processed_path(public_clone, PRIVATE_DIRECTORY / "ep_001.csv")
    path.write_bytes(original.replace(b"frame_index,", b"timestamp_seconds,"))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"][0]["public_csv_sha256"] = digest(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="numerical columns"):
        validate_public_release(directory)


@pytest.mark.parametrize("field,value,reason", [
    ("object_x", "nan", "normalized"), ("valid", "2", "flags"),
    ("frame_index", "20", "contiguous"),
])
def test_numeric_schema_rejects_bad_values(field, value, reason):
    data = (ROOT / PUBLIC_DIRECTORY / "trajectories/ep_001.csv").read_text()
    rows = list(csv.DictReader(io.StringIO(data)))
    rows[0][field] = value
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=PUBLIC_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    with pytest.raises(ValueError, match=reason):
        validate_numeric_csv(output.getvalue().encode())
