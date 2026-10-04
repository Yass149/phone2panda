"""Replay a saved controller using derived motion priors, not private frame data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from phone2panda.evaluation.phase4b import (
    Demonstration,
    make_scenario,
    select_method_path,
)
from phone2panda.evaluation.phase4e import _execution_config
from phone2panda.evaluation.phase5a import (
    ACTION_NAMES,
    FEATURE_NAMES,
    _public_record,
    _run_episode,
)
from phone2panda.evaluation.phase5b import _policy_factory, load_phase5b_config
from phone2panda.policy.gru import GRUPolicy

BUNDLE_DIRECTORY = Path("assets/motion_priors")
CHECKPOINT = Path("results/phase5b/gru_policy.npz")
SPLIT_MANIFEST = Path("results/phase5b/split_manifest.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_motion_priors(directory: Path) -> tuple[list[Demonstration], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported motion-prior bundle version")
    archive = directory / "routes.npz"
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Motion-prior archive checksum mismatch")
    with np.load(archive, allow_pickle=False) as data:
        paths = data["dmp_robot"].copy()
        endpoints = data["raw_endpoints"].copy()
        obstacle = data["obstacle_xy"].copy()
        half_size = data["obstacle_half_size"].copy()
    entries = manifest["demonstrations"]
    if not 1 <= len(entries) <= 36 or paths.shape != (len(entries), 80, 2):
        raise ValueError("Invalid motion-prior dimensions")
    if endpoints.shape != (len(entries), 2, 2):
        raise ValueError("Invalid endpoint dimensions")
    if obstacle.shape != (2,) or half_size.shape != (2,) or np.any(half_size <= 0):
        raise ValueError("Invalid obstacle geometry")
    if not all(np.isfinite(values).all() for values in (paths, endpoints, obstacle, half_size)):
        raise ValueError("Motion-prior bundle contains non-finite values")
    if len({row["episode_id"] for row in entries}) != len(entries):
        raise ValueError("Duplicate demonstration identifier")
    demos = []
    for index, row in enumerate(entries):
        if row["start_id"] not in {"s1", "s2", "s3"} or row["route"] not in {"left", "right"}:
            raise ValueError("Invalid start or route label")
        if not all(0 <= float(row[key]) <= 1 for key in ("coverage", "confidence")):
            raise ValueError("Invalid demonstration confidence")
        demos.append(
            Demonstration(
                episode_id=str(row["episode_id"]),
                start_id=str(row["start_id"]),
                route=str(row["route"]),
                split=str(row["split"]),
                coverage=float(row["coverage"]),
                confidence=float(row["confidence"]),
                # Only endpoints are retained; raw-replay evaluation is not supported.
                raw_robot=endpoints[index],
                dmp_robot=paths[index],
            )
        )
    return demos, {"obstacle_xy": obstacle, "obstacle_half_size": half_size}


def representative_scenarios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cover every available start/route pair before adding a fifth fixed case."""
    selected = []
    seen = set()
    for row in rows:
        pair = (row["start_id"], row["selected_route"])
        if pair not in seen:
            selected.append(row)
            seen.add(pair)
    remaining = [row for row in rows if row not in selected]
    return (selected + remaining)[:5]


def run_public_demo(root: Path, *, episodes: int = 5) -> dict[str, Any]:
    if not 1 <= episodes <= 5:
        raise ValueError("The public smoke test runs between one and five episodes")
    root = root.resolve()
    config = load_phase5b_config(root / "configs/phase5b.yaml")
    execution, _ = _execution_config(config.phase5a.phase4e)
    directory = root / BUNDLE_DIRECTORY
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected_hash in manifest["reproduction_sha256"].items():
        if sha256(root / relative) != expected_hash:
            raise ValueError(f"Reproduction input changed: {relative}")
    demos, geometry = load_motion_priors(directory)
    policy, input_normalizer, output_normalizer, _ = GRUPolicy.load(root / CHECKPOINT)
    if (policy.input_dim, policy.output_dim) != (len(FEATURE_NAMES), len(ACTION_NAMES)):
        raise ValueError("Checkpoint feature/action dimensions do not match the controller")
    specifications = json.loads((root / SPLIT_MANIFEST).read_text())["splits"]["testing"]
    records = []
    for row in representative_scenarios(specifications)[:episodes]:
        seed, index = int(row["seed"]), int(row["scenario_index"])
        scenario = make_scenario(execution, demos, geometry, seed, index)
        _, selection = select_method_path(execution, demos, scenario, "dmp_route_confidence")
        for field, actual in (
            ("start_xy_m", scenario.start_xy),
            ("goal_xy_m", scenario.goal_xy),
            ("obstacle_xy_m", scenario.obstacle_xy),
            ("obstacle_half_size_m", scenario.obstacle_half_size),
        ):
            if not np.allclose(actual, row[field], rtol=0, atol=1e-12):
                raise ValueError(f"Saved scenario geometry changed for seed {seed}")
        if (selection["source_episode"], selection["source_route"]) != (
            row["source_episode"], row["selected_route"]
        ):
            raise ValueError(f"Saved route selection changed for seed {seed}")
        record, _ = _run_episode(
            config.phase5a, execution, demos, geometry, seed, index,
            _policy_factory(policy, input_normalizer, output_normalizer),
        )
        records.append(_public_record(record))
    successes = sum(record["phase5a_gate_success"] for record in records)
    return {
        "schema_version": 1,
        "purpose": "Saved-checkpoint simulation smoke test, not a new benchmark",
        "private_frame_data_required": False,
        "episodes": len(records),
        "safe_successes": successes,
        "passed": successes == len(records),
        "checkpoint_sha256": sha256(root / CHECKPOINT),
        "motion_prior_sha256": manifest["archive_sha256"],
        "rollouts": records,
    }
