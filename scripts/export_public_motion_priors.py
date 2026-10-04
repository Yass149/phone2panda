#!/usr/bin/env python3
"""Export a derived DMP bundle locally without publishing recordings or frame data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phone2panda.evaluation.phase4b import build_library
from phone2panda.evaluation.phase4e import _execution_config
from phone2panda.evaluation.phase5b import load_phase5b_config
from phone2panda.public_demo import BUNDLE_DIRECTORY, CHECKPOINT, SPLIT_MANIFEST, sha256


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_phase5b_config(root / "configs/phase5b.yaml")
    execution, _ = _execution_config(config.phase5a.phase4e)
    demos, geometry = build_library(execution)
    directory = root / BUNDLE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "routes.npz"
    np.savez_compressed(
        archive,
        dmp_robot=np.asarray([demo.dmp_robot for demo in demos]),
        raw_endpoints=np.asarray([demo.raw_robot[[0, -1]] for demo in demos]),
        obstacle_xy=geometry["obstacle_xy"],
        obstacle_half_size=geometry["obstacle_half_size"],
    )
    inputs = [
        CHECKPOINT, SPLIT_MANIFEST,
        *(Path(f"configs/phase{phase}.yaml") for phase in ("4a", "4b", "4e", "5a", "5b")),
    ]
    manifest = {
        "schema_version": 1,
        "description": "80-point mapped DMP priors and endpoint anchors from 36 personal videos",
        "archive_sha256": sha256(archive),
        "source_dataset_report_sha256": sha256(execution.project_path("dataset_report")),
        "reproduction_sha256": {str(path): sha256(root / path) for path in inputs},
        "privacy": "No frames, audio, timestamps, hand imagery or frame-level trajectories",
        "supported_methods": ["dmp_route_confidence", "gru_reference_tracking"],
        "demonstrations": [
            {
                "episode_id": demo.episode_id, "start_id": demo.start_id,
                "route": demo.route, "split": demo.split,
                "coverage": demo.coverage, "confidence": demo.confidence,
            }
            for demo in demos
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(demos)} derived priors; archive {archive.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
