from __future__ import annotations

from pathlib import Path

from phone2panda.evaluation.phase4b import build_library
from phone2panda.evaluation.phase4e import _execution_config
from phone2panda.evaluation.phase5b import build_balanced_manifest, load_phase5b_config


def test_phase5b_manifest_is_balanced_and_seed_disjoint() -> None:
    config = load_phase5b_config(Path("configs/phase5b.yaml"))
    execution, _ = _execution_config(config.phase5a.phase4e)
    demonstrations, geometry = build_library(execution)
    manifest = build_balanced_manifest(config, execution, demonstrations, geometry)

    assert manifest["summary"]["training"]["start_counts"] == {
        "s1": 40,
        "s2": 40,
        "s3": 40,
    }
    assert manifest["summary"]["training"]["route_counts"] == {
        "left": 60,
        "right": 60,
    }
    assert manifest["summary"]["validation"]["route_counts"] == {
        "left": 10,
        "right": 10,
    }
    assert manifest["summary"]["testing"]["route_counts"] == {
        "left": 25,
        "right": 25,
    }
    seed_sets = {
        name: {row["seed"] for row in rows}
        for name, rows in manifest["splits"].items()
    }
    assert seed_sets["training"].isdisjoint(seed_sets["validation"])
    assert seed_sets["training"].isdisjoint(seed_sets["testing"])
    assert seed_sets["validation"].isdisjoint(seed_sets["testing"])
