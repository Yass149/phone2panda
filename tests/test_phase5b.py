from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phone2panda.evaluation import phase5b
from phone2panda.evaluation.phase4b import Scenario
from phone2panda.evaluation.phase4e import _execution_config
from phone2panda.evaluation.phase5b import build_balanced_manifest, load_phase5b_config


def test_phase5b_manifest_is_balanced_and_seed_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_phase5b_config(Path("configs/phase5b.yaml"))
    execution, _ = _execution_config(config.phase5a.phase4e)

    def fake_scenario(_execution, _demos, _geometry, seed: int, index: int) -> Scenario:
        start_id = ("s1", "s2", "s3")[index % 3]
        return Scenario(
            index=index,
            seed=seed,
            start_id=start_id,
            start_xy=np.asarray([0.04, -0.18], dtype=np.float64),
            goal_xy=np.asarray([0.22, 0.14], dtype=np.float64),
            obstacle_xy=np.asarray([0.13, 0.0], dtype=np.float64),
            obstacle_half_size=np.asarray([0.035, 0.055], dtype=np.float64),
        )

    def fake_selection(_execution, _demos, scenario: Scenario, _method):
        route = (
            "left"
            if scenario.start_id == "s1" or (scenario.start_id == "s2" and scenario.seed % 2 == 0)
            else "right"
        )
        return np.zeros((2, 2), dtype=np.float64), {
            "source_route": route,
            "source_episode": f"synthetic_{route}",
            "source_confidence": 1.0,
        }

    monkeypatch.setattr(phase5b, "make_scenario", fake_scenario)
    monkeypatch.setattr(phase5b, "select_method_path", fake_selection)
    manifest = build_balanced_manifest(config, execution, [], {})

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
