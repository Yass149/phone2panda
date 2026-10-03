from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from phone2panda.evaluation.phase6 import (
    load_phase6_config,
    select_balanced_demonstration_subsets,
)
from phone2panda.policy.gru import GRUPolicy


def test_phase6_demonstration_subsets_are_balanced_and_fixed() -> None:
    config = load_phase6_config(Path("configs/phase6.yaml"))
    subsets = select_balanced_demonstration_subsets(config)
    with Path("data/metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata = {row["episode_id"]: row for row in csv.DictReader(handle)}

    assert subsets["5"] == ["ep_001", "ep_002", "ep_003", "ep_004", "ep_005"]
    for size in (5, 15, 30):
        selected = subsets[str(size)]
        starts = Counter(metadata[episode]["start_id"] for episode in selected)
        routes = Counter(metadata[episode]["route"] for episode in selected)
        assert max(starts.values()) - min(starts.values()) <= 1
        assert max(routes.values()) - min(routes.values()) <= 1


def test_phase6_float32_checkpoint_preserves_dtype_and_predictions(tmp_path: Path) -> None:
    source = Path("results/phase5b/gru_policy.npz")
    policy64, input64, output64, metadata = GRUPolicy.load(source)
    policy32, input32, output32, _ = GRUPolicy.load(source, dtype=np.float32)
    destination = tmp_path / "float32.npz"
    policy32.save(destination, input32, output32, metadata)
    loaded, loaded_input, loaded_output, _ = GRUPolicy.load(
        destination, dtype=np.float32
    )
    features = np.zeros(policy64.input_dim, dtype=np.float64)
    expected, _ = policy64.step(input64.transform(features), policy64.initial_state())
    actual, _ = loaded.step(
        loaded_input.transform(features.astype(np.float32)), loaded.initial_state()
    )

    assert loaded.parameters["Wz"].dtype == np.float32
    assert loaded_input.mean.dtype == np.float32
    assert loaded_output.mean.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
