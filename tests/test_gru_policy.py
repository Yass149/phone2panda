from __future__ import annotations

from pathlib import Path

import numpy as np

from phone2panda.policy.gru import GRUPolicy, Normalizer, pad_sequences


def test_normalizer_round_trip_and_constant_feature() -> None:
    values = np.asarray([[1.0, 4.0], [3.0, 4.0]], dtype=np.float64)
    normalizer = Normalizer.fit(values, minimum_scale=0.01)
    transformed = normalizer.transform(values)

    assert normalizer.scale[1] == 0.01
    np.testing.assert_allclose(normalizer.inverse(transformed), values)


def test_sequence_batching_preserves_episode_boundaries() -> None:
    inputs = [np.ones((2, 3)), np.full((4, 3), 2.0)]
    targets = [np.ones((2, 2)), np.full((4, 2), 3.0)]
    batch = pad_sequences(inputs, targets)

    np.testing.assert_array_equal(batch.lengths, [2, 4])
    np.testing.assert_array_equal(batch.mask, [[True, True, False, False], [True] * 4])
    np.testing.assert_allclose(batch.inputs[0, 2:], 0.0)


def test_checkpoint_reload_preserves_policy_output(tmp_path: Path) -> None:
    policy = GRUPolicy(input_dim=3, hidden_dim=4, output_dim=2, seed=7)
    input_normalizer = Normalizer.fit(np.asarray([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]))
    output_normalizer = Normalizer.fit(np.asarray([[-1.0, 0.5], [1.0, 0.7]]))
    inputs = np.asarray([0.2, -0.3, 0.4])
    expected, _ = policy.step(inputs, policy.initial_state())
    checkpoint = tmp_path / "policy.npz"

    policy.save(checkpoint, input_normalizer, output_normalizer, {"name": "test"})
    loaded, loaded_input, loaded_output, metadata = GRUPolicy.load(checkpoint)
    actual, _ = loaded.step(inputs, loaded.initial_state())

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(loaded_input.mean, input_normalizer.mean)
    np.testing.assert_allclose(loaded_output.scale, output_normalizer.scale)
    assert metadata == {"name": "test"}
