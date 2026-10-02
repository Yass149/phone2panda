from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _sigmoid(value: FloatArray) -> FloatArray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class Normalizer:
    mean: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, values: FloatArray, minimum_scale: float = 1e-6) -> Normalizer:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("Normalizer requires a non-empty two-dimensional array")
        mean = np.mean(array, axis=0)
        scale = np.maximum(np.std(array, axis=0), float(minimum_scale))
        return cls(mean=mean, scale=scale)

    def transform(self, values: FloatArray) -> FloatArray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse(self, values: FloatArray) -> FloatArray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


@dataclass(frozen=True)
class SequenceBatch:
    inputs: FloatArray
    targets: FloatArray
    mask: NDArray[np.bool_]
    lengths: NDArray[np.int64]


def pad_sequences(
    inputs: list[FloatArray], targets: list[FloatArray]
) -> SequenceBatch:
    if not inputs or len(inputs) != len(targets):
        raise ValueError("Input and target sequence lists must be non-empty and aligned")
    lengths = np.asarray([len(sequence) for sequence in inputs], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("Sequences must not be empty")
    input_dim = int(np.asarray(inputs[0]).shape[1])
    target_dim = int(np.asarray(targets[0]).shape[1])
    maximum = int(np.max(lengths))
    padded_inputs = np.zeros((len(inputs), maximum, input_dim), dtype=np.float64)
    padded_targets = np.zeros((len(inputs), maximum, target_dim), dtype=np.float64)
    mask = np.zeros((len(inputs), maximum), dtype=np.bool_)
    for index, (input_sequence, target_sequence) in enumerate(zip(inputs, targets, strict=True)):
        x = np.asarray(input_sequence, dtype=np.float64)
        y = np.asarray(target_sequence, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("Every input and target sequence must be aligned and two-dimensional")
        if x.shape[1] != input_dim or y.shape[1] != target_dim:
            raise ValueError("All sequences must share feature dimensions")
        length = len(x)
        padded_inputs[index, :length] = x
        padded_targets[index, :length] = y
        mask[index, :length] = True
    return SequenceBatch(padded_inputs, padded_targets, mask, lengths)


class GRUPolicy:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        rng = np.random.default_rng(seed)

        def weight(rows: int, columns: int) -> FloatArray:
            limit = np.sqrt(6.0 / (rows + columns))
            return rng.uniform(-limit, limit, size=(rows, columns))

        self.parameters: dict[str, FloatArray] = {
            "Wz": weight(self.input_dim, self.hidden_dim),
            "Uz": weight(self.hidden_dim, self.hidden_dim),
            "bz": np.zeros(self.hidden_dim, dtype=np.float64),
            "Wr": weight(self.input_dim, self.hidden_dim),
            "Ur": weight(self.hidden_dim, self.hidden_dim),
            "br": np.zeros(self.hidden_dim, dtype=np.float64),
            "Wn": weight(self.input_dim, self.hidden_dim),
            "Un": weight(self.hidden_dim, self.hidden_dim),
            "bn": np.zeros(self.hidden_dim, dtype=np.float64),
            "Wy": weight(self.hidden_dim, self.output_dim),
            "by": np.zeros(self.output_dim, dtype=np.float64),
        }

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.parameters.values())

    def initial_state(self, batch_size: int = 1) -> FloatArray:
        return np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

    def step(self, inputs: FloatArray, hidden: FloatArray) -> tuple[FloatArray, FloatArray]:
        x = np.asarray(inputs, dtype=np.float64)
        h_previous = np.asarray(hidden, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if h_previous.ndim == 1:
            h_previous = h_previous[None, :]
        p = self.parameters
        z = _sigmoid(x @ p["Wz"] + h_previous @ p["Uz"] + p["bz"])
        r = _sigmoid(x @ p["Wr"] + h_previous @ p["Ur"] + p["br"])
        candidate = np.tanh(x @ p["Wn"] + (r * h_previous) @ p["Un"] + p["bn"])
        h = (1.0 - z) * candidate + z * h_previous
        output = h @ p["Wy"] + p["by"]
        return output, h

    def _forward(self, batch: SequenceBatch) -> tuple[FloatArray, list[dict[str, FloatArray]]]:
        x = batch.inputs
        batch_size, steps, _ = x.shape
        outputs = np.zeros((batch_size, steps, self.output_dim), dtype=np.float64)
        h = self.initial_state(batch_size)
        cache: list[dict[str, FloatArray]] = []
        p = self.parameters
        for step in range(steps):
            current = x[:, step]
            previous = h
            z = _sigmoid(current @ p["Wz"] + previous @ p["Uz"] + p["bz"])
            r = _sigmoid(current @ p["Wr"] + previous @ p["Ur"] + p["br"])
            candidate = np.tanh(
                current @ p["Wn"] + (r * previous) @ p["Un"] + p["bn"]
            )
            proposed = (1.0 - z) * candidate + z * previous
            active = batch.mask[:, step, None]
            h = np.where(active, proposed, previous)
            outputs[:, step] = h @ p["Wy"] + p["by"]
            cache.append(
                {
                    "x": current,
                    "h_previous": previous,
                    "z": z,
                    "r": r,
                    "candidate": candidate,
                    "active": active.astype(np.float64),
                    "h": h,
                }
            )
        return outputs, cache

    def loss_and_gradients(
        self, batch: SequenceBatch
    ) -> tuple[float, dict[str, FloatArray]]:
        predictions, cache = self._forward(batch)
        active = batch.mask[:, :, None].astype(np.float64)
        difference = (predictions - batch.targets) * active
        denominator = float(np.sum(batch.mask) * self.output_dim)
        loss = float(np.sum(difference * difference) / denominator)
        output_gradient = 2.0 * difference / denominator
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        p = self.parameters
        hidden_gradient = np.zeros((batch.inputs.shape[0], self.hidden_dim), dtype=np.float64)
        for step in range(batch.inputs.shape[1] - 1, -1, -1):
            item = cache[step]
            dy = output_gradient[:, step]
            gradients["Wy"] += item["h"].T @ dy
            gradients["by"] += np.sum(dy, axis=0)
            dh = dy @ p["Wy"].T + hidden_gradient
            active_step = item["active"]
            dh_active = dh * active_step
            hidden_gradient = dh * (1.0 - active_step)

            candidate_gradient = dh_active * (1.0 - item["z"])
            update_gradient = dh_active * (item["h_previous"] - item["candidate"])
            hidden_gradient += dh_active * item["z"]

            candidate_pre = candidate_gradient * (1.0 - item["candidate"] ** 2)
            recurrent_candidate = item["r"] * item["h_previous"]
            gradients["Wn"] += item["x"].T @ candidate_pre
            gradients["Un"] += recurrent_candidate.T @ candidate_pre
            gradients["bn"] += np.sum(candidate_pre, axis=0)
            recurrent_candidate_gradient = candidate_pre @ p["Un"].T
            reset_gradient = recurrent_candidate_gradient * item["h_previous"]
            hidden_gradient += recurrent_candidate_gradient * item["r"]

            reset_pre = reset_gradient * item["r"] * (1.0 - item["r"])
            gradients["Wr"] += item["x"].T @ reset_pre
            gradients["Ur"] += item["h_previous"].T @ reset_pre
            gradients["br"] += np.sum(reset_pre, axis=0)
            hidden_gradient += reset_pre @ p["Ur"].T

            update_pre = update_gradient * item["z"] * (1.0 - item["z"])
            gradients["Wz"] += item["x"].T @ update_pre
            gradients["Uz"] += item["h_previous"].T @ update_pre
            gradients["bz"] += np.sum(update_pre, axis=0)
            hidden_gradient += update_pre @ p["Uz"].T
        return loss, gradients

    def fit(
        self,
        batch: SequenceBatch,
        *,
        maximum_epochs: int,
        learning_rate: float,
        patience: int,
        minimum_improvement: float,
        target_loss: float,
        gradient_clip_norm: float,
    ) -> list[dict[str, float | int]]:
        first_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        second_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        best = {name: value.copy() for name, value in self.parameters.items()}
        best_loss = float("inf")
        stale_epochs = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, maximum_epochs + 1):
            loss, gradients = self.loss_and_gradients(batch)
            gradient_norm = float(
                np.sqrt(sum(np.sum(value * value) for value in gradients.values()))
            )
            if gradient_norm > gradient_clip_norm:
                scale = gradient_clip_norm / gradient_norm
                gradients = {name: value * scale for name, value in gradients.items()}
            for name, parameter in self.parameters.items():
                gradient = gradients[name]
                first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
                second_moment[name] = 0.999 * second_moment[name] + 0.001 * gradient * gradient
                first_unbiased = first_moment[name] / (1.0 - 0.9**epoch)
                second_unbiased = second_moment[name] / (1.0 - 0.999**epoch)
                parameter -= learning_rate * first_unbiased / (np.sqrt(second_unbiased) + 1e-8)
            history.append({"epoch": epoch, "loss": loss, "gradient_norm": gradient_norm})
            if best_loss - loss > minimum_improvement:
                best_loss = loss
                best = {name: value.copy() for name, value in self.parameters.items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if loss <= target_loss or stale_epochs >= patience:
                break
        self.parameters = best
        return history

    def save(
        self,
        path: Path,
        input_normalizer: Normalizer,
        output_normalizer: Normalizer,
        metadata: dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            **self.parameters,
            "input_mean": input_normalizer.mean,
            "input_scale": input_normalizer.scale,
            "output_mean": output_normalizer.mean,
            "output_scale": output_normalizer.scale,
            "architecture": np.asarray(
                [self.input_dim, self.hidden_dim, self.output_dim], dtype=np.int64
            ),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(
        cls, path: Path
    ) -> tuple[GRUPolicy, Normalizer, Normalizer, dict[str, Any]]:
        with np.load(path, allow_pickle=False) as payload:
            input_dim, hidden_dim, output_dim = (
                int(value) for value in payload["architecture"]
            )
            policy = cls(input_dim, hidden_dim, output_dim, seed=0)
            for name in policy.parameters:
                policy.parameters[name] = np.asarray(payload[name], dtype=np.float64)
            input_normalizer = Normalizer(
                np.asarray(payload["input_mean"], dtype=np.float64),
                np.asarray(payload["input_scale"], dtype=np.float64),
            )
            output_normalizer = Normalizer(
                np.asarray(payload["output_mean"], dtype=np.float64),
                np.asarray(payload["output_scale"], dtype=np.float64),
            )
            metadata = json.loads(str(payload["metadata_json"]))
        return policy, input_normalizer, output_normalizer, metadata
