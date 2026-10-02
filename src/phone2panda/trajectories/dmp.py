from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DMPModel:
    weights: FloatArray
    centers: FloatArray
    widths: FloatArray
    start: FloatArray
    goal: FloatArray
    alpha_z: float = 25.0
    beta_z: float = 6.25
    alpha_s: float = 4.0


def fit_dmp(path: FloatArray, basis_functions: int = 40) -> DMPModel:
    """Fit a stable discrete DMP to a two-dimensional path."""
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 5:
        raise ValueError("DMP path must have shape (timesteps, dimensions)")
    times = np.linspace(0.0, 1.0, len(values))
    dt = times[1] - times[0]
    velocity = np.gradient(values, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    alpha_z, beta_z, alpha_s = 25.0, 6.25, 4.0
    phase = np.exp(-alpha_s * times)
    centers = np.exp(-alpha_s * np.linspace(0.0, 1.0, basis_functions))
    widths = np.full(basis_functions, basis_functions**1.5 / centers)
    activations = np.exp(-widths[None, :] * (phase[:, None] - centers[None, :]) ** 2)
    design = activations / np.maximum(activations.sum(axis=1, keepdims=True), 1e-12)
    design *= phase[:, None]
    target_forcing = acceleration - alpha_z * (
        beta_z * (values[-1] - values) - velocity
    )
    weights = np.linalg.lstsq(design, target_forcing, rcond=1e-8)[0].T
    return DMPModel(
        weights=weights,
        centers=centers,
        widths=widths,
        start=values[0].copy(),
        goal=values[-1].copy(),
    )


def rollout_dmp(model: DMPModel, samples: int) -> FloatArray:
    if samples < 2:
        raise ValueError("DMP rollout requires at least two samples")
    dt = 1.0 / (samples - 1)
    position = model.start.copy()
    velocity = np.zeros_like(position)
    output = np.empty((samples, len(position)), dtype=np.float64)
    for index in range(samples):
        output[index] = position
        phase = np.exp(-model.alpha_s * index * dt)
        activation = np.exp(-model.widths * (phase - model.centers) ** 2)
        forcing = (activation @ model.weights.T) / max(float(activation.sum()), 1e-12)
        forcing *= phase
        acceleration = model.alpha_z * (
            model.beta_z * (model.goal - position) - velocity
        ) + forcing
        velocity += acceleration * dt
        position += velocity * dt
    output[0] = model.start
    output[-1] = model.goal
    return output
