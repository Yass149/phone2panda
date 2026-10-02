from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VideoMetadata:
    codec: str
    width: int
    height: int
    fps: float
    average_rate: Fraction
    declared_frames: int
    duration_seconds: float
    has_audio: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path: Path) -> VideoMetadata:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or Fraction(30, 1)
        duration = 0.0
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        return VideoMetadata(
            codec=stream.codec_context.name,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=float(rate),
            average_rate=rate,
            declared_frames=int(stream.frames or 0),
            duration_seconds=duration,
            has_audio=bool(container.streams.audio),
        )


def iter_frames(path: Path) -> Iterator[tuple[int, float, NDArray[np.uint8]]]:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30.0)
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index / fps
            yield index, timestamp, frame.to_ndarray(format="bgr24")


def collect_timestamps_and_samples(
    path: Path, declared_frames: int, sample_count: int
) -> tuple[list[float], list[NDArray[np.uint8]]]:
    stride = max(1, declared_frames // max(sample_count - 1, 1)) if declared_frames else 15
    timestamps: list[float] = []
    samples: list[NDArray[np.uint8]] = []
    last_frame: NDArray[np.uint8] | None = None
    last_index = -1
    for index, timestamp, frame in iter_frames(path):
        timestamps.append(timestamp)
        last_frame = frame
        last_index = index
        if index % stride == 0 and len(samples) < sample_count:
            samples.append(frame)
    if last_frame is not None and (last_index % stride != 0 or not samples):
        samples.append(last_frame)
    return timestamps, samples


def estimate_dropped_frames(timestamps: list[float], fps: float) -> tuple[int, list[int]]:
    if len(timestamps) < 2:
        return 0, []
    dropped = 0
    gap_after: list[int] = []
    for index, (left, right) in enumerate(zip(timestamps[:-1], timestamps[1:], strict=True)):
        missing = max(0, round((right - left) * fps) - 1)
        if missing:
            dropped += missing
            gap_after.append(index)
    return dropped, gap_after
