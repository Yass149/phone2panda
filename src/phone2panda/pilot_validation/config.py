from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PilotSpec:
    file: Path
    episode_id: str
    expected_start: str
    expected_route: str


@dataclass(frozen=True)
class ValidationConfig:
    root: Path
    schema_version: int
    output_dir: Path
    videos: tuple[PilotSpec, ...]
    geometry: dict[str, Any]
    detection: dict[str, Any]
    validation: dict[str, Any]
    overlay: dict[str, Any]


def load_config(path: Path) -> ValidationConfig:
    config_path = path.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"schema_version", "output_dir", "videos", "geometry", "detection", "validation"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")

    root = config_path.parent.parent
    videos = tuple(
        PilotSpec(
            file=root / item["file"],
            episode_id=str(item["episode_id"]),
            expected_start=str(item["expected_start"]),
            expected_route=str(item["expected_route"]),
        )
        for item in raw["videos"]
    )
    if len(videos) != 5:
        raise ValueError(f"Pilot gate expects exactly five videos, got {len(videos)}")
    for pilot in videos:
        if not pilot.file.is_file():
            raise FileNotFoundError(pilot.file)
        if pilot.expected_route not in {"left", "right"}:
            raise ValueError(f"Unsupported route for {pilot.episode_id}: {pilot.expected_route}")

    return ValidationConfig(
        root=root,
        schema_version=int(raw["schema_version"]),
        output_dir=root / raw["output_dir"],
        videos=videos,
        geometry=raw["geometry"],
        detection=raw["detection"],
        validation=raw["validation"],
        overlay=raw.get("overlay", {"width": 1280, "height": 720, "crf": 23}),
    )
