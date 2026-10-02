from __future__ import annotations

import csv
import json
import logging
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phone2panda.pilot_validation.artifacts import (
    write_json,
    write_overlay_video,
    write_trajectory_plot,
)
from phone2panda.pilot_validation.config import ValidationConfig, load_config
from phone2panda.pilot_validation.geometry import (
    Rect,
    classify_route,
    homography_from_corners,
    path_length,
    point_in_rect,
    signed_distance_to_rect,
)
from phone2panda.pilot_validation.video import (
    estimate_dropped_frames,
    inspect_video,
    iter_frames,
    sha256_file,
)
from phone2panda.pilot_validation.vision import (
    CORNER_ORDER,
    CornerDetection,
    ObjectCandidate,
    corner_observation,
    corner_reference,
    detect_corner_markers,
    detect_object_candidates,
    detect_obstacle,
    temporal_median,
    track_object,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetEpisode:
    episode_id: str
    video_file: str
    start_id: str
    target_id: str
    expected_route: str
    split: str
    notes: str

    @property
    def expected_start(self) -> str:
        return self.start_id


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    schema_version: int
    base: ValidationConfig
    metadata_csv: Path
    raw_dir: Path
    processed_dir: Path
    report_dir: Path
    opening_closing_frames: int
    max_camera_drift_px: float
    representative_accepted_overlays: int


@dataclass
class EpisodeRuntime:
    episode: DatasetEpisode
    source: Path
    report: dict[str, Any]
    fps: Any
    homography: np.ndarray
    corner_reference: dict[str, CornerDetection]
    corner_direct: list[dict[str, bool]]
    selected: list[ObjectCandidate | None]
    points: np.ndarray
    interpolated: list[bool]
    confidences: list[float]
    obstacle: Rect
    start_zone: Rect
    target_zone: Rect


def load_dataset_config(path: Path) -> DatasetConfig:
    config_path = path.resolve()
    root = config_path.parent.parent
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = load_config(root / raw["base_validation_config"])
    base = replace(
        base,
        detection={**base.detection, **raw.get("detection_overrides", {})},
    )
    return DatasetConfig(
        root=root,
        schema_version=int(raw["schema_version"]),
        base=base,
        metadata_csv=root / raw["metadata_csv"],
        raw_dir=root / raw["raw_dir"],
        processed_dir=root / raw["processed_dir"],
        report_dir=root / raw["report_dir"],
        opening_closing_frames=int(raw["calibration"]["opening_closing_frames"]),
        max_camera_drift_px=float(raw["calibration"]["max_camera_drift_px"]),
        representative_accepted_overlays=int(
            raw["artifacts"]["representative_accepted_overlays"]
        ),
    )


def load_episodes(metadata_csv: Path) -> list[DatasetEpisode]:
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    episodes = [
        DatasetEpisode(
            episode_id=row["episode_id"].strip(),
            video_file=row["video_file"].strip(),
            start_id=row["start_id"].strip(),
            target_id=row["target_id"].strip(),
            expected_route=row["route"].strip(),
            split=row["split"].strip(),
            notes=row["notes"].strip(),
        )
        for row in rows
    ]
    return episodes


def inventory_mismatches(config: DatasetConfig, episodes: list[DatasetEpisode]) -> list[str]:
    expected = [episode.video_file for episode in episodes]
    actual_paths = sorted(config.raw_dir.glob("*.MOV"))
    actual = [path.name for path in actual_paths]
    superseded_originals: set[str] = set()
    for name in expected:
        stem = name.removesuffix(".MOV")
        if "_redo" not in stem:
            continue
        base, version_text = stem.rsplit("_redo", 1)
        superseded_originals.add(base + ".MOV")
        if version_text.isdigit():
            for version in range(1, int(version_text)):
                suffix = "_redo" if version == 1 else f"_redo{version}"
                superseded_originals.add(base + suffix + ".MOV")
    problems: list[str] = []
    problems.extend(f"missing recording: {name}" for name in sorted(set(expected) - set(actual)))
    problems.extend(
        f"unlisted recording: {name}"
        for name in sorted(set(actual) - set(expected) - superseded_originals)
    )
    problems.extend(
        f"duplicate metadata filename: {name}"
        for name, count in sorted(Counter(expected).items())
        if count > 1
    )
    problems.extend(
        f"duplicate episode_id: {name}"
        for name, count in sorted(Counter(ep.episode_id for ep in episodes).items())
        if count > 1
    )
    problems.extend(
        f"empty recording: {path.name}"
        for path in actual_paths
        if path.stat().st_size == 0
    )
    if len(episodes) != 36:
        problems.append(f"metadata row count: expected 36, found {len(episodes)}")
    return problems


def _rect(values: list[float]) -> Rect:
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _round_or_none(value: float, digits: int = 6) -> float | None:
    return _round(value, digits) if np.isfinite(value) else None


def _scan_frames(
    path: Path, declared_frames: int, sample_count: int, window_size: int
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    stride = max(1, declared_frames // max(1, sample_count - 1)) if declared_frames else 15
    timestamps: list[float] = []
    samples: list[np.ndarray] = []
    opening: list[np.ndarray] = []
    closing: deque[np.ndarray] = deque(maxlen=window_size)
    last_frame: np.ndarray | None = None
    last_index = -1
    for index, timestamp, frame in iter_frames(path):
        timestamps.append(timestamp)
        if len(opening) < window_size:
            opening.append(frame)
        closing.append(frame)
        if index % stride == 0 and len(samples) < sample_count:
            samples.append(frame)
        last_frame = frame
        last_index = index
    if last_frame is not None and (last_index % stride or not samples):
        samples.append(last_frame)
    return timestamps, samples, opening, list(closing)


def _is_upright(reference: dict[str, CornerDetection]) -> bool:
    points = {name: detection.point for name, detection in reference.items()}
    return max(points["tl"][1], points["tr"][1]) < min(
        points["bl"][1], points["br"][1]
    ) and max(points["tl"][0], points["bl"][0]) < min(
        points["tr"][0], points["br"][0]
    )


def _candidate_confidences(selected: list[ObjectCandidate | None]) -> list[float]:
    areas = [candidate.area for candidate in selected if candidate is not None]
    median_area = float(np.median(areas)) if areas else 1.0
    confidences: list[float] = []
    for candidate in selected:
        if candidate is None:
            confidences.append(0.0)
            continue
        consistency = float(np.exp(-abs(np.log(max(candidate.area, 1.0) / median_area))))
        confidences.append(0.5 * consistency + 0.5 * min(1.0, candidate.solidity / 0.85))
    return confidences


def _smooth_points(points: np.ndarray) -> np.ndarray:
    output = points.copy()
    weights = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0])
    for index in range(len(points)):
        left = max(0, index - 2)
        right = min(len(points), index + 3)
        values = points[left:right]
        valid = np.isfinite(values).all(axis=1)
        if valid.any():
            local_weights = weights[2 - (index - left) : 2 + (right - index)][valid]
            output[index] = np.average(values[valid], axis=0, weights=local_weights)
    return output


def _zone_ratio(points: np.ndarray, indices: np.ndarray, rect: Rect, margin: float = 0.0) -> float:
    valid = [index for index in indices if np.isfinite(points[index]).all()]
    if not valid:
        return 0.0
    return float(
        np.mean(
            [
                point_in_rect((float(points[index, 0]), float(points[index, 1])), rect, margin)
                for index in valid
            ]
        )
    )


def _gate(
    config: DatasetConfig,
    upright: bool,
    opening_coverage: float,
    closing_coverage: float,
    marker_jitter_px: float,
    camera_drift_px: float,
    tracking_coverage: float,
    dropped_frames: int,
    start_ratio: float,
    start_footprint_ratio: float,
    final_ratio: float,
    final_footprint_ratio: float,
    expected_route: str,
    observed_route: str,
    footprint_clearance: float,
) -> dict[str, Any]:
    validation = config.base.validation
    checks = {
        "orientation": (upright, "decoded marker order is not upright landscape"),
        "calibration_start_window": (
            opening_coverage >= float(validation["min_calibration_window_coverage"]),
            "corner_markers_missing_during_initial_calibration",
        ),
        "calibration_end_window": (
            closing_coverage >= float(validation["min_calibration_window_coverage"]),
            "corner_markers_missing_during_final_calibration_check",
        ),
        "calibration_marker_jitter": (
            marker_jitter_px <= float(validation["max_calibration_jitter_rms_px"]),
            "corner_marker_jitter_above_threshold",
        ),
        "camera_drift": (
            camera_drift_px <= config.max_camera_drift_px,
            "camera_or_canvas_moved_during_recording",
        ),
        "red_marker_tracking": (
            tracking_coverage >= float(validation["min_object_tracking_coverage"]),
            "red_marker_tracking_below_threshold",
        ),
        "decoded_frames": (
            dropped_frames <= int(validation["max_decode_dropped_frames"]),
            "decoded_frame_gap_detected",
        ),
        "start_position": (
            start_ratio >= float(validation["min_zone_hold_ratio"]),
            "object_not_held_in_expected_start_zone",
        ),
        "start_footprint": (
            start_footprint_ratio >= float(validation["min_footprint_hold_ratio"]),
            "object_footprint_not_inside_start_zone",
        ),
        "route": (
            observed_route == expected_route,
            "observed_route_does_not_match_label",
        ),
        "obstacle_clearance": (
            footprint_clearance
            >= float(validation["min_obstacle_footprint_clearance"]),
            "object_footprint_clearance_below_threshold",
        ),
        "final_placement": (
            final_ratio >= float(validation["min_zone_hold_ratio"]),
            "object_not_held_in_target_zone",
        ),
        "final_footprint": (
            final_footprint_ratio >= float(validation["min_footprint_hold_ratio"]),
            "object_footprint_not_inside_target_zone",
        ),
    }
    criteria = {name: passed for name, (passed, _) in checks.items()}
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "failure_reasons": [reason for passed, reason in checks.values() if not passed],
    }


def _write_trajectory_csv(
    path: Path,
    episode: DatasetEpisode,
    timestamps: list[float],
    corner_direct: list[dict[str, bool]],
    selected: list[ObjectCandidate | None],
    points: np.ndarray,
    smoothed: np.ndarray,
    interpolated: list[bool],
    confidences: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_id",
        "split",
        "frame_index",
        "timestamp_seconds",
        "object_x",
        "object_y",
        "smoothed_x",
        "smoothed_y",
        "tracking_confidence",
        "object_detected",
        "object_interpolated",
        "all_corners_direct",
        "route",
        "valid",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            valid = bool(np.isfinite(points[index]).all())
            writer.writerow(
                {
                    "episode_id": episode.episode_id,
                    "split": episode.split,
                    "frame_index": index,
                    "timestamp_seconds": f"{timestamp:.6f}",
                    "object_x": f"{points[index, 0]:.8f}" if valid else "",
                    "object_y": f"{points[index, 1]:.8f}" if valid else "",
                    "smoothed_x": f"{smoothed[index, 0]:.8f}" if valid else "",
                    "smoothed_y": f"{smoothed[index, 1]:.8f}" if valid else "",
                    "tracking_confidence": f"{confidences[index]:.6f}",
                    "object_detected": int(selected[index] is not None),
                    "object_interpolated": int(interpolated[index]),
                    "all_corners_direct": int(all(corner_direct[index].values())),
                    "route": episode.expected_route,
                    "valid": int(valid),
                }
            )


def _process_episode(config: DatasetConfig, episode: DatasetEpisode) -> EpisodeRuntime:
    source = config.raw_dir / episode.video_file
    metadata = inspect_video(source)
    timestamps, samples, opening_frames, closing_frames = _scan_frames(
        source,
        metadata.declared_frames,
        int(config.base.detection["median_sample_count"]),
        config.opening_closing_frames,
    )
    if not timestamps:
        raise ValueError(f"No video frames decoded from {source.name}")
    median_frame = temporal_median(samples)
    opening_reference = corner_reference(temporal_median(opening_frames), config.base.detection)
    closing_detections = detect_corner_markers(
        temporal_median(closing_frames), config.base.detection
    )
    global_reference = opening_reference
    upright = _is_upright(opening_reference)
    homography = homography_from_corners(
        {name: global_reference[name].point for name in CORNER_ORDER}
    )
    obstacle, obstacle_confidence = detect_obstacle(
        median_frame, homography, config.base.detection
    )

    corner_direct: list[dict[str, bool]] = []
    corner_points: list[dict[str, tuple[float, float] | None]] = []
    candidates_by_frame: list[list[ObjectCandidate]] = []
    tolerance = float(config.base.detection["corner_reference_tolerance_px"])
    for _, _, frame in iter_frames(source):
        detected_corners = detect_corner_markers(frame, config.base.detection)
        direct: dict[str, bool] = {}
        point_row: dict[str, tuple[float, float] | None] = {}
        for name in CORNER_ORDER:
            is_direct, _, _ = corner_observation(
                detected_corners[name], global_reference[name], tolerance
            )
            direct[name] = is_direct
            point_row[name] = (
                detected_corners[name].point
                if is_direct and detected_corners[name] is not None
                else None
            )
        corner_direct.append(direct)
        corner_points.append(point_row)
        candidates_by_frame.append(
            detect_object_candidates(frame, homography, config.base.detection)
        )

    start_zone = _rect(config.base.geometry["start_zones"][episode.start_id])
    target_zone = _rect(config.base.geometry["target_zone"])
    selected, points, interpolated = track_object(
        candidates_by_frame,
        start_zone,
        float(config.base.detection["max_tracking_step"]),
        int(config.base.detection["max_interpolation_gap_frames"]),
    )
    smoothed = _smooth_points(points)
    confidences = _candidate_confidences(selected)
    frame_count = len(timestamps)
    if len(candidates_by_frame) != frame_count:
        raise RuntimeError(f"Frame count changed between decode passes for {source.name}")
    all_corners = np.asarray([all(row.values()) for row in corner_direct])
    corner_coverage = float(all_corners.mean())
    tracking_coverage = sum(candidate is not None for candidate in selected) / frame_count
    dropped_frames, gap_after = estimate_dropped_frames(timestamps, metadata.fps)
    times = np.asarray(timestamps)
    hold_seconds = float(config.base.validation["hold_seconds"])
    start_indices = np.flatnonzero(times <= times[0] + hold_seconds)
    final_indices = np.flatnonzero(times >= times[-1] - hold_seconds)
    opening_coverage = float(all_corners[start_indices].mean())
    closing_coverage = float(all_corners[final_indices].mean())
    marker_jitter_by_marker: dict[str, float] = {}
    for name in CORNER_ORDER:
        window_jitter: list[float] = []
        for indices in (start_indices, final_indices):
            window_points = [
                corner_points[index][name]
                for index in indices
                if corner_points[index][name] is not None
            ]
            if not window_points:
                window_jitter.append(float("inf"))
                continue
            values = np.asarray(window_points)
            median = np.median(values, axis=0)
            distances = np.linalg.norm(values - median, axis=1)
            window_jitter.append(float(np.sqrt(np.mean(np.square(distances)))))
        marker_jitter_by_marker[name] = max(window_jitter)
    marker_jitter_px = max(marker_jitter_by_marker.values())
    drift_by_marker: dict[str, float] = {}
    for name in CORNER_ORDER:
        opening_points = [
            corner_points[index][name]
            for index in start_indices
            if corner_points[index][name] is not None
        ]
        closing_points = [
            corner_points[index][name]
            for index in final_indices
            if corner_points[index][name] is not None
        ]
        if not opening_points or not closing_points:
            drift_by_marker[name] = float("inf")
            continue
        opening_median = np.median(np.asarray(opening_points), axis=0)
        closing_median = np.median(np.asarray(closing_points), axis=0)
        drift_by_marker[name] = float(np.linalg.norm(closing_median - opening_median))
    camera_drift_px = max(drift_by_marker.values())
    object_radius = float(config.base.geometry["object_footprint_radius"])
    start_ratio = _zone_ratio(points, start_indices, start_zone)
    start_footprint_ratio = _zone_ratio(points, start_indices, start_zone, object_radius)
    final_ratio = _zone_ratio(points, final_indices, target_zone)
    final_footprint_ratio = _zone_ratio(points, final_indices, target_zone, object_radius)
    observed_route, crossing_x, route_samples = classify_route(
        points,
        obstacle,
        float(config.base.geometry["route_vertical_margin"]),
    )
    valid_points = points[np.isfinite(points).all(axis=1)]
    clearances = [
        signed_distance_to_rect((float(point[0]), float(point[1])), obstacle)
        for point in valid_points
    ]
    center_clearance = min(clearances) if clearances else float("-inf")
    footprint_clearance = center_clearance - object_radius
    gate = _gate(
        config,
        upright,
        opening_coverage,
        closing_coverage,
        marker_jitter_px,
        camera_drift_px,
        tracking_coverage,
        dropped_frames,
        start_ratio,
        start_footprint_ratio,
        final_ratio,
        final_footprint_ratio,
        episode.expected_route,
        observed_route,
        footprint_clearance,
    )
    finite_confidences = [value for value in confidences if value > 0]
    trajectory_rel = Path("trajectories") / f"{episode.episode_id}.csv"
    _write_trajectory_csv(
        config.processed_dir / trajectory_rel,
        episode,
        timestamps,
        corner_direct,
        selected,
        points,
        smoothed,
        interpolated,
        confidences,
    )
    report: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "video_file": episode.video_file,
        "split": episode.split,
        "expected_start": episode.start_id,
        "expected_route": episode.expected_route,
        "accepted": gate["passed"],
        "gate": gate,
        "video": {
            "codec": metadata.codec,
            "width": metadata.width,
            "height": metadata.height,
            "fps": _round(metadata.fps),
            "decoded_frames": frame_count,
            "duration_seconds": _round(timestamps[-1] - timestamps[0] + 1 / metadata.fps),
            "decode_dropped_frames": dropped_frames,
            "timestamp_gap_after_frames": gap_after,
        },
        "calibration": {
            "orientation_basis": "decoded_marker_positions",
            "decoded_orientation": "upright_landscape" if upright else "invalid",
            "rotation_applied_degrees": 0,
            "opening_markers": {
                name: [_round(value, 3) for value in opening_reference[name].point]
                for name in CORNER_ORDER
            },
            "closing_markers": {
                name: (
                    [_round(value, 3) for value in closing_detections[name].point]
                    if closing_detections[name] is not None
                    else None
                )
                for name in CORNER_ORDER
            },
            "drift_px_by_marker": {
                name: _round_or_none(value, 3)
                for name, value in drift_by_marker.items()
            },
            "maximum_camera_drift_px": _round_or_none(camera_drift_px, 3),
            "opening_window_coverage": _round(opening_coverage),
            "closing_window_coverage": _round(closing_coverage),
            "marker_jitter_rms_px_by_marker": {
                name: _round_or_none(value, 3)
                for name, value in marker_jitter_by_marker.items()
            },
            "maximum_marker_jitter_rms_px": _round_or_none(marker_jitter_px, 3),
            "all_four_direct_coverage": _round(corner_coverage),
            "per_marker_direct_coverage": {
                name: _round(float(np.mean([row[name] for row in corner_direct])))
                for name in CORNER_ORDER
            },
        },
        "tracking": {
            "detected_coverage": _round(tracking_coverage),
            "usable_coverage": _round(float(np.isfinite(points).all(axis=1).mean())),
            "interpolated_frames": sum(interpolated),
            "mean_confidence": _round(
                float(np.mean(finite_confidences)) if finite_confidences else 0.0
            ),
        },
        "task": {
            "start_center_hold_ratio": _round(start_ratio),
            "start_footprint_hold_ratio": _round(start_footprint_ratio),
            "observed_route": observed_route,
            "route_crossing_median_x": _round(crossing_x) if np.isfinite(crossing_x) else None,
            "route_sample_count": route_samples,
            "obstacle_rect": [_round(value) for value in obstacle],
            "obstacle_detection_confidence": _round(obstacle_confidence),
            "minimum_center_clearance": _round_or_none(center_clearance),
            "minimum_footprint_clearance": _round_or_none(footprint_clearance),
            "final_center_hold_ratio": _round(final_ratio),
            "final_footprint_hold_ratio": _round(final_footprint_ratio),
            "normalised_path_length": _round(path_length(points)),
        },
        "processed_trajectory": str(Path("data/processed/v1") / trajectory_rel),
    }
    return EpisodeRuntime(
        episode=episode,
        source=source,
        report=report,
        fps=metadata.average_rate,
        homography=homography,
        corner_reference=global_reference,
        corner_direct=corner_direct,
        selected=selected,
        points=points,
        interpolated=interpolated,
        confidences=confidences,
        obstacle=obstacle,
        start_zone=start_zone,
        target_zone=target_zone,
    )


def _representative_accepted(runtimes: list[EpisodeRuntime], limit: int) -> list[EpisodeRuntime]:
    if limit <= 0:
        return []
    accepted = [runtime for runtime in runtimes if runtime.report["accepted"]]
    selected: list[EpisodeRuntime] = []
    covered: set[tuple[str, str]] = set()
    for runtime in accepted:
        key = (runtime.episode.start_id, runtime.episode.expected_route)
        if key not in covered:
            selected.append(runtime)
            covered.add(key)
        if len(selected) == limit:
            break
    selected_ids = {runtime.episode.episode_id for runtime in selected}
    for runtime in accepted:
        if runtime.episode.episode_id not in selected_ids:
            selected.append(runtime)
        if len(selected) == limit:
            break
    return selected


def _write_artifacts(config: DatasetConfig, runtimes: list[EpisodeRuntime]) -> list[dict[str, str]]:
    rejected = [runtime for runtime in runtimes if not runtime.report["accepted"]]
    representative = _representative_accepted(
        runtimes, config.representative_accepted_overlays
    )
    chosen = [(runtime, "rejected") for runtime in rejected]
    chosen.extend((runtime, "representative_accepted") for runtime in representative)
    artifacts: list[dict[str, str]] = []
    for runtime, reason in chosen:
        episode_id = runtime.episode.episode_id
        overlay_rel = Path("overlays") / f"{episode_id}.mp4"
        plot_rel = Path("plots") / f"{episode_id}.png"
        (config.report_dir / overlay_rel).parent.mkdir(parents=True, exist_ok=True)
        (config.report_dir / plot_rel).parent.mkdir(parents=True, exist_ok=True)
        write_overlay_video(
            config.report_dir / overlay_rel,
            runtime.source,
            runtime.fps,
            (int(config.base.overlay["width"]), int(config.base.overlay["height"])),
            int(config.base.overlay["crf"]),
            runtime.homography,
            runtime.corner_reference,
            runtime.corner_direct,
            runtime.selected,
            runtime.points,
            runtime.interpolated,
            runtime.confidences,
            runtime.obstacle,
            runtime.start_zone,
            runtime.target_zone,
            runtime.episode.expected_route,
            runtime.report["task"]["observed_route"],
            runtime.report["accepted"],
        )
        write_trajectory_plot(
            config.report_dir / plot_rel,
            runtime.episode,  # type: ignore[arg-type]
            runtime.points,
            runtime.obstacle,
            runtime.start_zone,
            runtime.target_zone,
            float(config.base.geometry["object_footprint_radius"]),
            runtime.report["task"]["observed_route"],
            runtime.report["task"]["minimum_footprint_clearance"]
            if runtime.report["task"]["minimum_footprint_clearance"] is not None
            else float("-inf"),
        )
        artifacts.append(
            {
                "episode_id": episode_id,
                "selection_reason": reason,
                "overlay": str(overlay_rel),
                "trajectory_plot": str(plot_rel),
            }
        )
    return artifacts


def _write_summary_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    fields = [
        "episode_id",
        "video_file",
        "split",
        "accepted",
        "corner_coverage",
        "tracking_coverage",
        "camera_drift_px",
        "observed_route",
        "minimum_footprint_clearance",
        "failure_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "episode_id": report["episode_id"],
                    "video_file": report["video_file"],
                    "split": report["split"],
                    "accepted": report["accepted"],
                    "corner_coverage": report["calibration"]["all_four_direct_coverage"],
                    "tracking_coverage": report["tracking"]["detected_coverage"],
                    "camera_drift_px": report["calibration"]["maximum_camera_drift_px"],
                    "observed_route": report["task"]["observed_route"],
                    "minimum_footprint_clearance": report["task"][
                        "minimum_footprint_clearance"
                    ],
                    "failure_reasons": " | ".join(report["gate"]["failure_reasons"]),
                }
            )


def run_dataset_validation(
    config: DatasetConfig, episode_ids: set[str] | None = None
) -> dict[str, Any]:
    episodes = load_episodes(config.metadata_csv)
    mismatches = inventory_mismatches(config, episodes)
    if mismatches:
        return {"inventory_passed": False, "mismatches": mismatches}
    selected_ids = episode_ids or {episode.episode_id for episode in episodes}
    known_ids = {episode.episode_id for episode in episodes}
    unknown_ids = sorted(selected_ids - known_ids)
    if unknown_ids:
        return {
            "inventory_passed": False,
            "mismatches": [f"unknown episode_id: {name}" for name in unknown_ids],
        }
    selected_episodes = [
        episode for episode in episodes if episode.episode_id in selected_ids
    ]
    partial_run = len(selected_episodes) != len(episodes)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    sources = [config.raw_dir / episode.video_file for episode in selected_episodes]
    hashes_before = {path.name: sha256_file(path) for path in sources}
    runtimes: list[EpisodeRuntime] = []
    for index, episode in enumerate(selected_episodes, start=1):
        LOGGER.info("[%d/%d] %s", index, len(selected_episodes), episode.video_file)
        runtimes.append(_process_episode(config, episode))
    hashes_after = {path.name: sha256_file(path) for path in sources}
    changed = [name for name in hashes_before if hashes_before[name] != hashes_after[name]]
    if changed:
        raise RuntimeError(f"Raw recordings changed during processing: {changed}")
    artifact_config = (
        replace(config, representative_accepted_overlays=0) if partial_run else config
    )
    new_artifacts = _write_artifacts(artifact_config, runtimes)
    new_reports = {runtime.episode.episode_id: runtime.report for runtime in runtimes}
    previous: dict[str, Any] = {}
    report_path = config.report_dir / "quality_report.json"
    if partial_run:
        if not report_path.exists():
            raise RuntimeError("A subset run requires an existing aggregate quality report")
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous_reports = {
            report["episode_id"]: report for report in previous["episodes"]
        }
        missing_previous = sorted(known_ids - set(previous_reports))
        if missing_previous:
            raise RuntimeError(
                "Existing aggregate report is missing episodes: "
                + ", ".join(missing_previous)
            )
        reports = [
            new_reports.get(episode.episode_id, previous_reports[episode.episode_id])
            for episode in episodes
        ]
        artifacts = [
            artifact
            for artifact in previous.get("representative_artifacts", [])
            if artifact["episode_id"] not in selected_ids
        ] + new_artifacts
        rejected_artifacts = [
            artifact
            for artifact in artifacts
            if artifact["selection_reason"] == "rejected"
        ]
        accepted_artifacts = [
            artifact
            for artifact in artifacts
            if artifact["selection_reason"] == "representative_accepted"
        ][: config.representative_accepted_overlays]
        artifacts = rejected_artifacts + accepted_artifacts
    else:
        reports = [new_reports[episode.episode_id] for episode in episodes]
        artifacts = new_artifacts
    split_counts: dict[str, dict[str, int]] = {}
    for split in sorted({episode.split for episode in episodes}):
        split_reports = [report for report in reports if report["split"] == split]
        accepted = sum(bool(report["accepted"]) for report in split_reports)
        split_counts[split] = {
            "total": len(split_reports),
            "accepted": accepted,
            "rejected": len(split_reports) - accepted,
        }
    accepted_count = sum(bool(report["accepted"]) for report in reports)
    previous_manifest_path = config.report_dir / "source_manifest.json"
    previous_recordings: list[dict[str, Any]] = []
    if partial_run and previous_manifest_path.exists():
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        previous_recordings = previous_manifest.get("recordings", [])
    manifest_records = {
        recording["file"]: recording for recording in previous_recordings
    }
    manifest_records.update(
        {
            path.name: {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256_before": hashes_before[path.name],
                "sha256_after": hashes_after[path.name],
                "preserved": True,
            }
            for path in sources
        }
    )
    manifest = {
        "schema_version": config.schema_version,
        "recording_count": len(manifest_records),
        "all_raw_recordings_preserved": True,
        "recordings": [manifest_records[name] for name in sorted(manifest_records)],
    }
    write_json(config.report_dir / "source_manifest.json", manifest)
    aggregate = {
        "schema_version": config.schema_version,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "phase": "phase_3_dataset_quality",
        "run_scope": "subset" if partial_run else "full_dataset",
        "revalidated_episode_ids": sorted(selected_ids),
        "inventory_passed": True,
        "dataset_gate_passed": accepted_count == len(reports),
        "total_recordings": len(reports),
        "accepted_count": accepted_count,
        "rejected_count": len(reports) - accepted_count,
        "split_counts": split_counts,
        "thresholds": {
            **config.base.validation,
            "max_camera_drift_px": config.max_camera_drift_px,
        },
        "all_raw_recordings_preserved": True,
        "source_manifest": "source_manifest.json",
        "representative_artifacts": artifacts,
        "episodes": reports,
    }
    write_json(config.report_dir / "quality_report.json", aggregate)
    _write_summary_csv(config.report_dir / "quality_report.csv", reports)
    return aggregate
