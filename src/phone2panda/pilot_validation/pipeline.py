from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .artifacts import (
    write_frames_csv,
    write_json,
    write_overlay_video,
    write_summary_csv,
    write_trajectory_plot,
)
from .config import PilotSpec, ValidationConfig
from .geometry import (
    Rect,
    classify_route,
    homography_from_corners,
    path_length,
    point_in_rect,
    signed_distance_to_rect,
)
from .video import (
    collect_timestamps_and_samples,
    estimate_dropped_frames,
    inspect_video,
    iter_frames,
    sha256_file,
)
from .vision import (
    CORNER_ORDER,
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


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _rect(values: list[float]) -> Rect:
    if len(values) != 4:
        raise ValueError(f"Rectangle must have four values, got {values}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _longest_missing_run(selected: list[ObjectCandidate | None]) -> int:
    longest = current = 0
    for candidate in selected:
        current = current + 1 if candidate is None else 0
        longest = max(longest, current)
    return longest


def _candidate_confidences(selected: list[ObjectCandidate | None]) -> list[float]:
    areas = [candidate.area for candidate in selected if candidate is not None]
    median_area = float(np.median(areas)) if areas else 1.0
    output: list[float] = []
    for candidate in selected:
        if candidate is None:
            output.append(0.0)
            continue
        area_consistency = float(np.exp(-abs(np.log(max(candidate.area, 1.0) / median_area))))
        solidity_score = min(1.0, candidate.solidity / 0.85)
        output.append(0.5 * area_consistency + 0.5 * solidity_score)
    return output


def _zone_ratio(points: np.ndarray, indices: np.ndarray, rect: Rect, margin: float = 0.0) -> float:
    if not len(indices):
        return 0.0
    matches = [
        point_in_rect((float(points[index, 0]), float(points[index, 1])), rect, margin)
        for index in indices
        if np.isfinite(points[index]).all()
    ]
    return float(np.mean(matches)) if matches else 0.0


def _gate(
    config: ValidationConfig,
    calibration_start_coverage: float,
    calibration_end_coverage: float,
    calibration_max_jitter_px: float,
    calibration_max_drift_px: float,
    object_coverage: float,
    dropped_frames: int,
    start_ratio: float,
    start_footprint_ratio: float,
    final_ratio: float,
    final_footprint_ratio: float,
    expected_route: str,
    observed_route: str,
    footprint_clearance: float,
) -> dict[str, Any]:
    criteria = {
        "calibration_start_window": calibration_start_coverage
        >= float(config.validation["min_calibration_window_coverage"]),
        "calibration_end_window": calibration_end_coverage
        >= float(config.validation["min_calibration_window_coverage"]),
        "calibration_marker_jitter": calibration_max_jitter_px
        <= float(config.validation["max_calibration_jitter_rms_px"]),
        "calibration_start_end_drift": calibration_max_drift_px
        <= float(config.validation["max_calibration_start_end_drift_px"]),
        "object_tracking_coverage": object_coverage
        >= float(config.validation["min_object_tracking_coverage"]),
        "decode_dropped_frames": dropped_frames
        <= int(config.validation["max_decode_dropped_frames"]),
        "start_zone": start_ratio >= float(config.validation["min_zone_hold_ratio"]),
        "start_footprint": start_footprint_ratio
        >= float(config.validation["min_footprint_hold_ratio"]),
        "route": observed_route == expected_route,
        "obstacle_clearance": footprint_clearance
        >= float(config.validation["min_obstacle_footprint_clearance"]),
        "final_target": final_ratio >= float(config.validation["min_zone_hold_ratio"]),
        "final_footprint": final_footprint_ratio
        >= float(config.validation["min_footprint_hold_ratio"]),
    }
    failure_messages = {
        "calibration_start_window": "corner_markers_missing_during_initial_calibration",
        "calibration_end_window": "corner_markers_missing_during_final_calibration_check",
        "calibration_marker_jitter": "corner_marker_jitter_above_threshold",
        "calibration_start_end_drift": "camera_or_canvas_moved_during_recording",
        "object_tracking_coverage": "red_marker_tracking_below_threshold",
        "decode_dropped_frames": "decoded_frame_gap_detected",
        "start_zone": "object_not_held_in_expected_start_zone",
        "start_footprint": "object_footprint_not_inside_start_zone",
        "route": "observed_route_does_not_match_label",
        "obstacle_clearance": "object_footprint_clearance_below_threshold",
        "final_target": "object_not_held_in_target_zone",
        "final_footprint": "object_footprint_not_inside_target_zone",
    }
    failures = [failure_messages[name] for name, passed in criteria.items() if not passed]
    return {"passed": all(criteria.values()), "criteria": criteria, "failure_reasons": failures}


def _process_pilot(config: ValidationConfig, spec: PilotSpec) -> dict[str, Any]:
    LOGGER.info("Processing %s", spec.file.name)
    metadata = inspect_video(spec.file)
    source_hash_before = sha256_file(spec.file)
    timestamps, samples = collect_timestamps_and_samples(
        spec.file, metadata.declared_frames, int(config.detection["median_sample_count"])
    )
    if not timestamps:
        raise ValueError(f"No video frames decoded from {spec.file.name}")
    median_frame = temporal_median(samples)
    references = corner_reference(median_frame, config.detection)
    reference_points = {name: references[name].point for name in CORNER_ORDER}
    top_markers_above_bottom = max(reference_points["tl"][1], reference_points["tr"][1]) < min(
        reference_points["bl"][1], reference_points["br"][1]
    )
    left_markers_left_of_right = max(reference_points["tl"][0], reference_points["bl"][0]) < min(
        reference_points["tr"][0], reference_points["br"][0]
    )
    decoded_upright_landscape = top_markers_above_bottom and left_markers_left_of_right
    if not decoded_upright_landscape:
        raise ValueError(
            f"Decoded marker order is not upright landscape for {spec.file.name}; "
            "refusing to infer a rotation from container metadata"
        )
    homography = homography_from_corners(reference_points)
    obstacle, obstacle_confidence = detect_obstacle(median_frame, homography, config.detection)

    corner_direct: list[dict[str, bool]] = []
    corner_points: list[dict[str, tuple[float, float] | None]] = []
    corner_confidences: list[dict[str, float]] = []
    corner_distances: dict[str, list[float]] = {name: [] for name in CORNER_ORDER}
    candidates_by_frame: list[list[ObjectCandidate]] = []
    tolerance = float(config.detection["corner_reference_tolerance_px"])
    for _, _, frame in iter_frames(spec.file):
        detections = detect_corner_markers(frame, config.detection)
        direct_row: dict[str, bool] = {}
        point_row: dict[str, tuple[float, float] | None] = {}
        confidence_row: dict[str, float] = {}
        for name in CORNER_ORDER:
            direct, confidence, distance = corner_observation(
                detections[name], references[name], tolerance
            )
            direct_row[name] = direct
            point_row[name] = detections[name].point if direct and detections[name] else None
            confidence_row[name] = confidence
            if np.isfinite(distance) and direct:
                corner_distances[name].append(distance)
        corner_direct.append(direct_row)
        corner_points.append(point_row)
        corner_confidences.append(confidence_row)
        candidates_by_frame.append(detect_object_candidates(frame, homography, config.detection))

    frame_count = len(timestamps)
    if len(candidates_by_frame) != frame_count:
        raise RuntimeError("Video frame count changed between decoding passes")
    start_zone = _rect(config.geometry["start_zones"][spec.expected_start])
    target_zone = _rect(config.geometry["target_zone"])
    selected, points, interpolated = track_object(
        candidates_by_frame,
        start_zone,
        float(config.detection["max_tracking_step"]),
        int(config.detection["max_interpolation_gap_frames"]),
    )
    confidences = _candidate_confidences(selected)
    detected_count = sum(candidate is not None for candidate in selected)
    interpolated_count = sum(interpolated)
    object_coverage = detected_count / frame_count
    usable_tracking_coverage = float(np.isfinite(points).all(axis=1).mean())
    all_four = np.asarray([all(row.values()) for row in corner_direct], dtype=bool)
    per_corner_coverage = {
        name: float(np.mean([row[name] for row in corner_direct])) for name in CORNER_ORDER
    }
    corner_coverage = float(all_four.mean())
    dropped_frames, timestamp_gap_after = estimate_dropped_frames(timestamps, metadata.fps)

    hold_seconds = float(config.validation["hold_seconds"])
    time_values = np.asarray(timestamps)
    start_indices = np.flatnonzero(time_values <= time_values[0] + hold_seconds)
    final_indices = np.flatnonzero(time_values >= time_values[-1] - hold_seconds)
    calibration_start_coverage = float(all_four[start_indices].mean())
    calibration_end_coverage = float(all_four[final_indices].mean())
    marker_jitter_rms = {
        name: (
            float(np.sqrt(np.mean(np.square(corner_distances[name]))))
            if corner_distances[name]
            else float("inf")
        )
        for name in CORNER_ORDER
    }
    start_end_marker_drift: dict[str, float] = {}
    for name in CORNER_ORDER:
        start_points = [
            corner_points[index][name]
            for index in start_indices
            if corner_points[index][name] is not None
        ]
        end_points = [
            corner_points[index][name]
            for index in final_indices
            if corner_points[index][name] is not None
        ]
        if not start_points or not end_points:
            start_end_marker_drift[name] = float("inf")
            continue
        start_median = np.median(np.asarray(start_points), axis=0)
        end_median = np.median(np.asarray(end_points), axis=0)
        start_end_marker_drift[name] = float(np.linalg.norm(start_median - end_median))
    calibration_max_jitter = max(marker_jitter_rms.values())
    calibration_max_drift = max(start_end_marker_drift.values())
    object_radius = float(config.geometry["object_footprint_radius"])
    start_ratio = _zone_ratio(points, start_indices, start_zone)
    start_footprint_ratio = _zone_ratio(points, start_indices, start_zone, object_radius)
    final_ratio = _zone_ratio(points, final_indices, target_zone)
    final_footprint_ratio = _zone_ratio(points, final_indices, target_zone, object_radius)
    observed_route, crossing_x, route_sample_count = classify_route(
        points, obstacle, float(config.geometry["route_vertical_margin"])
    )
    valid_points = points[np.isfinite(points).all(axis=1)]
    center_clearances = [
        signed_distance_to_rect((float(point[0]), float(point[1])), obstacle)
        for point in valid_points
    ]
    minimum_center_clearance = min(center_clearances) if center_clearances else float("-inf")
    minimum_footprint_clearance = minimum_center_clearance - object_radius
    gate = _gate(
        config,
        calibration_start_coverage,
        calibration_end_coverage,
        calibration_max_jitter,
        calibration_max_drift,
        object_coverage,
        dropped_frames,
        start_ratio,
        start_footprint_ratio,
        final_ratio,
        final_footprint_ratio,
        spec.expected_route,
        observed_route,
        minimum_footprint_clearance,
    )

    pilot_dir = config.output_dir / spec.episode_id
    pilot_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    for index in range(frame_count):
        candidate = selected[index]
        frame_rows.append(
            {
                "frame_index": index,
                "timestamp_seconds": f"{timestamps[index]:.6f}",
                "corner_tl_direct": int(corner_direct[index]["tl"]),
                "corner_tr_direct": int(corner_direct[index]["tr"]),
                "corner_bl_direct": int(corner_direct[index]["bl"]),
                "corner_br_direct": int(corner_direct[index]["br"]),
                "all_corners_direct": int(all_four[index]),
                "object_detected": int(candidate is not None),
                "object_interpolated": int(interpolated[index]),
                "object_x": "" if not np.isfinite(points[index, 0]) else f"{points[index, 0]:.8f}",
                "object_y": "" if not np.isfinite(points[index, 1]) else f"{points[index, 1]:.8f}",
                "tracking_confidence": f"{confidences[index]:.6f}",
            }
        )
    write_frames_csv(pilot_dir / "frames.csv", frame_rows)
    write_trajectory_plot(
        pilot_dir / "trajectory.png",
        spec,
        points,
        obstacle,
        start_zone,
        target_zone,
        object_radius,
        observed_route,
        minimum_footprint_clearance,
    )
    write_overlay_video(
        pilot_dir / "overlay.mp4",
        spec.file,
        metadata.average_rate,
        (int(config.overlay["width"]), int(config.overlay["height"])),
        int(config.overlay["crf"]),
        homography,
        references,
        corner_direct,
        selected,
        points,
        interpolated,
        confidences,
        obstacle,
        start_zone,
        target_zone,
        spec.expected_route,
        observed_route,
        bool(gate["passed"]),
    )
    source_hash_after = sha256_file(spec.file)
    if source_hash_before != source_hash_after:
        raise RuntimeError(f"Original video changed while processing: {spec.file.name}")

    finite_confidences = [value for value in confidences if value > 0]
    report: dict[str, Any] = {
        "schema_version": config.schema_version,
        "episode_id": spec.episode_id,
        "source": {
            "file": spec.file.name,
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "original_preserved": source_hash_before == source_hash_after,
        },
        "video": {
            "codec": metadata.codec,
            "width": metadata.width,
            "height": metadata.height,
            "fps": _round(metadata.fps),
            "duration_seconds": _round(timestamps[-1] - timestamps[0] + 1.0 / metadata.fps),
            "declared_frames": metadata.declared_frames,
            "decoded_frames": frame_count,
            "decode_dropped_frames": dropped_frames,
            "timestamp_gap_after_frames": timestamp_gap_after,
            "source_has_audio": metadata.has_audio,
            "overlay_has_audio": False,
        },
        "calibration": {
            "method": "four_custom_L_markers_with_static_camera_fallback",
            "orientation_basis": "decoded_marker_positions",
            "decoded_orientation": "upright_landscape",
            "rotation_applied_degrees": 0,
            "reference_marker_pixels": {
                name: [_round(value, 3) for value in references[name].point]
                for name in CORNER_ORDER
            },
            "homography_pixel_to_normalised": [
                [_round(value, 10) for value in row] for row in homography
            ],
            "per_marker_direct_coverage": {
                name: _round(per_corner_coverage[name]) for name in CORNER_ORDER
            },
            "all_four_direct_coverage": _round(corner_coverage),
            "start_window_direct_coverage": _round(calibration_start_coverage),
            "end_window_direct_coverage": _round(calibration_end_coverage),
            "usable_coverage": 1.0,
            "static_fallback_frames": int((~all_four).sum()),
            "marker_jitter_rms_px": {
                name: _round(marker_jitter_rms[name], 3) for name in CORNER_ORDER
            },
            "maximum_marker_jitter_rms_px": _round(calibration_max_jitter, 3),
            "start_end_marker_drift_px": {
                name: _round(start_end_marker_drift[name], 3) for name in CORNER_ORDER
            },
            "maximum_start_end_marker_drift_px": _round(calibration_max_drift, 3),
        },
        "tracking": {
            "method": "HSV_red_segmentation_with_continuity_filter",
            "detected_frames": detected_count,
            "interpolated_frames": interpolated_count,
            "missing_frames_after_interpolation": int((~np.isfinite(points).all(axis=1)).sum()),
            "detected_coverage": _round(object_coverage),
            "usable_coverage": _round(usable_tracking_coverage),
            "mean_confidence": _round(
                float(np.mean(finite_confidences)) if finite_confidences else 0.0
            ),
            "minimum_confidence": _round(min(finite_confidences) if finite_confidences else 0.0),
            "longest_raw_gap_frames": _longest_missing_run(selected),
        },
        "obstacle": {
            "normalised_rect": [_round(value) for value in obstacle],
            "detection_confidence": _round(obstacle_confidence),
        },
        "task": {
            "expected_start": spec.expected_start,
            "start_zone": list(start_zone),
            "start_center_hold_ratio": _round(start_ratio),
            "start_footprint_hold_ratio": _round(start_footprint_ratio),
            "target_zone": list(target_zone),
            "final_center_hold_ratio": _round(final_ratio),
            "final_footprint_hold_ratio": _round(final_footprint_ratio),
            "expected_route": spec.expected_route,
            "observed_route": observed_route,
            "route_crossing_median_x": _round(crossing_x) if np.isfinite(crossing_x) else None,
            "route_sample_count": route_sample_count,
            "minimum_center_clearance": _round(minimum_center_clearance),
            "object_footprint_radius": object_radius,
            "minimum_footprint_clearance": _round(minimum_footprint_clearance),
            "normalised_path_length": _round(path_length(points)),
        },
        "gate": gate,
        "artifacts": {
            "overlay": f"{spec.episode_id}/overlay.mp4",
            "trajectory_plot": f"{spec.episode_id}/trajectory.png",
            "frame_table": f"{spec.episode_id}/frames.csv",
        },
    }
    write_json(pilot_dir / "quality.json", report)
    return report


def run_validation(config: ValidationConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_originals = sorted(config.root.glob("*.MOV"))
    hashes_before = {path.name: sha256_file(path) for path in all_originals}
    reports = [_process_pilot(config, spec) for spec in config.videos]
    hashes_after = {path.name: sha256_file(path) for path in all_originals}
    manifest_entries = [
        {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256_before": hashes_before[path.name],
            "sha256_after": hashes_after[path.name],
            "original_preserved": hashes_before[path.name] == hashes_after[path.name],
            "selected_for_validation": path in {spec.file for spec in config.videos},
        }
        for path in all_originals
    ]
    if not all(entry["original_preserved"] for entry in manifest_entries):
        changed = [entry["file"] for entry in manifest_entries if not entry["original_preserved"]]
        raise RuntimeError(f"Original recordings changed during validation: {changed}")
    write_json(
        config.output_dir / "source_manifest.json",
        {
            "schema_version": config.schema_version,
            "recording_count": len(manifest_entries),
            "all_originals_preserved": True,
            "recordings": manifest_entries,
        },
    )
    aggregate_passed = all(report["gate"]["passed"] for report in reports)
    aggregate = {
        "schema_version": config.schema_version,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "milestone": "five_pilot_validation",
        "passed": aggregate_passed,
        "passed_count": sum(report["gate"]["passed"] for report in reports),
        "pilot_count": len(reports),
        "source_manifest": "source_manifest.json",
        "all_originals_preserved": True,
        "thresholds": config.validation,
        "pilots": reports,
        "recording_changes_required": []
        if aggregate_passed
        else [
            (
                "Keep all four corner markers visible during the opening and closing "
                "hands-free calibration windows."
            ),
            (
                "Keep the phone and canvas fixed for the entire recording; temporary marker "
                "occlusion during manipulation is supported by the locked calibration."
            ),
            (
                "Keep the phone and canvas fixed; retain the current even lighting and red "
                "marker, which track reliably."
            ),
        ],
    }
    write_json(config.output_dir / "quality_report.json", aggregate)
    write_summary_csv(config.output_dir / "quality_report.csv", reports)
    return aggregate
