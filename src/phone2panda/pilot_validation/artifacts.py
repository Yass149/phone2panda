from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from .config import PilotSpec
from .geometry import Rect, transform_points
from .video import iter_frames
from .vision import CornerDetection, ObjectCandidate

matplotlib.use("Agg")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_frames_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty frame table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    columns = [
        "episode_id",
        "passed",
        "frame_count",
        "duration_seconds",
        "corner_direct_coverage",
        "calibration_start_coverage",
        "calibration_end_coverage",
        "calibration_max_jitter_px",
        "calibration_max_drift_px",
        "calibration_coverage",
        "object_tracking_coverage",
        "mean_tracking_confidence",
        "decode_dropped_frames",
        "expected_route",
        "observed_route",
        "start_hold_ratio",
        "final_hold_ratio",
        "minimum_center_clearance",
        "minimum_footprint_clearance",
        "failure_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "episode_id": report["episode_id"],
                    "passed": report["gate"]["passed"],
                    "frame_count": report["video"]["decoded_frames"],
                    "duration_seconds": report["video"]["duration_seconds"],
                    "corner_direct_coverage": report["calibration"]["all_four_direct_coverage"],
                    "calibration_start_coverage": report["calibration"][
                        "start_window_direct_coverage"
                    ],
                    "calibration_end_coverage": report["calibration"]["end_window_direct_coverage"],
                    "calibration_max_jitter_px": report["calibration"][
                        "maximum_marker_jitter_rms_px"
                    ],
                    "calibration_max_drift_px": report["calibration"][
                        "maximum_start_end_marker_drift_px"
                    ],
                    "calibration_coverage": report["calibration"]["usable_coverage"],
                    "object_tracking_coverage": report["tracking"]["detected_coverage"],
                    "mean_tracking_confidence": report["tracking"]["mean_confidence"],
                    "decode_dropped_frames": report["video"]["decode_dropped_frames"],
                    "expected_route": report["task"]["expected_route"],
                    "observed_route": report["task"]["observed_route"],
                    "start_hold_ratio": report["task"]["start_center_hold_ratio"],
                    "final_hold_ratio": report["task"]["final_center_hold_ratio"],
                    "minimum_center_clearance": report["task"]["minimum_center_clearance"],
                    "minimum_footprint_clearance": report["task"]["minimum_footprint_clearance"],
                    "failure_reasons": "|".join(report["gate"]["failure_reasons"]),
                }
            )


def write_trajectory_plot(
    path: Path,
    spec: PilotSpec,
    points: NDArray[np.float64],
    obstacle: Rect,
    start_zone: Rect,
    target_zone: Rect,
    object_radius: float,
    observed_route: str,
    minimum_clearance: float,
) -> None:
    valid = np.isfinite(points).all(axis=1)
    trajectory = points[valid]
    figure, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    axis.add_patch(
        Rectangle(
            (start_zone[0], start_zone[1]),
            start_zone[2] - start_zone[0],
            start_zone[3] - start_zone[1],
            facecolor="#4c78a8",
            edgecolor="#2f4b6c",
            alpha=0.16,
            label=f"expected start ({spec.expected_start})",
        )
    )
    axis.add_patch(
        Rectangle(
            (target_zone[0], target_zone[1]),
            target_zone[2] - target_zone[0],
            target_zone[3] - target_zone[1],
            facecolor="#59a14f",
            edgecolor="#2f6b2a",
            alpha=0.18,
            label="target",
        )
    )
    axis.add_patch(
        Rectangle(
            (obstacle[0], obstacle[1]),
            obstacle[2] - obstacle[0],
            obstacle[3] - obstacle[1],
            facecolor="#4e79a7",
            edgecolor="#17365d",
            alpha=0.72,
            label="detected obstacle",
        )
    )
    axis.add_patch(
        Rectangle(
            (obstacle[0] - object_radius, obstacle[1] - object_radius),
            obstacle[2] - obstacle[0] + 2 * object_radius,
            obstacle[3] - obstacle[1] + 2 * object_radius,
            fill=False,
            edgecolor="#e15759",
            linestyle="--",
            linewidth=1.4,
            label="obstacle + box footprint",
        )
    )
    if len(trajectory):
        colour = np.linspace(0.0, 1.0, len(trajectory))
        axis.scatter(trajectory[:, 0], trajectory[:, 1], c=colour, cmap="plasma", s=8, alpha=0.75)
        axis.plot(trajectory[:, 0], trajectory[:, 1], color="#6f2dbd", linewidth=1.2, alpha=0.7)
        axis.scatter(*trajectory[0], color="#1b9e77", marker="o", s=80, zorder=5, label="start")
        axis.scatter(*trajectory[-1], color="#d95f02", marker="X", s=90, zorder=5, label="final")
    axis.set(
        xlim=(-0.03, 1.03),
        ylim=(1.03, -0.03),
        xlabel="normalised canvas x",
        ylabel="normalised canvas y (top = 0)",
        title=(
            f"{spec.episode_id}: expected {spec.expected_route}, observed {observed_route}\n"
            f"minimum footprint clearance = {minimum_clearance:.3f} canvas units"
        ),
        aspect="equal",
    )
    axis.grid(alpha=0.18)
    axis.legend(loc="lower right", fontsize=8)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _project_rect(rect: Rect, inverse_homography: NDArray[np.float64]) -> NDArray[np.int32]:
    normalised = [
        (rect[0], rect[1]),
        (rect[2], rect[1]),
        (rect[2], rect[3]),
        (rect[0], rect[3]),
    ]
    return np.rint(transform_points(normalised, inverse_homography)).astype(np.int32)


def _outlined_text(
    image: NDArray[np.uint8], text: str, origin: tuple[int, int], scale: float = 0.68
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
    )


def write_overlay_video(
    path: Path,
    source: Path,
    fps: Fraction,
    size: tuple[int, int],
    crf: int,
    homography: NDArray[np.float64],
    corner_reference: dict[str, CornerDetection],
    corner_direct: list[dict[str, bool]],
    selected: list[ObjectCandidate | None],
    points: NDArray[np.float64],
    interpolated: list[bool],
    confidences: list[float],
    obstacle: Rect,
    start_zone: Rect,
    target_zone: Rect,
    expected_route: str,
    observed_route: str,
    gate_passed: bool,
) -> None:
    width, height = size
    inverse = np.linalg.inv(homography)
    canvas_polygon = np.rint(transform_points([(0, 0), (1, 0), (1, 1), (0, 1)], inverse)).astype(
        np.int32
    )
    start_polygon = _project_rect(start_zone, inverse)
    target_polygon = _project_rect(target_zone, inverse)
    obstacle_polygon = _project_rect(obstacle, inverse)
    trail: list[tuple[int, int]] = []

    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium", "movflags": "+faststart"}
        for index, timestamp, frame in iter_frames(source):
            cv2.polylines(frame, [canvas_polygon], True, (255, 180, 0), 3, cv2.LINE_AA)
            cv2.polylines(frame, [start_polygon], True, (255, 140, 30), 3, cv2.LINE_AA)
            cv2.polylines(frame, [target_polygon], True, (40, 180, 40), 3, cv2.LINE_AA)
            cv2.polylines(frame, [obstacle_polygon], True, (255, 80, 40), 3, cv2.LINE_AA)
            for name, reference in corner_reference.items():
                direct = corner_direct[index][name]
                colour = (20, 200, 20) if direct else (0, 165, 255)
                center = tuple(np.rint(reference.point).astype(int))
                cv2.circle(frame, center, 12, colour, 3, cv2.LINE_AA)
                _outlined_text(frame, name.upper(), (center[0] + 14, center[1] - 8), 0.55)

            if np.isfinite(points[index]).all():
                pixel = transform_points([tuple(points[index])], inverse)[0]
                center = tuple(np.rint(pixel).astype(int))
                trail.append(center)
                if len(trail) > 180:
                    trail.pop(0)
                if len(trail) > 1:
                    cv2.polylines(
                        frame,
                        [np.asarray(trail, dtype=np.int32)],
                        False,
                        (180, 50, 220),
                        4,
                        cv2.LINE_AA,
                    )
                colour = (0, 220, 255) if interpolated[index] else (0, 0, 255)
                cv2.circle(frame, center, 13, colour, -1, cv2.LINE_AA)

            detected = selected[index] is not None
            tracking_status = "DETECTED" if detected else "MISSING"
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            _outlined_text(resized, f"t={timestamp:5.2f}s  frame={index}", (24, 32))
            _outlined_text(
                resized,
                f"red marker: {tracking_status}  confidence={confidences[index]:.2f}",
                (24, 62),
            )
            _outlined_text(
                resized,
                f"route: expected={expected_route} observed={observed_route}",
                (24, 92),
            )
            gate_colour = (20, 180, 20) if gate_passed else (0, 0, 220)
            cv2.putText(
                resized,
                f"PILOT GATE: {'PASS' if gate_passed else 'FAIL'}",
                (24, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                resized,
                f"PILOT GATE: {'PASS' if gate_passed else 'FAIL'}",
                (24, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                gate_colour,
                2,
                cv2.LINE_AA,
            )
            video_frame = av.VideoFrame.from_ndarray(resized, format="bgr24")
            for packet in stream.encode(video_frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
