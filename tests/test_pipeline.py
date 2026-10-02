from __future__ import annotations

from pathlib import Path

from phone2panda.pilot_validation.config import ValidationConfig
from phone2panda.pilot_validation.pipeline import _gate


def _config() -> ValidationConfig:
    return ValidationConfig(
        root=Path("."),
        schema_version=2,
        output_dir=Path("results"),
        videos=(),
        geometry={},
        detection={},
        validation={
            "min_calibration_window_coverage": 0.95,
            "max_calibration_jitter_rms_px": 3.0,
            "max_calibration_start_end_drift_px": 5.0,
            "min_object_tracking_coverage": 0.95,
            "max_decode_dropped_frames": 0,
            "min_zone_hold_ratio": 0.90,
            "min_footprint_hold_ratio": 0.80,
            "min_obstacle_footprint_clearance": 0.01,
        },
        overlay={},
    )


def test_gate_accepts_temporary_corner_occlusion_with_stable_calibration() -> None:
    result = _gate(
        _config(),
        calibration_start_coverage=1.0,
        calibration_end_coverage=1.0,
        calibration_max_jitter_px=1.7,
        calibration_max_drift_px=3.2,
        object_coverage=1.0,
        dropped_frames=0,
        start_ratio=1.0,
        start_footprint_ratio=1.0,
        final_ratio=1.0,
        final_footprint_ratio=1.0,
        expected_route="left",
        observed_route="left",
        footprint_clearance=0.12,
    )

    assert result["passed"]


def test_gate_rejects_camera_drift() -> None:
    result = _gate(
        _config(),
        calibration_start_coverage=1.0,
        calibration_end_coverage=1.0,
        calibration_max_jitter_px=1.0,
        calibration_max_drift_px=8.0,
        object_coverage=1.0,
        dropped_frames=0,
        start_ratio=1.0,
        start_footprint_ratio=1.0,
        final_ratio=1.0,
        final_footprint_ratio=1.0,
        expected_route="right",
        observed_route="right",
        footprint_clearance=0.12,
    )

    assert not result["passed"]
    assert result["failure_reasons"] == ["camera_or_canvas_moved_during_recording"]
