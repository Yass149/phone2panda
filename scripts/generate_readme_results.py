#!/usr/bin/env python3
"""Generate the README result block directly from committed JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def mm(value_m: float) -> str:
    return f"{value_m * 1000:.2f} mm"


def row(label: str, metrics: dict) -> str:
    return (
        f"| {label} | {metrics['safe_successes']}/{metrics['episodes']} | "
        f"{metrics['object_collision_rollouts']} | "
        f"{metrics['unintended_contact_rollouts']} | {metrics['drop_rollouts']} | "
        f"{mm(metrics['placement_error_median_m'])} | "
        f"{mm(metrics['minimum_clearance_median_m'])} |"
    )


def condition(table: list[dict], name: str) -> dict:
    return next(item["metrics"] for item in table if item["condition"] == name)


def generated_block() -> str:
    dataset = load("results/dataset_quality/quality_report.json")
    source_manifest = load("results/dataset_quality/source_manifest.json")
    phase5b = load("results/phase5b/evaluation.json")
    phase6 = load("results/phase6/ablation_summary.json")
    precision = load("results/phase6/precision_benchmark.json")
    held_out = phase5b["held_out_evaluation"]
    table = phase6["table"]

    dmp = held_out["dmp_route_confidence"]
    gru = held_out["gru_policy"]
    straight = condition(table, "straight_line")
    calibrated = condition(table, "homography_calibrated")
    naive = condition(table, "naive_pixel_scaling")
    smooth_on = condition(table, "smoothing_enabled")
    smooth_off = condition(table, "smoothing_disabled")
    filtering_on = condition(table, "filtering_enabled")
    filtering_off = condition(table, "filtering_disabled")
    demos = [condition(table, f"demo_{count}") for count in (5, 15, 30)]
    f64 = precision["float64"]
    f32 = precision["float32"]

    lines = [
        START,
        f"All **{dataset['accepted_count']} final selected recordings** passed the quality "
        f"gate; the raw manifest preserves **{source_manifest['recording_count']} recordings** "
        "including pilots and retakes. On the same 50 held-out calibrated "
        "simulator scenarios:",
        "",
        "| System | Safe success | Object collisions | Unintended contacts | Drops | "
        "Median placement error | Median clearance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        row("Human-derived DMP teacher", dmp),
        row("24-unit GRU", gru),
        row("Straight line", straight),
        "",
        "Ablations are one-factor-at-a-time on 50 fixed scenarios:",
        "",
        f"- Demonstration count did not change safe success: "
        f"{demos[0]['safe_successes']}/50, {demos[1]['safe_successes']}/50 and "
        f"{demos[2]['safe_successes']}/50 for 5, 15 and 30 demonstrations.",
        f"- Calibration improved the safety margin: {calibrated['safe_successes']}/50 "
        f"safe with {mm(calibrated['minimum_clearance_median_m'])} median clearance "
        f"versus {naive['safe_successes']}/50 and "
        f"{mm(naive['minimum_clearance_median_m'])} for naive scaling.",
        f"- Smoothing did not improve success here: both settings achieved "
        f"{smooth_on['safe_successes']}/50; median clearance was "
        f"{mm(smooth_on['minimum_clearance_median_m'])} enabled and "
        f"{mm(smooth_off['minimum_clearance_median_m'])} disabled.",
        f"- Route/confidence filtering improved safety: {filtering_on['safe_successes']}/50 "
        f"safe when enabled versus {filtering_off['safe_successes']}/50, with "
        f"{filtering_off['object_collision_rollouts']} object-collision rollouts, "
        f"{filtering_off['unintended_contact_rollouts']} unintended-contact rollout and "
        f"{filtering_off['drop_rollouts']} drop when disabled.",
        f"- Float32 reduced the checkpoint from {f64['checkpoint_bytes']:,} to "
        f"{f32['checkpoint_bytes']:,} bytes but was slightly slower in this CPU benchmark "
        f"({f32['median_latency_ms']:.4f} vs {f64['median_latency_ms']:.4f} ms median).",
        END,
    ]
    return "\n".join(lines)


def update(readme: str) -> str:
    if START not in readme or END not in readme:
        raise ValueError("README generated-result markers are missing")
    before, remainder = readme.split(START, 1)
    _, after = remainder.split(END, 1)
    return before + generated_block() + after


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "README.md"
    current = path.read_text(encoding="utf-8")
    expected = update(current)
    if args.check:
        if current != expected:
            print("README metrics do not match committed machine-readable results.")
            return 1
        print("README metrics match committed machine-readable results.")
        return 0
    path.write_text(expected, encoding="utf-8")
    print("Updated README metrics from committed results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
