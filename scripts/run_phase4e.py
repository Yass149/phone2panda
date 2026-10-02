#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phone2panda.evaluation.phase4e import (
    load_phase4e_config,
    reassess_saved_diagnostics,
    run_diagnostics,
    run_evaluation,
    run_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the calibrated Phase 4E check")
    parser.add_argument("--config", type=Path, default=Path("configs/phase4e.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diagnostics", action="store_true")
    mode.add_argument("--reassess-diagnostics", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    config = load_phase4e_config(args.config)
    if args.reassess_diagnostics:
        report = reassess_saved_diagnostics(config)
        print(f"Phase 4E diagnostic reassessment: {'PASS' if report['passed'] else 'FAIL'}")
        return 0 if report["passed"] else 2
    if args.diagnostics:
        report = run_diagnostics(config)
        print(f"Phase 4E diagnostics: {'PASS' if report['passed'] else 'FAIL'}")
        return 0 if report["passed"] else 2
    if args.preflight_only:
        report = run_preflight(config)
        successes = report["aggregate"]["calibrated_safety_successes"]
        print(f"Phase 4E preflight: {'PASS' if report['passed'] else 'FAIL'}; safe={successes}/5")
        return 0 if report["passed"] else 2
    report = run_evaluation(config)
    counts = report["method_episode_counts"]
    summary = ", ".join(
        f"{method}={report['aggregate'][method]['successes']}/{counts[method]}"
        for method in counts
    )
    print(f"Phase 4E evaluation: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
