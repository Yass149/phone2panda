#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phone2panda.evaluation.phase4d import (
    load_phase4d_config,
    run_evaluation,
    run_preflight_grid,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the route-side Phase 4D attempt")
    parser.add_argument("--config", type=Path, default=Path("configs/phase4d.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    config = load_phase4d_config(args.config)
    if args.preflight_only:
        report = run_preflight_grid(config)
        print(
            f"Phase 4D preflight: {'PASS' if report['passed'] else 'FAIL'}; "
            f"selected_offset={report['selected_offset_m']}"
        )
        return 0 if report["passed"] else 2
    report = run_evaluation(config)
    print(
        f"Phase 4D evaluation: "
        f"safe={report['aggregate']['safety_successes']}/50"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
