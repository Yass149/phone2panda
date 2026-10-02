#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phone2panda.evaluation.phase4b import (
    load_phase4b_config,
    run_evaluation,
    run_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-scenario Phase 4B comparison")
    parser.add_argument("--config", type=Path, default=Path("configs/phase4b.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    config = load_phase4b_config(args.config)
    if args.preflight_only:
        report = run_preflight(config)
        print(
            f"Phase 4B preflight: {'PASS' if report['passed'] else 'FAIL'}; "
            f"infrastructure failures={len(report['infrastructure_failures'])}"
        )
        return 0 if report["passed"] else 2
    report = run_evaluation(config)
    print(f"Phase 4B evaluation: PASS; methods={report['method_episode_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
