#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phone2panda.evaluation.phase4c import (
    audit_original_contacts,
    load_phase4c_config,
    run_safety_evaluation,
    run_safety_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the focused Phase 4C safety audit")
    parser.add_argument("--config", type=Path, default=Path("configs/phase4c.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-original", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    config = load_phase4c_config(args.config)
    if args.audit_original:
        report = audit_original_contacts(config)
        print(
            f"Phase 4C original audit: {report['rollouts']} rollouts; "
            f"unintended={report['contains_unintended_contacts']}"
        )
        return 0
    if args.preflight_only:
        report = run_safety_preflight(config)
        print(
            f"Phase 4C preflight: {'PASS' if report['passed'] else 'FAIL'}; "
            f"safe={report['safety_successes']}/{report['rollouts']}"
        )
        return 0 if report["passed"] else 2
    report = run_safety_evaluation(config)
    print(
        f"Phase 4C evaluation: PASS; "
        f"safe={report['aggregate']['safety_successes']}/50"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
