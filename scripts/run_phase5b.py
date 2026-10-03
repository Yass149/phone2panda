#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phone2panda.evaluation.phase5b import load_phase5b_config, run_phase5b


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5B held-out GRU evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/phase5b.yaml"))
    args = parser.parse_args()
    report = run_phase5b(load_phase5b_config(args.config))
    evaluation = report["held_out_evaluation"]
    gru = evaluation["gru_policy"]["safe_successes"]
    teacher = evaluation["dmp_route_confidence"]["safe_successes"]
    print(
        f"Phase 5B: {'PASS' if report['passed'] else 'FAIL'}; "
        f"GRU={gru}/50; DMP={teacher}/50"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
