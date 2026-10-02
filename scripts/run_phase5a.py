#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phone2panda.evaluation.phase5a import load_phase5a_config, run_phase5a


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Phase 5A GRU overfit gate")
    parser.add_argument("--config", type=Path, default=Path("configs/phase5a.yaml"))
    args = parser.parse_args()
    report = run_phase5a(load_phase5a_config(args.config))
    gate = report["gate"]
    print(
        f"Phase 5A: {'PASS' if report['passed'] else 'FAIL'}; "
        f"closed_loop={gate['successes']}/5"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
