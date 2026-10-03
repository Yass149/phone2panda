#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phone2panda.evaluation.phase6 import load_phase6_config, run_phase6


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-seed Phase 6 ablations")
    parser.add_argument("--config", type=Path, default=Path("configs/phase6.yaml"))
    args = parser.parse_args()
    report = run_phase6(load_phase6_config(args.config))
    print(f"Phase 6 complete: {report['ablation_summary']}, {report['precision_benchmark']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
