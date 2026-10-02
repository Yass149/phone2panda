#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phone2panda.sim.phase4a import load_phase4a_config, run_phase4a


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one deterministic human-DMP Panda rollout")
    parser.add_argument("--config", type=Path, default=Path("configs/phase4a.yaml"))
    args = parser.parse_args()
    metrics = run_phase4a(load_phase4a_config(args.config))
    print(
        f"Phase 4A rollout: {'PASS' if metrics['accepted'] else 'FAIL'}; "
        f"placement error={metrics['placement_error_m']:.4f} m"
    )
    return 0 if metrics["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
