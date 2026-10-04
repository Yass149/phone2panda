#!/usr/bin/env python3
"""Run bounded saved-model simulation from the public checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phone2panda.public_demo import run_public_demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5, choices=range(1, 6))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run_public_demo(root, episodes=args.episodes)
    output = root / "results/public_demo/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in report["rollouts"]:
        print(
            f"seed={row['scenario']['seed']} route={row['source_route']} "
            f"safe={row['phase5a_gate_success']} "
            f"placement={1000 * row['placement_error_m']:.2f} mm"
        )
    print(f"Saved-model smoke test: {report['safe_successes']}/{report['episodes']} safe")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
