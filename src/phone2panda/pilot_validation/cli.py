from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .pipeline import run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the five Phone2Panda pilot recordings")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilot_validation.yaml"),
        help="Validation configuration file",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run_validation(load_config(args.config))
    passed = bool(report["passed"])
    print(
        f"Pilot validation: {'PASS' if passed else 'FAIL'} "
        f"({report['passed_count']}/{report['pilot_count']} pilots passed)"
    )
    print("Report: results/pilot_validation/quality_report.json")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
