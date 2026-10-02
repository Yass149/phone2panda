#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from phone2panda.dataset_validation import load_dataset_config, run_dataset_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the full Phone2Panda recording dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_validation.yaml"),
    )
    parser.add_argument(
        "--episode-id",
        action="append",
        dest="episode_ids",
        help="Revalidate only this episode ID; repeat for multiple episodes.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run_dataset_validation(
        load_dataset_config(args.config),
        set(args.episode_ids) if args.episode_ids else None,
    )
    if not report["inventory_passed"]:
        for mismatch in report["mismatches"]:
            print(mismatch)
        return 2
    print(
        f"Dataset validation: {report['accepted_count']} accepted, "
        f"{report['rejected_count']} rejected"
    )
    return 0 if report["dataset_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
