#!/usr/bin/env python3
"""Validate the public Phase 7 demo and result provenance without simulation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import av
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "media/phone2panda_demo.mp4"


def check_demo() -> list[str]:
    errors: list[str] = []
    with av.open(str(DEMO)) as container:
        duration = float(container.duration or 0) / 1_000_000
        if not 60 <= duration <= 90:
            errors.append(f"demo duration is {duration:.3f}s, expected 60-90s")
        if any(stream.type == "audio" for stream in container.streams):
            errors.append("demo contains audio")
        video = next((stream for stream in container.streams if stream.type == "video"), None)
        if video is None:
            errors.append("demo has no video stream")
        else:
            decoded = sum(1 for _ in container.decode(video))
            if decoded == 0:
                errors.append("demo has no decodable frames")
        metadata = dict(container.metadata)
        for stream in container.streams:
            metadata.update(stream.metadata)
        unsafe = ("gps", "location", "latitude", "longitude", "device", "make", "model")
        if any(
            term in str(key).lower() or term in str(value).lower()
            for key, value in metadata.items()
            for term in unsafe
        ):
            errors.append("demo contains location or device metadata")
    return errors


def check_metrics() -> list[str]:
    errors: list[str] = []
    phase5b = json.loads((ROOT / "results/phase5b/evaluation.json").read_text())
    phase6 = json.loads((ROOT / "results/phase6/ablation_summary.json").read_text())
    held_out = phase5b["held_out_evaluation"]
    if held_out["dmp_route_confidence"]["safe_successes"] != 50:
        errors.append("DMP headline result is not 50/50")
    if held_out["gru_policy"]["safe_successes"] != 50:
        errors.append("GRU headline result is not 50/50")
    straight = next(row for row in phase6["table"] if row["condition"] == "straight_line")
    if straight["metrics"]["safe_successes"] != 0:
        errors.append("straight-line headline result is not 0/50")
    command = [sys.executable, str(ROOT / "scripts/generate_readme_results.py"), "--check"]
    if subprocess.run(command, cwd=ROOT, check=False).returncode:
        errors.append("README results are stale")
    return errors


def check_tracked_media() -> tuple[list[str], int]:
    errors: list[str] = []
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "*.mp4", "*.png"], cwd=ROOT
    )
    paths = [ROOT / item.decode() for item in output.split(b"\0") if item]
    unsafe = ("gps", "location", "latitude", "longitude", "device", "make", "model")
    for path in paths:
        try:
            if path.suffix.lower() == ".png":
                with Image.open(path) as image:
                    metadata = dict(image.info)
                    image.verify()
                if any(
                    term in str(key).lower() or term in str(value).lower()
                    for key, value in metadata.items()
                    for term in unsafe
                ):
                    errors.append(f"{path.relative_to(ROOT)} has private metadata")
                continue
            with av.open(str(path)) as container:
                if any(stream.type == "audio" for stream in container.streams):
                    errors.append(f"{path.relative_to(ROOT)} contains audio")
                metadata = dict(container.metadata)
                for stream in container.streams:
                    metadata.update(stream.metadata)
                video = next(
                    (stream for stream in container.streams if stream.type == "video"), None
                )
                decoded = 0 if video is None else sum(1 for _ in container.decode(video))
                if decoded == 0:
                    errors.append(f"{path.relative_to(ROOT)} has no decodable frames")
                if any(
                    term in str(key).lower() or term in str(value).lower()
                    for key, value in metadata.items()
                    for term in unsafe
                ):
                    errors.append(f"{path.relative_to(ROOT)} has private metadata")
        except Exception as exc:  # report corrupt media without hiding the source path
            errors.append(f"{path.relative_to(ROOT)} could not be opened: {exc}")
    return errors, len(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-only", action="store_true")
    args = parser.parse_args()
    errors = check_demo()
    media_count = 1
    if not args.demo_only:
        errors.extend(check_metrics())
        media_errors, media_count = check_tracked_media()
        errors.extend(media_errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    with av.open(str(DEMO)) as container:
        duration = float(container.duration or 0) / 1_000_000
    print(
        f"Submission check passed: {DEMO.relative_to(ROOT)} "
        f"({duration:.2f}s, silent); {media_count} tracked media files verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
