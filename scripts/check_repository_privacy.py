#!/usr/bin/env python3
"""Reject private, machine-local, secret, or oversized staged content."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

from phone2panda.privacy import text_findings_bytes
from phone2panda.trajectories.public_data import validate_numeric_csv

MAX_FILE_BYTES = 5 * 1024 * 1024
PUBLIC_MEDIA = {
    "results/dataset_quality/plots/ep_001.png",
    "results/phase4a/phase4a_rollout.mp4",
    "results/phase4b/comparison.png",
    "results/phase4b/representative_failure.mp4",
    "results/phase4b/representative_success.mp4",
    "results/phase4e/comparison.png",
    "results/phase4e/representative_calibrated_failure.mp4",
    "results/phase4e/representative_calibrated_success.mp4",
    "results/phase5a/representative_success.mp4",
    "results/phase5a/training_curve.png",
    "results/phase5b/representative_success.mp4",
    "results/phase5b/training_curve.png",
    "results/phase6/ablation_plot.png",
    "media/phone_demo_sanitized.mp4",
    "media/phone2panda_demo.mp4",
    "media/phone2panda_preview.gif",
}
PRIVATE_PREFIXES = (
    ".cache/",
    ".local/",
    ".remember/",
    ".tools/",
    ".venv/",
    "data/processed/",
    "data/raw_private/",
)
PRIVATE_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
RAW_SUFFIXES = {".heic", ".mov"}
MEDIA_SUFFIXES = {".avi", ".gif", ".mkv", ".mp4", ".png", ".webm"}
SECRET_NAME_PARTS = ("credential", "secret")


def staged_paths() -> list[Path]:
    output = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"]
    )
    return [Path(item.decode()) for item in output.split(b"\0") if item]


def text_findings(path: Path) -> list[str]:
    return text_findings_bytes(path.read_bytes())


def media_findings(path: Path, data: bytes | None = None) -> list[str]:
    relative = path.as_posix()
    if relative not in PUBLIC_MEDIA:
        return ["media is not on the public artifact allowlist"]
    if path.suffix.lower() in {".gif", ".png"}:
        try:
            from PIL import Image
        except ImportError:
            return ["Pillow is required to audit staged PNG metadata"]
        source = io.BytesIO(data) if data is not None else path
        with Image.open(source) as image:
            metadata = {str(key).lower(): str(value).lower() for key, value in image.info.items()}
        allowed = {"background", "duration", "extension", "loop", "transparency", "version"}
        unexpected = set(metadata) if path.suffix.lower() == ".png" else set(metadata) - allowed
        if data is None and unexpected:
            return ["image retains descriptive metadata"]
        unsafe = ("gps", "location", "latitude", "longitude", "device", "make", "model")
        if any(term in key or term in value for key, value in metadata.items() for term in unsafe):
            return ["image contains location or device metadata"]
        return []

    try:
        import av
    except ImportError:
        return ["PyAV is required to audit staged video metadata"]
    source = io.BytesIO(data) if data is not None else str(path)
    with av.open(source) as container:
        if any(stream.type == "audio" for stream in container.streams):
            return ["video contains an audio stream"]
        metadata = dict(container.metadata)
        if data is None:
            allowed_format = {"major_brand", "minor_version", "compatible_brands"}
            if set(metadata) - allowed_format:
                return ["video retains descriptive container metadata"]
        for stream in container.streams:
            if data is None:
                allowed_stream = {"language", "handler_name", "vendor_id"}
                if set(stream.metadata) - allowed_stream:
                    return ["video retains descriptive stream metadata"]
            metadata.update(stream.metadata)
    unsafe = ("gps", "location", "latitude", "longitude", "device", "make", "model")
    if any(
        term in str(key).lower() or term in str(value).lower()
        for key, value in metadata.items()
        for term in unsafe
    ):
        return ["video contains location or device metadata"]
    return []


def path_findings(path: Path, size: int, data: bytes) -> list[str]:
    relative = path.as_posix()
    lowered = relative.lower()
    reasons: list[str] = []
    if lowered.startswith(PRIVATE_PREFIXES) or PRIVATE_PARTS.intersection(path.parts):
        reasons.append("private or machine-local path")
    if path.suffix.lower() in RAW_SUFFIXES:
        reasons.append("raw phone media")
    if path.name == ".env" or path.name.startswith(".env."):
        reasons.append("environment file")
    if any(part in path.name.lower() for part in SECRET_NAME_PARTS):
        reasons.append("secret-like filename")
    if size > MAX_FILE_BYTES:
        reasons.append(f"file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MiB")
    if path.parts[:2] == ("data", "public") and path.suffix.lower() == ".csv":
        try:
            validate_numeric_csv(data)
        except (ValueError, UnicodeError, csv.Error) as error:
            reasons.append(f"unapproved public trajectory content: {error}")
    reasons.extend(text_findings_bytes(data))
    if path.suffix.lower() in MEDIA_SUFFIXES:
        reasons.extend(media_findings(path, data))
    return sorted(set(reasons))


def audit(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.as_posix()
        data = path.read_bytes()
        reasons = path_findings(path, len(data), data)
        if reasons:
            errors.append(f"{relative}: {', '.join(reasons)}")
    return errors


def tracked_paths() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(item.decode()) for item in output.split(b"\0") if item]


def history_blobs() -> list[tuple[str, Path]]:
    output = subprocess.check_output(["git", "rev-list", "--objects", "--all"])
    object_paths: list[tuple[str, Path]] = []
    for line in output.decode(errors="replace").splitlines():
        oid, separator, name = line.partition(" ")
        if separator and name:
            object_paths.append((oid, Path(name)))
    unique_oids = sorted({oid for oid, _ in object_paths})
    batch = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        input="\n".join(unique_oids) + "\n",
        text=True,
        check=True,
        capture_output=True,
    ).stdout
    blob_oids = {line.split()[0] for line in batch.splitlines() if line.endswith(" blob")}
    return [(oid, path) for oid, path in object_paths if oid in blob_oids]


def audit_history() -> list[str]:
    errors: list[str] = []
    audited: set[tuple[str, str]] = set()
    for oid, path in history_blobs():
        key = (oid, path.as_posix())
        if key in audited:
            continue
        audited.add(key)
        data = subprocess.check_output(["git", "cat-file", "blob", oid])
        reasons = path_findings(path, len(data), data)
        if reasons:
            errors.append(f"{path.as_posix()}@{oid[:12]}: {', '.join(reasons)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tracked", action="store_true", help="audit the current Git index")
    group.add_argument("--history", action="store_true", help="audit every blob in Git history")
    args = parser.parse_args()
    if args.history:
        errors = audit_history()
        scope = "history"
    elif args.tracked:
        errors = audit(tracked_paths())
        scope = "tracked repository"
    else:
        errors = audit(staged_paths())
        scope = "staged content"
    if errors:
        print(f"Repository privacy check failed ({scope}):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Repository privacy check passed ({scope}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
