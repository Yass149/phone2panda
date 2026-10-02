#!/usr/bin/env python3
"""Reject private, machine-local, secret, or oversized staged content."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

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
MEDIA_SUFFIXES = {".avi", ".mkv", ".mp4", ".png", ".webm"}
SECRET_NAME_PARTS = ("credential", "secret")


def staged_paths() -> list[Path]:
    output = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"]
    )
    return [Path(item.decode()) for item in output.split(b"\0") if item]


def text_findings(path: Path) -> list[str]:
    data = path.read_bytes()
    if b"\0" in data:
        return []
    text = data.decode("utf-8", errors="replace")
    patterns = {
        "absolute macOS home path": re.compile("/" + r"Users/[^/\s]+/"),
        "absolute Linux home path": re.compile("/" + r"home/[^/\s]+/"),
        "absolute Windows home path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
        "AWS access key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
        "GitHub token": re.compile("gh" + r"[pousr]_[A-Za-z0-9_]{20,}"),
        "private key": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }
    return [label for label, pattern in patterns.items() if pattern.search(text)]


def media_findings(path: Path) -> list[str]:
    relative = path.as_posix()
    if relative not in PUBLIC_MEDIA:
        return ["media is not on the public artifact allowlist"]
    if path.suffix.lower() == ".png":
        try:
            from PIL import Image
        except ImportError:
            return ["Pillow is required to audit staged PNG metadata"]
        with Image.open(path) as image:
            metadata = {str(key).lower(): str(value).lower() for key, value in image.info.items()}
        unsafe = ("gps", "location", "latitude", "longitude", "device", "make", "model")
        if any(term in key or term in value for key, value in metadata.items() for term in unsafe):
            return ["PNG contains location or device metadata"]
        return []

    try:
        import av
    except ImportError:
        return ["PyAV is required to audit staged video metadata"]
    with av.open(str(path)) as container:
        if any(stream.type == "audio" for stream in container.streams):
            return ["video contains an audio stream"]
        metadata = dict(container.metadata)
        for stream in container.streams:
            metadata.update(stream.metadata)
    unsafe = ("gps", "location", "latitude", "longitude", "device", "make", "model")
    if any(
        term in str(key).lower() or term in str(value).lower()
        for key, value in metadata.items()
        for term in unsafe
    ):
        return ["video contains location or device metadata"]
    return []


def audit(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        relative = path.as_posix()
        lowered = relative.lower()
        if not path.is_file():
            continue
        reasons: list[str] = []
        if lowered.startswith(PRIVATE_PREFIXES) or PRIVATE_PARTS.intersection(path.parts):
            reasons.append("private or machine-local path")
        if path.suffix.lower() in RAW_SUFFIXES:
            reasons.append("raw phone media")
        if path.name == ".env" or path.name.startswith(".env."):
            reasons.append("environment file")
        if any(part in path.name.lower() for part in SECRET_NAME_PARTS):
            reasons.append("secret-like filename")
        if path.stat().st_size > MAX_FILE_BYTES:
            reasons.append(f"file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MiB")
        reasons.extend(text_findings(path))
        if path.suffix.lower() in MEDIA_SUFFIXES:
            reasons.extend(media_findings(path))
        if reasons:
            errors.append(f"{relative}: {', '.join(sorted(set(reasons)))}")
    return errors


def main() -> int:
    errors = audit(staged_paths())
    if errors:
        print("Repository privacy check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Repository privacy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
