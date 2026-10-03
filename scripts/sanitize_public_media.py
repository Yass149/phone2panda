#!/usr/bin/env python3
"""Strip descriptive metadata and non-video streams from tracked public media."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def tracked_media() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "*.mp4", "*.png"], cwd=ROOT
    )
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def sanitize_mp4(path: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-c",
                "copy",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-fflags",
                "+bitexact",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sanitize_png(path: Path) -> None:
    with Image.open(path) as image:
        pixels = image.copy()
    with tempfile.NamedTemporaryFile(suffix=".png", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pixels.save(temporary, format="PNG", optimize=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    paths = tracked_media()
    for path in paths:
        if path.suffix.lower() == ".mp4":
            sanitize_mp4(path)
        else:
            sanitize_png(path)
    print(f"Sanitized {len(paths)} tracked public media files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
