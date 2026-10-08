"""Portable provenance for future experiment manifests."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepositoryProvenance:
    git_commit: str | None
    git_dirty: bool | None
    uv_lock_sha256: str | None


def _run_git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_repository_provenance(root: Path) -> RepositoryProvenance:
    """Collect identifiers without embedding machine-local paths."""

    commit = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    return RepositoryProvenance(
        git_commit=commit,
        git_dirty=None if status is None else bool(status),
        uv_lock_sha256=_sha256(root / "uv.lock"),
    )


def build_run_manifest(
    root: Path,
    *,
    config_sha256: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Build the common manifest written by future experiment runs."""

    return {
        "schema_version": 2,
        "config_sha256": config_sha256,
        "configuration": configuration,
        "provenance": asdict(collect_repository_provenance(root)),
    }
