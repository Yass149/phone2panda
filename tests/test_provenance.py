from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from phone2panda.provenance import build_run_manifest, collect_repository_provenance


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def test_repository_provenance_records_commit_dirty_state_and_lock_hash(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("locked\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "add", "uv.lock")
    _git(tmp_path, "commit", "-qm", "initial")

    clean = collect_repository_provenance(tmp_path)
    assert clean.git_commit is not None and len(clean.git_commit) == 40
    assert clean.git_dirty is False
    assert clean.uv_lock_sha256 == hashlib.sha256(b"locked\n").hexdigest()

    lock.write_text("changed\n", encoding="utf-8")
    assert collect_repository_provenance(tmp_path).git_dirty is True


def test_run_manifest_is_portable_outside_git(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("locked\n", encoding="utf-8")

    manifest = build_run_manifest(
        tmp_path,
        config_sha256="abc123",
        configuration={"output_dir": "results/example"},
    )

    assert manifest["provenance"]["git_commit"] is None
    assert manifest["provenance"]["git_dirty"] is None
    assert manifest["config_sha256"] == "abc123"
    assert str(tmp_path) not in json.dumps(manifest)
