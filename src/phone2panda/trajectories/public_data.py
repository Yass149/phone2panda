"""Validate the public numerical dataset and resolve private/public input paths."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

PUBLIC_DIRECTORY = Path("data/public")
PRIVATE_DIRECTORY = Path("data/processed/v1/trajectories")
PUBLIC_FIELDS = (
    "frame_index", "object_x", "object_y", "smoothed_x", "smoothed_y",
    "tracking_confidence", "object_detected", "object_interpolated",
    "all_corners_direct", "valid",
)
FLAG_FIELDS = ("object_detected", "object_interpolated", "all_corners_direct", "valid")
FLOAT_FIELDS = ("object_x", "object_y", "smoothed_x", "smoothed_y", "tracking_confidence")
# BLAS implementations can round DMP fitting differently; one nanometre in robot coordinates.
REBUILT_PRIOR_ATOL_M = 1e-9


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_numeric_csv(data: bytes) -> dict[str, int]:
    """Reject extra metadata, invalid numbers, broken frame order and bad flags."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != PUBLIC_FIELDS:
        raise ValueError("Public trajectory must contain only the approved numerical columns")
    count = valid_count = 0
    for index, row in enumerate(reader):
        if set(row) != set(PUBLIC_FIELDS) or any(value is None for value in row.values()):
            raise ValueError("Public trajectory row has missing or extra fields")
        if int(row["frame_index"]) != index:
            raise ValueError("Public trajectory frame indices must be contiguous from zero")
        for field in FLOAT_FIELDS:
            value = float(row[field])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid normalized public trajectory value: {field}")
        if any(row[field] not in {"0", "1"} for field in FLAG_FIELDS):
            raise ValueError("Public trajectory flags must be zero or one")
        count += 1
        valid_count += int(row["valid"])
    if valid_count < 5:
        raise ValueError("Public trajectory needs at least five valid observations")
    return {"rows": count, "valid_rows": valid_count}


def read_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or tuple(manifest.get("fields", ())) != PUBLIC_FIELDS:
        raise ValueError("Unsupported public trajectory schema")
    rows = manifest["episodes"]
    identifiers = {row["episode_id"] for row in rows}
    if len(rows) != 36 or identifiers != {f"ep_{index:03d}" for index in range(1, 37)}:
        raise ValueError("Public release must contain the 36 unique accepted episodes")
    for row in rows:
        if row["path"] != f"trajectories/{row['episode_id']}.csv":
            raise ValueError("Invalid public trajectory path")
        if row["route"] not in {"left", "right"} or row["start_id"] not in {"s1", "s2", "s3"}:
            raise ValueError("Invalid public episode labels")
    return manifest


def validate_public_release(directory: Path) -> dict[str, Any]:
    manifest = read_manifest(directory)
    expected = {row["path"] for row in manifest["episodes"]}
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*.csv")}
    if actual != expected:
        raise ValueError("Public trajectory files do not match the manifest")
    total = 0
    for row in manifest["episodes"]:
        data = (directory / row["path"]).read_bytes()
        if digest(data) != row["public_csv_sha256"]:
            raise ValueError(f"Public trajectory checksum mismatch: {row['episode_id']}")
        counts = validate_numeric_csv(data)
        if any(counts[key] != row[key] for key in counts):
            raise ValueError(f"Public trajectory row count mismatch: {row['episode_id']}")
        total += counts["rows"]
    if total != manifest["total_rows"]:
        raise ValueError("Public release total row count mismatch")
    root = directory.parent.parent
    approved_inputs = {"data/metadata.csv", "results/dataset_quality/quality_report.json"}
    if set(manifest["input_sha256"]) != approved_inputs:
        raise ValueError("Public release is missing its provenance inputs")
    for relative, expected_hash in manifest["input_sha256"].items():
        if digest((root / relative).read_bytes()) != expected_hash:
            raise ValueError(f"Public release provenance input changed: {relative}")
    with (root / "data/metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata = {row["episode_id"]: row for row in csv.DictReader(handle)}
    for row in manifest["episodes"]:
        source = metadata[row["episode_id"]]
        labels = (row["route"], row["start_id"], row["collection_split"])
        if labels != (source["route"], source["start_id"], source["split"]):
            raise ValueError(f"Public episode labels changed: {row['episode_id']}")
    return {"passed": True, "episodes": len(expected), "rows": total}


def resolve_processed_path(root: Path, relative: str | Path, *, public_only: bool = False) -> Path:
    """Keep local experiments intact; a clone falls back to the audited release."""
    relative = Path(relative)
    local = root / relative
    if relative.parent != PRIVATE_DIRECTORY:
        return local
    if local.is_file() and not public_only:
        return local
    directory = root / PUBLIC_DIRECTORY
    manifest = read_manifest(directory)
    entry = next((row for row in manifest["episodes"] if row["episode_id"] == relative.stem), None)
    if entry is None:
        raise ValueError(f"Trajectory is absent from the public release: {relative.stem}")
    public = directory / entry["path"]
    if digest(public.read_bytes()) != entry["public_csv_sha256"]:
        raise ValueError(f"Public trajectory checksum mismatch: {relative.stem}")
    return public
