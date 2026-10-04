#!/usr/bin/env python3
"""Release numerical observations without timestamps or original media metadata."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from phone2panda.trajectories.public_data import (
    PUBLIC_DIRECTORY,
    PUBLIC_FIELDS,
    digest,
    validate_numeric_csv,
    validate_public_release,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "results/dataset_quality/quality_report.json").read_text())
    with (root / "data/metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata = {row["episode_id"]: row for row in csv.DictReader(handle)}
    episodes = [row for row in report["episodes"] if row["accepted"]]
    if len(episodes) != 36 or len(metadata) != 36:
        raise ValueError("Only the complete 36-episode accepted dataset may be released")
    directory = root / PUBLIC_DIRECTORY
    (directory / "trajectories").mkdir(parents=True, exist_ok=True)
    entries = []
    for episode in episodes:
        identifier = episode["episode_id"]
        source = root / episode["processed_trajectory"]
        before = source.read_bytes()
        reader = csv.DictReader(io.StringIO(before.decode("utf-8")))
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=PUBLIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            writer.writerow({field: row[field] for field in PUBLIC_FIELDS})
        data = output.getvalue().encode("utf-8")
        counts = validate_numeric_csv(data)
        relative = f"trajectories/{identifier}.csv"
        (directory / relative).write_bytes(data)
        if source.read_bytes() != before:
            raise RuntimeError(f"Private source changed during export: {identifier}")
        entries.append({
            "episode_id": identifier, "path": relative,
            "route": metadata[identifier]["route"], "start_id": metadata[identifier]["start_id"],
            "collection_split": metadata[identifier]["split"], **counts,
            "source_csv_sha256": digest(before), "public_csv_sha256": digest(data),
        })
    provenance = ("data/metadata.csv", "results/dataset_quality/quality_report.json")
    manifest = {
        "schema_version": 1, "fields": PUBLIC_FIELDS, "episodes": entries,
        "total_rows": sum(row["rows"] for row in entries),
        "coordinates": "Homography-normalized canvas coordinates, x/y in [0, 1]",
        "processing": (
            "Original numerical strings preserved; no rounding, resampling or relabelling"
        ),
        "excluded": ["timestamps", "images", "audio", "device metadata", "absolute paths"],
        "input_sha256": {path: digest((root / path).read_bytes()) for path in provenance},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    result = validate_public_release(directory)
    print(
        f"Exported {result['episodes']} public trajectories, "
        f"{result['rows']} rows; sources unchanged"
    )


if __name__ == "__main__":
    main()
