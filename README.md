# Phone2Panda pilot validation

This repository implements pilot validation and the Phase 3 full-dataset quality gate. It reads untouched iPhone HEVC `.MOV` files with PyAV, calibrates the canvas from the four custom L markers, tracks the red box marker, evaluates the labelled route and task geometry, and writes privacy-safe, audio-free overlays and machine-readable reports.

The phone is fixed, so the homography is established from the hands-free opening frames and checked again after the hand leaves. The gate requires at least 95% simultaneous corner visibility in both calibration windows, low marker jitter, and less than five pixels of start-to-end drift. Temporary hand or forearm occlusion during manipulation is reported as a diagnostic but does not invalidate an otherwise stable calibration.

The current recordings use custom L markers rather than the ArUco markers proposed in the original action plan. That deviation is explicit in `DECISIONS.md` and in every quality report.

## Reproduce locally

The bootstrap installs `uv` and CPython 3.11.17 under this project, then creates `.venv`. It does not use or modify global Anaconda.

```bash
make setup
make test
make lint
make validate-pilots
make validate-dataset
```

`make validate-pilots` returns exit code 0 only when all five recordings pass every gate. Exit code 2 means the reports and visual artifacts were produced, but one or more pilots require recording changes.

`make validate-dataset` checks `data/metadata.csv`, processes all matched recordings, and returns exit code 0 only when the dataset passes. It writes trajectory CSVs under `data/processed/v1/trajectories/` and the aggregate report plus selected review artifacts under `results/dataset_quality/`.

To revalidate replacements without decoding the full dataset, repeat `--episode-id`:

```bash
.venv/bin/python scripts/validate_dataset.py --episode-id ep_028
```

## Phase 4A deterministic rollout

Run the single accepted human-DMP Panda rollout headlessly:

```bash
make phase4a
```

The command uses accepted trajectory `ep_001`, fits its left-route DMP, executes explicit pick-and-place phases in robosuite, and writes the video, metrics, and transition arrays under `results/phase4a/`. It does not run baseline evaluation or policy training.

## Phase 4B baseline comparison

Run the bounded infrastructure preflight first, then the fixed 50-episode comparison:

```bash
make phase4b-preflight
make phase4b
```

The second command refuses to run unless the current configuration has a passing five-seed preflight. It evaluates straight-line control, nearest raw replay, DMP retargeting, and confidence-filtered DMP route selection on the same 50 seeds. Outputs under `results/phase4b/` include `rollouts.jsonl`, `aggregate.json`, `run_config.json`, `comparison.md`, `comparison.png`, and one representative success and failure video. Phase 4B does not train a policy.

## Phase 4C safety audit

Attribute the original whole-arm contacts, then run the winning-method safety preflight:

```bash
make phase4c-audit
make phase4c-preflight
```

The original 50-rollout audit is reproducible and saved under `results/phase4c/`. The current conservative posture candidate fails the zero-unintended-contact preflight, so `make phase4c` refuses to run. No corrected 50-episode result or policy-training result is claimed.

## Phase 4D lateral-placement attempt

Run the bounded three-offset route-side preflight:

```bash
make phase4d-preflight
```

The command tests only 0.06, 0.09, and 0.12 m using the existing five seeds and stops at the first safe configuration. No offset currently passes, so `make phase4d` refuses to run and no corrected video is claimed.

## Phase 4E calibrated obstacle

The 0.14 m Phase 4B–4D obstacle is retained as an uncalibrated tall-obstacle stress test. Phase 4E separately models the approximately 0.04 m physical glasses case:

```bash
make phase4e-diagnostics
```

The deliberately unsafe straight diagnostic passes when it demonstrates the expected obstacle failure; those object and gripper contacts are recorded rather than treated as a DMP blocker. The calibrated DMP route/confidence diagnostic passed with retained grasp, zero unintended contacts, and 1.16 mm placement error. Its five-seed safety preflight passed 5/5, followed by the fixed-seed four-method comparison in `results/phase4e/`.

## Phase 5A bounded GRU overfit gate

Phase 5A distils exactly five fixed-seed, human-derived route/confidence DMP rollouts into a 24-unit normalized GRU:

```bash
make phase5a
```

The GRU is intentionally selected for reproducibility, modest hardware requirements, and controlled latency measurement. The bounded overfit gate passed 5/5 on the same scenarios with zero object collisions, unintended robot contacts, or drops. This is an overfit/reloadability check only; it is not evidence of generalization and does not start full policy training.

Outputs are written under `results/pilot_validation/`:

- `quality_report.json` and `quality_report.csv`: aggregate machine-readable gate results.
- `source_manifest.json`: SHA-256 before/after integrity records for every local original MOV.
- `<episode>/quality.json`: full per-pilot metrics and failure reasons.
- `<episode>/frames.csv`: per-frame detections, normalised coordinates, and confidence.
- `<episode>/overlay.mp4`: H.264 annotated video with no audio stream.
- `<episode>/trajectory.png`: calibrated route, zones, obstacle, and conservative clearance.

The source `.MOV` files are opened read-only and checked with SHA-256 before and after processing. Their hashes are recorded in each per-pilot report.

Orientation is determined from decoded marker positions. The pipeline requires TL/TR to be above BL/BR and applies no rotation from QuickTime/Finder metadata.
