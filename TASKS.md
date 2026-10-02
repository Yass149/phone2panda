# Tasks

## Pilot-validation milestone

- [x] Read the build prompt and action plan completely.
- [x] Audit the folder and inspect all five pilot recordings.
- [x] Create a project-local Python 3.11 environment with locked dependencies.
- [x] Implement reliable HEVC decoding and source-integrity checks.
- [x] Implement custom four-corner marker calibration and perspective correction.
- [x] Implement red-marker tracking with confidence and short-gap interpolation.
- [x] Implement normalised start, route, clearance, and target checks.
- [x] Generate per-pilot overlays, trajectory plots, frame tables, and quality reports.
- [x] Add unit tests, linting, and reproducible run instructions.
- [x] Pass the automated gate on all five pilots.

## Phase 3 — Dataset quality

- [x] Match all 36 metadata rows to non-empty recordings.
- [x] Preserve and hash every raw recording before and after processing.
- [x] Process normalised trajectories with the validated pilot components.
- [x] Validate orientation, calibration, drift, tracking, task geometry, and placement.
- [x] Generate the aggregate report plus selected review artifacts.
- [x] Revalidate only the eight replacement recordings and merge their results.
- [x] Replace and revalidate `ep_028` using `ep_028_s3_t1_right_redo2.MOV`.
- [x] Pass the complete 36-recording dataset quality gate.

Phase 3 is complete. Model training and simulator work were not started.

## Phase 4A — Deterministic DMP rollout

- [x] Run one bounded random-action Panda smoke test headlessly.
- [x] Select accepted training trajectory `ep_001` and fit its left-route DMP.
- [x] Map the human path into the Panda workspace with explicit task phases.
- [x] Add focused mapping, action-bound, and deterministic-reset tests.
- [x] Save one deterministic successful rollout, video, configuration, and metrics.
- [x] Run the 50-episode baseline evaluation only after explicit approval.

Phase 4A is complete. Policy training was not started.

## Phase 4B — Fixed-scenario baseline comparison

- [x] Implement straight-line, nearest raw replay, DMP, and route/confidence DMP methods.
- [x] Pass five fixed-seed infrastructure preflights per method.
- [x] Evaluate exactly 50 identical seeded scenarios per method.
- [x] Save per-rollout records, aggregates, confidence intervals, comparison plot, and two videos.
- [x] Record object collisions, robot contacts, drops, placement, efficiency, clearance, saturation, and latency.

Phase 4B is complete. Policy training and Phase 5 were not started.
