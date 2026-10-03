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

## Phase 4C — Full-arm safety audit

- [x] Replay and attribute exact contacts for all 50 winning-method Phase 4B scenarios.
- [x] Separate intended gripper-cube, robot-obstacle, robot-table, and self contacts.
- [x] Add a conservative height, robot-envelope route score, goal-side lowering posture, and an unintended-contact gate.
- [x] Run the five-seed winning-method safety preflight.
- [ ] Run the corrected 50-episode evaluation only after the safety preflight passes.

Phase 4C stopped at the failed safety preflight as required. Phase 5 was not started.

## Phase 4D — Route-side lateral placement

- [x] Preserve the selected DMP transport path and add workspace-clamped route-side Y staging.
- [x] Lower laterally, place horizontally, release, and retreat through the same corridor.
- [x] Attribute every contact to its execution phase.
- [x] Test only 0.06, 0.09, and 0.12 m offsets on the five fixed preflight seeds.
- [ ] Run 50 corrected episodes only if an offset passes 5/5 placement with zero unintended contacts.

No offset passed. Phase 4D stopped before the 50-run evaluation, video generation, whole-arm planning, or Phase 5.

## Phase 4E — Calibrated 4 cm obstacle check

- [x] Preserve and relabel the earlier 0.14 m environment as an uncalibrated stress test.
- [x] Derive a low transport height from cube size, grasp offset, and table clearance.
- [x] Run one straight and one DMP diagnostic on the same deterministic seed.
- [x] Treat the straight diagnostic's expected obstacle collision as baseline evidence, not a DMP gate failure.
- [x] Run the five-seed calibrated DMP safety preflight (5/5 safe placements).
- [x] Run exactly 50 calibrated episodes for each of the four Phase 4B methods.

The calibrated route/confidence DMP achieved 50/50 safe successes with zero unintended-contact rollouts.

## Phase 5A — Five-episode GRU overfit gate

- [x] Collect exactly five successful fixed-seed teacher rollouts from the calibrated human-derived DMP controller.
- [x] Preserve episode boundaries, state/context, route, gripper state, actions, and human-DMP provenance.
- [x] Implement and cap deterministic training of a normalized 24-unit GRU.
- [x] Add focused normalization, sequence-batching, and checkpoint-reload tests.
- [x] Pass all five same-scenario closed-loop evaluations with zero collisions, unintended contacts, or drops.
- [x] Save one passing video, checkpoint, schema, curve, and machine-readable evaluation.

Phase 5A is complete. Full policy training and optimization have not started.

## Phase 5B — Held-out calibrated GRU evaluation

- [x] Fix disjoint balanced manifests for 120 training, 20 validation, and 50 test scenarios.
- [x] Generate headless teacher actions from human-derived route/confidence DMP trajectories.
- [x] Reuse the normalized 24-unit GRU and train with episode-disjoint validation early stopping.
- [x] Evaluate GRU and DMP teacher on the identical 50 untouched calibrated scenarios.
- [x] Pass the 45/50 gate: GRU 50/50 and teacher 50/50, with zero safety failures.
- [x] Save provenance, checkpoint, curve, per-rollout results, and one success video.

Phase 5B is complete. Optimization and ablations have not started.
