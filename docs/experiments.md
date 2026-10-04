# Experiments

## 2026-10-02 — Pilot media audit

- Input: five iPhone QuickTime `.MOV` files.
- Codec: HEVC video, AAC audio, timed metadata.
- Resolution: 1920×1080.
- Rate: approximately 30 fps.
- Durations: 8.7–11.8 seconds.
- Full decode: 1,501 video frames decoded successfully with PyAV 16.0.1.
- Observation: all recordings use custom L corner markers, a red marker on a black box, a blue central obstacle, three bottom start zones, and one top target.

## 2026-10-02 — Detector selection

- Corner threshold sweep: grayscale thresholds 120–200.
- Selected: 160, because the grey right-side markers are missed below this level while the marker-plus-label components remain isolated from the carpet.
- Red saturation sweep: 90–160.
- Selected: HSV saturation ≥130, which retains the red marker and rejects skin-colour false positives across the five pilots.
- Homography: marker outer elbows mapped to the unit square.

Final acceptance-command results are appended after the checked pipeline run.

## 2026-10-02 — Five-pilot automated gate

- Command: `.venv/bin/python scripts/validate_pilots.py --config configs/pilot_validation.yaml`
- Tests before run: 6 passed; Ruff passed.
- Result: **FAIL, 2/5 pilots passed**. This is an expected non-zero exit until replacement recordings pass.
- Passed: `pilot_02_s3_right`, `pilot_04_s2_right`.
- Failed only on simultaneous direct corner-marker coverage:
  - `pilot_01_s1_left`: 74.60% (bottom-left marker occluded).
  - `pilot_03_s2_left`: 73.15% (bottom-left marker occluded).
  - `pilot_05_s2_right_repeat`: 90.42% (bottom-left marker occluded).
- Red-marker detected coverage: 100% for all five.
- Decode-dropped frames: 0 for all five.
- Expected route, start zone, conservative obstacle clearance, and final placement: passed for all five.
- Minimum footprint-adjusted clearance range: 0.123–0.173 normalised canvas units.
- Original-integrity SHA-256 before/after: identical for all five.
- Overlay verification: H.264, 1280×720, source frame count preserved, zero audio streams.

## 2026-10-02 — Replacement-pilot rerun

- Replacements: `pilot_01_s1_left_v2.MOV`, `pilot_03_s2_left_v2.MOV`, and `pilot_05_s2_right_repeat_v2.MOV`.
- Unchanged inputs: pilots 02 and 04.
- Decoded orientation: 1920×1080 landscape for all replacements. Marker ordering showed TL/TR at y≈40–45 px and BL/BR at y≈1019–1028 px. Rotation applied: 0°.
- Result with the unchanged 95% gate: **FAIL, 2/5 pilots passed**.
- Replacement direct corner coverage:
  - Pilot 01 v2: 86.67%; BL marker hidden for frames 133–168.
  - Pilot 03 v2: 82.25%; BL marker hidden for frames 122–170.
  - Pilot 05 v2: 82.79%; BL marker hidden for frames 123–175.
- All replacements passed red tracking (100%), decoded-frame integrity (0 drops), labelled route, start zone, conservative clearance, and final placement.
- Integrity manifest: all eight MOV originals had identical SHA-256 values before and after processing.
- Public-artifact audit: overlay videos have one H.264 video stream and no audio/subtitle streams; plots have no EXIF; reports contain no absolute private paths, GPS/location strings, or device identifiers.

## 2026-10-02 — Fixed-camera calibration-integrity gate

- Rationale: whole-video direct corner visibility rejected valid manipulation frames when the forearm temporarily crossed the bottom-left marker, even though the phone and canvas remained fixed.
- Revised acceptance measures: at least 95% simultaneous four-marker visibility in both 0.75-second hands-free windows, no more than 3 px RMS marker jitter, and no more than 5 px start-to-end marker drift.
- Whole-video direct marker coverage remains in reports as a diagnostic and is not hidden.
- Tests: 8 passed; Ruff passed.
- Result: **PASS, 5/5 pilots passed**.
- Opening and closing calibration-window coverage: 100% for every pilot.
- Maximum per-pilot marker jitter: 0.885–1.608 px RMS.
- Maximum per-pilot start-to-end marker drift: 1.000–3.162 px.
- Red-marker tracking: 100% for every pilot.
- Decode-dropped frames: 0 for every pilot.
- Expected route, start zone, conservative obstacle clearance, and final placement: passed for every pilot.

## 2026-10-02 — Phase 3 dataset quality gate

- Command: `.venv/bin/python scripts/validate_dataset.py --config configs/dataset_validation.yaml`
- Inventory: 36/36 metadata rows matched non-empty MOV files; no missing, extra, or duplicate recordings.
- Result: **FAIL, 28 accepted and 8 rejected**; train 20/20, validation 4/4, test 4/12.
- Seven test recordings failed only the 80% S3 start-footprint hold gate; ratios were 0.000000–0.695652.
- `ep_032_s3_t1_left.MOV` failed only red-marker tracking: 92.3497%, below the unchanged 95% gate.
- All source SHA-256 hashes matched before and after processing.
- Artifact audit: video-only overlays, no private metadata or absolute paths; 36 trajectory CSVs written.

## 2026-10-02 — Targeted replacement revalidation

- Updated eight metadata filenames to their `_redo.MOV` replacements and decoded only those episodes.
- Lighter grey tape required a dataset-only corner threshold of 200; the pilot configuration remains unchanged.
- BL shoulder occlusion did not fail calibration: every replacement had 100% opening and closing four-marker coverage.
- Result: **FAIL, 35 accepted and 1 rejected**; train 20/20, validation 4/4, test 11/12.
- `ep_028_s3_t1_right_redo.MOV` exceeded both calibration limits: 3.624 px window jitter and 6.083 px start-to-end drift.
- All other replacement checks passed, including 100% red-marker tracking, start footprint, route, clearance, and final placement.

## 2026-10-02 — Final targeted replacement

- Revalidated only `ep_028_s3_t1_right_redo2.MOV`; the other 35 episode results were reused.
- Result: **PASS, 36/36 accepted**; train 20/20, validation 4/4, test 12/12.
- Opening and closing calibration coverage: 100%; marker jitter: 0.752 px; start-to-end drift: 1.000 px.
- Red-marker tracking, start footprint, route, clearance, and final placement: all passed.

## 2026-10-02 — Phase 4A smoke and deterministic rollout

- Environment: project `.venv`, Python 3.11.17, robosuite 1.5.1, MuJoCo 3.2.7, macOS arm64, headless CGL.
- Compatibility check: robosuite failed before stepping with MuJoCo 3.14.0; pinning 3.2.7 resolved the joint-enum incompatibility.
- Smoke command: 25 seeded random bounded actions in headless `PickPlaceCan`; passed with maximum absolute action 0.04929.
- Acceptance command: `MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4a.py --config configs/phase4a.yaml`.
- Source: accepted training trajectory `ep_001`, labelled left route; 80 smoothed/resampled DMP waypoints.
- Result: **PASS on the first deterministic rollout**, seed 11, 268 steps, 13.4 seconds.
- Placement error: 0.00513 m within the 0.055 m target radius; obstacle collisions: 0; minimum obstacle center clearance: 0.06336 m.
- Maximum transport deviation from the straight start-goal line: 0.09934 m, demonstrating material control by the human path.
- Action-bound violations: 0; deterministic reset maximum error: 0.0.
- Tests: 11 passed; Ruff passed. The 50-episode baseline evaluation and policy training were not run.

## 2026-10-02 — Phase 4B fixed-scenario comparison

- Preflight: five seeds per method, 20 rollouts total; no infrastructure, reset, finite-metric, or action-bound failures.
- Evaluation: exactly 50 episodes per method using identical seeds 4100–4149 and scenario geometry.
- Straight line: 0/50 task successes, 50 object collisions, 22 drops, median clearance −48.7 mm.
- Nearest raw replay: 48/50 successes, 2 object collisions, no drops, median clearance 20.9 mm.
- DMP retargeting: 47/50 successes, 3 object collisions, no drops, median clearance 19.5 mm.
- DMP with route/confidence selection: 50/50 successes, no object collisions or drops, median clearance 23.1 mm; Wilson 95% success interval 92.9–100.0%.
- Median controller latency was 0.0076–0.0077 ms; p95 was 0.0113–0.0143 ms. Every method had zero action-bound violations.
- Whole-arm diagnostic: Panda-link obstacle contact occurred in 39/50 route/confidence runs, so future rollout collection must filter these contacts or add robot-aware clearance before policy training.
- Artifacts: 200-line raw rollout record, aggregate JSON, resolved configuration, Markdown table, one reproducible plot, one successful replay, and one failure replay.
- Tests: 14 passed; Ruff passed. Phase 5 was not started.

## 2026-10-02 — Phase 4C full-arm contact audit

- Reused the Phase 4B logs to identify 39 affected winning-method seeds and 687 legacy robot-contact steps.
- Deterministically replayed the 50 winning-method scenarios with named MuJoCo contact-pair logging; every legacy per-rollout count matched.
- Intended gripper-cube: 93,245 contact events across all 50 rollouts. Cube-table: 14,878 reported events.
- Unintended robot-obstacle: 688 events across 39 rollouts—687 `robot0_link7_collision` events and one `gripper0_right_hand_collision` event against `route_obstacle_geom`.
- Unintended robot-table contacts: 0. Self-collisions: 0.
- A conservative 1.03 m transport height, 0.055 m robot-envelope route score, and 0.056 m goal-side lowering posture eliminated the worst original seed in isolation.
- Official five-seed safety preflight: **FAIL, 0/5 safety successes**. Object placement remained 5/5, but robot-obstacle contact remained 5/5 and included links 5–7.
- Per the gate, the corrected 50-episode run and representative corrected video were not produced. Phase 5 was not started.

## 2026-10-02 — Phase 4D route-side lateral placement

- Preserved the Phase 4B DMP XY transport path; added target-X, route-side-Y staging with workspace clamping and phase-level contact attribution.
- Tested the required offsets in order on seeds 401–405.
- 0.06 m: 5/5 placement, 0/5 safety; 90 robot-obstacle events—4 while lowering, 7 while placing, 78 during release, and 1 during low retreat.
- 0.09 m: 1/5 placement, 0/5 safety; 91 robot-obstacle events—7 while lowering, 11 while placing, and 73 during release.
- 0.12 m: 0/5 placement, 0/5 safety; 91 robot-obstacle events—1 while lowering, 10 while placing, and 80 during release.
- Transport and route-side staging had zero unintended contacts for every offset. Robot-table contacts and self-collisions were also zero.
- No offset passed, so the corrected 50-run evaluation and video were not produced. Collision-aware IK or whole-arm planning is required; Phase 5 was not started.

## 2026-10-02 — Phase 4E calibrated obstacle diagnostic

- Physical and simulated obstacle height: 0.040 m. Earlier 0.14 m Phase 4B–4D results remain unchanged and are now labelled an uncalibrated tall-obstacle stress test.
- Simulated cube size range: 0.040–0.044 m; seed-402 cube: 0.04020 × 0.04191 × 0.04125 m.
- Derived end-effector transport height: 0.847715 m; measured median: 0.84703 m. Median cube-bottom clearance was 0.01115 m straight and 0.01092 m DMP.
- Straight diagnostic: object-obstacle collision on 90 control steps / 222 contacts, −52.6 mm footprint clearance, grasp retained, but 44 gripper-finger/obstacle contacts on 25 steps.
- DMP route/confidence diagnostic: zero object-obstacle, robot-obstacle, robot-table, or self contacts; grasp retained; 18.9 mm minimum clearance; 1.16 mm placement error.
- Corrected diagnostic interpretation: **PASS**. The straight run demonstrated its expected obstacle failure; its gripper contacts were recorded but did not block DMP evaluation. The saved diagnostics were reassessed without rerunning simulation.
- Five-seed DMP route/confidence preflight: **PASS, 5/5** placements with retained grasp and zero unintended contacts.
- Fixed scenarios, seeds 4100–4149, 50 episodes per method: straight 0/50 safe successes, nearest raw replay 49/50, DMP 49/50, and DMP + route/confidence 50/50.
- Route/confidence DMP: zero object collisions, unintended contacts, or drops; 1.39 mm median placement error; 22.43 mm median and 9.88 mm worst obstacle clearance; 3.84% action saturation.
- The only non-straight failures were at seed 4140: one gripper-finger/obstacle event for raw replay and 14 for unfiltered DMP. Phase 5 was not started.

## 2026-10-02 — Phase 5A bounded GRU overfit gate

- Teacher data: exactly five successful route/confidence DMP rollouts, seeds 401–405, with preserved episode boundaries and source-episode/route/confidence provenance.
- Policy: normalized 24-unit GRU, 3,559 parameters, 31,706-byte compressed checkpoint; deterministic seed 5001.
- Training: 500-epoch hard cap reached in 6.96 seconds; best normalized action MSE 0.00248018. Early stopping was enabled with a 50-epoch patience.
- Closed-loop result on the same five scenarios: **PASS, 5/5** acceptable placements, zero object collisions, zero unintended contacts, zero drops, and retained grasp in 5/5.
- Placement errors: 1.68, 2.72, 0.43, 2.15, and 0.53 mm; maximum 2.72 mm against the 10 mm gate.
- Policy latency: 0.0271 ms median and 0.0283 ms p95. Exactly one passing rollout video was rendered after the gate passed.
- This is an overfit gate only. Full policy training, hyperparameter optimization, ACT, LeRobot, and SmolVLA were not started.

## 2026-10-03 — Phase 5B held-out calibrated evaluation

- Fixed disjoint manifests before rollout generation: training 120 (40 per start, 60/60 routes), validation 20 (7/6/7 starts, 10/10 routes), and testing 50 (17/16/17 starts, 25/25 routes).
- Teacher provenance records accepted human episode, selected route/confidence, smoothed DMP retargeting, and bounded Panda action for every training and validation episode.
- Reused normalized 24-unit GRU: 3,559 parameters and a 31,942-byte checkpoint.
- Validation-monitored training reached the 250-epoch cap in 12.01 seconds; best validation MSE 0.00472591. No test seed was used for normalization, training, or checkpoint selection.
- Held-out result: **PASS**. GRU 50/50 and DMP teacher 50/50 safe successes; both had zero object collisions, unintended contacts, or drops.
- GRU versus DMP: median placement error 1.95 vs 1.53 mm; median minimum clearance 21.72 vs 22.07 mm; median latency 0.0269 vs 0.0140 ms.
- One representative GRU success video was generated. No failure video was generated because no held-out GRU rollout failed. Optimization and ablations were not started.

## 2026-10-03 — Phase 6 fixed-seed ablations

- Every condition used the same 50 calibrated Phase 5B scenarios; no ablation videos were generated. Four identical control conditions reused existing Phase 5B teacher records.
- Demonstration subsets were selected before rollouts and balanced across starts/routes: 5, 15, and 30 episodes. All three achieved 50/50 safe success.
- Calibrated homography achieved 50/50 with 22.07 mm median clearance; naive pixel scaling achieved 49/50 with one unintended contact and 14.35 mm median clearance.
- Smoothing enabled and disabled both achieved 50/50; median clearance was 22.07 and 24.22 mm respectively.
- Confidence/route filtering achieved 50/50; disabling it achieved 47/50, with three object-collision rollouts, one unintended-contact rollout, and one drop.
- Human-derived paths achieved 50/50. Straight-line paths achieved 0/50, with 50 object-collision and 50 unintended-contact rollouts plus five drops.
- Float32 differed from float64 by at most 1.80e-6 in action output, so it was rerun on all 50 scenarios and achieved 50/50 safe success.
- Float32 reduced checkpoint size from 31,942 to 17,795 bytes, but did not improve this small-model CPU benchmark: median/p95 latency was 0.0127/0.0143 ms versus 0.0121/0.0135 ms for float64.
- No ONNX, quantization framework, new dependency, optimization, or final presentation work was added.

## 2026-10-03 — Phase 7 submission preparation

- No new experiment or simulator rollout was run; the demo uses committed Phase 3, 4E, 5B and 6 artifacts.
- Published phone media was re-encoded video-only from a privacy-reviewed derived overlay; audio, chapters and descriptive source metadata were not copied.
- README result tables are generated from the committed dataset-quality, Phase 5B and Phase 6 JSON files and checked for staleness.
- The 70-second demo includes the phone motion, calibrated trajectory, human-derived DMP, GRU, expected straight-line collision, ablations and final 50-scenario comparison.
- Public clean-clone reproduction covers setup, lint, 25 tests, README/result validation, full demo decode and tracked-file privacy checks. Raw-video regeneration remains intentionally unavailable without private inputs.

## 2026-10-04 - Public numerical data reproduction

- Released 36 accepted trajectories containing 7,963 frame-level numerical observations after explicit author approval. Coordinates, confidence, frame order and flags match the private source strings exactly; timestamps and original media remain private.
- Reconciled every released row against its unchanged source CSV and recorded source/public SHA-256 hashes. All 36 refitted DMPs match the saved motion-prior bundle within absolute tolerance 1e-12.
- A fresh directory with its own locked environment and no private trajectories or raw MOVs passed lint, all 36 tests, data reconstruction, README/result consistency, complete demo decoding, and tracked-file privacy checks.
- The public-input diagnostic reproduced the expected direct-path collision and contact-free DMP placement. The five-seed DMP preflight and five saved-GRU rollouts each achieved 5/5 safe placements.
- Numerical experiment loaders now fall back to the released CSVs. Full sweeps and training were not repeated, and historical benchmark results and checkpoints were not changed. Video extraction and independent visual-quality validation still require private recordings.
