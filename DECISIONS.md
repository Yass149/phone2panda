# Decisions

## 2026-10-02 — Project-local runtime

Use a project-owned `uv` binary, managed CPython 3.11.17 installation, `.venv`, local caches, and a committed `uv.lock`. The global Anaconda installation is not used or modified.

## 2026-10-02 — Match perception to the recorded pilots

The action plan proposed ArUco markers, but the supplied pilots contain four hand-made black L corner markers and a red square on the black box. The pilot pipeline therefore uses connected-component L-marker detection and HSV red-marker segmentation. Requiring nonexistent ArUco IDs would make the real recordings impossible to validate.

## 2026-10-02 — Separate direct visibility from usable calibration

The phone is fixed. A robust temporal reference establishes the homography, while every frame still reports whether all four markers were directly observed. Missing corner observations may use the static reference for overlays and coordinates; the 95% gate applies to the opening and closing calibration windows.

## 2026-10-02 — Conservative geometry checks

Report both centerline and footprint-adjusted obstacle clearance. Gate on the latter using a configured 0.05-canvas-unit footprint radius. Starting and final placement are also checked both by marker center and by the contracted zone needed to contain that footprint.

## 2026-10-02 — Decoded marker order controls orientation

Ignore Finder/QuickTime display orientation when the decoded frame geometry is already correct. Require TL/TR above BL/BR and left markers left of right markers, then apply zero rotation. Refuse ambiguous ordering instead of silently rotating or transcoding originals.

## 2026-10-02 — Gate fixed-camera calibration integrity, not manipulation visibility

The original pilot gate required all four calibration markers to be directly visible in 95% of every frame. Repeated pilots showed that this rejects valid manipulation whenever a forearm temporarily occludes one marker, despite a fixed camera, 100% usable homography coverage, stable markers, and perfect object tracking. The revised gate requires 95% simultaneous marker visibility in both the opening and closing hands-free windows, no more than 3 px RMS marker jitter within those windows, and no more than 5 px start-to-end marker drift. Whole-video direct coverage remains a reported diagnostic. This is a change in what is measured, not a reduction of the thresholds.

## 2026-10-02 — Incremental dataset revalidation

Replacement runs decode only the named episode IDs, merge their reports into the existing 36-episode aggregate, and reuse untouched trajectories and results. Superseded originals remain preserved and hashed.

## 2026-10-02 — Phase 3 artifact policy

Store one CSV trajectory per recording and one aggregate quality report. Generate overlays and plots only for rejected recordings plus four deterministic representative successes. Keep raw MOV files and generated data out of Git, and verify raw hashes before and after processing.

## 2026-10-02 — Minimal Phase 4A simulator

Use robosuite 1.5.1 with MuJoCo 3.2.7 and Panda `OSC_POSE` fixed-impedance control. Extend the small built-in Lift scene with one fixed obstacle and one visual target rather than introducing another simulator. Use accepted `ep_001` as the representative left-route demonstration, fit a two-dimensional discrete DMP, and map canvas y to robot x and canvas x to robot y. Keep approach, grasp, lift, DMP transport, lower, release, and retreat as explicit logged phases.

## 2026-10-02 — Phase 4B fixed scenarios and collision semantics

Use 50 deterministic scenarios shared by all four methods, cycling through the three recorded start zones and applying only seeded centimetre-scale jitter. Retarget demonstration endpoints to each scenario with a two-dimensional similarity transform. Preserve the validated Phase 4A object-obstacle contact as the task collision gate; report Panda-link obstacle contacts separately because a two-dimensional object path does not model whole-arm clearance. Confidence-aware route selection requires at least 95% direct tracking coverage and 0.80 mean tracking confidence, then chooses the left or right DMP with the larger predicted footprint-adjusted clearance.

## 2026-10-02 — Phase 4C contact taxonomy and stop gate

Classify MuJoCo geom pairs at every control step: gripper-cube is intended manipulation; robot-obstacle, robot-table, and robot self-contact are unintended; cube-table is reported separately. Require zero unintended contacts in all five safety-preflight rollouts before permitting a corrected 50-episode evaluation. The conservative height, enlarged robot-envelope route score, and goal-side lowering posture did not pass this gate, so no corrected 50-run result or success video is claimed.

## 2026-10-02 — Phase 4D bounded lateral-placement attempt

Keep the selected human DMP transport path unchanged and test only three route-side Y offsets. Clamp staging coordinates to the configured workspace, lower at the lateral point, place horizontally, and retreat along the same corridor. Stop without a 50-run evaluation if no offset achieves both 5/5 placement and zero unintended contact. All three offsets failed, so collision-aware IK or planning is now required.

## 2026-10-02 — Phase 4E physical obstacle calibration

Treat the earlier 0.14 m obstacle as an uncalibrated tall-obstacle stress test and preserve all of its results unchanged. Model the measured glasses case as 0.04 m tall. With a maximum 0.022 m cube half-height, 0.012 m cube-bottom clearance, and 0.013715 m measured grasp offset, use a derived 0.847715 m end-effector transport height. The deliberately unsafe straight diagnostic passes when it demonstrates the expected obstacle failure; only the DMP run is required to place successfully, retain grasp, and have zero unintended contacts before calibrated preflight.

## 2026-10-02 — Phase 5A compact recurrent policy

Use a dependency-free, normalized 24-unit GRU rather than ACT, LeRobot, SmolVLA, or another framework. The GRU provides deterministic initialization and training, a 3,559-parameter checkpoint suitable for modest hardware, and direct controller-latency measurement. Limit Phase 5A to overfitting exactly five successful calibrated DMP episodes and require 5/5 safe closed-loop replay before any broader training.

## 2026-10-03 — Phase 5B split isolation

Fix the balanced seed manifest before generating rollouts: 120 training, 20 episode-disjoint validation, and 50 untouched testing scenarios. Fit normalization only on training data, select checkpoints only by validation loss, and compare the GRU with its DMP teacher on identical held-out geometry. Keep the validated 24-unit architecture because validation remained stable; do not tune after observing test outcomes.

## 2026-10-03 — Phase 6 fixed controls and precision scope

Use the untouched Phase 5B test manifest for every ablation and choose balanced demonstration IDs by deterministic round-robin over start/route cells before running outcomes. Reuse Phase 5B DMP records only for identical calibrated, smoothed, filtered, human-path control conditions. Compare the existing float64 GRU only with a direct float32 checkpoint; because outputs differ numerically, require a 50-scenario float32 replay, but do not add conversion or quantization frameworks.

## 2026-10-03 — Phase 7 public artifact boundary

Publish only a privacy-reviewed, silent derived phone excerpt and a 70-second composite demo. Strip descriptive metadata, chapters, audio and non-video streams; retain only unavoidable technical MP4 fields. Generate README numbers from committed JSON and make the public clean-clone check validate existing results rather than rerun experiments. Keep raw recordings and full processed trajectories private, and describe that reproduction boundary explicitly.
