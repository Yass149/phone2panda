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
