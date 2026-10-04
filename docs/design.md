# Design

## Objective

Phone2Panda tests whether a short overhead phone demonstration can supply path
geometry that changes how a Panda robot transports an object around an
obstacle. The system is deliberately small: deterministic vision, a
human-derived DMP teacher, and a compact recurrent policy.

![Architecture](architecture.svg)

## Data flow

1. PyAV decodes the untouched iPhone HEVC stream. SHA-256 is checked before
   and after processing; QuickTime display metadata does not control rotation.
2. Four custom L markers define a planar homography. Marker order in decoded
   frames determines orientation and opening/closing windows check camera
   stability.
3. HSV segmentation tracks the red box marker. Short gaps may be interpolated,
   but direct coverage, confidence and dropped frames remain explicit metrics.
4. Perspective-corrected points become unit-canvas coordinates. Start region,
   route side, footprint-adjusted obstacle clearance and target placement are
   checked before a trajectory is accepted.
5. A two-dimensional discrete DMP is fit to an accepted route and retargeted
   by a similarity transform into the Panda workspace. Pick, grasp, lift,
   transport, lower, release and retreat remain explicit phases.
6. The DMP + route/confidence controller generates teacher episodes. A
   normalized 24-unit GRU maps 22 state/reference/context features to seven
   bounded actions. It receives active DMP targets and retains explicit phases,
   so this is reference tracking rather than autonomous route learning.
   Episode boundaries and human-trajectory provenance are retained.

## Coordinates and control

Canvas coordinates use `(0, 0)` at the calibrated top-left. Retargeting maps
canvas `y` to robot `x` and canvas `x` to robot `y`; endpoints are aligned to
the seeded start and goal without removing lateral route shape. The simulated
40 mm obstacle matches the measured glasses-case height. The end-effector
transport height, 0.847715 m, is derived from the cube half-height, 12 mm
desired bottom clearance and measured grasp offset.

The policy output is clipped to the robosuite OSC_POSE action bounds. The
gripper channel is part of both the teacher record and policy output. Fixed
seeds define train, validation and held-out geometry before training.

## Safety accounting

Contacts are classified as intended gripper-cube, object-obstacle,
robot-obstacle, robot-table, robot self-contact, or cube-table. "Safe success"
requires successful placement, retained grasp until release, zero object
collisions, zero unintended robot contacts and zero drops. Clearance is the
minimum planar footprint clearance and may be negative when geometry overlaps.

The earlier 140 mm obstacle in Phases 4B–4D is an **uncalibrated tall-obstacle
stress test**. It is retained for failure analysis but is not a headline model
of the physical setup.

## Reproducibility boundaries

Dependencies and Python 3.11.17 are locked locally. Public results include
fixed seed manifests, configurations, per-rollout records and compact
checkpoints. The derived 80-point DMP bundle permits five saved-model simulation
rollouts from a clone without decoding videos or retraining. The sanitized
36-episode numerical release also allows refitting motion priors and rerunning
controller, training and ablation workflows. Raw phone recordings and original
timestamped processing outputs remain private; video extraction and independent
visual quality validation still require those unreleased inputs. Test seeds are
held out from training; the demonstration library itself is shared across splits.
