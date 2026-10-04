# Model card: compact GRU policy

## Model

The policy is a deterministic, normalized GRU with 22 input features, 24
hidden units, seven outputs and 3,559 parameters. It was selected over larger
frameworks for reproducibility, modest hardware requirements and controlled
latency measurement. The action represents OSC_POSE translation, rotation and
gripper control.

The 22 features include the current DMP/phase reference target and its delta
from the end effector. The learned component is a reference-tracking controller,
not an independent route planner or an end-to-end video-to-action policy.
Route selection and pick/place phase sequencing remain outside the GRU.

## Training data

The teacher is the calibrated DMP + route/confidence controller. Each teacher
path traces back to an accepted human recording, then smoothing, DMP fitting,
scenario retargeting and bounded Panda control. Phase 5B uses 120 balanced
training scenarios and 20 episode-disjoint validation scenarios. The 50 fixed
test seeds are never used for fitting or checkpoint selection.
These are held-out simulator seeds, not held-out demonstrations: all 36 accepted
human recordings form the motion-prior library used across simulator splits.

## Evaluation

On the 50 held-out calibrated scenarios, both GRU and DMP teacher achieved
50/50 safe successes with zero object collisions, unintended contacts or
drops. Median placement error was 1.95 mm for the GRU and 1.53 mm for the
teacher; median clearance was 21.72 mm and 22.07 mm respectively. The gate was
45/50 safe successes.

The float64 checkpoint is 31,942 bytes. A float32 checkpoint is 17,795 bytes
and also achieved 50/50, but its small-model NumPy benchmark was slightly
slower: 0.0127/0.0143 ms median/p95 versus 0.0121/0.0135 ms. Maximum action
output difference was 1.80e-6.

## Safety definition

Safe success requires placement within the configured threshold, no
object-obstacle collision, no robot-obstacle/table/self contact, no drop and a
retained grasp until release. These checks use simulator state and contacts;
they are not a physical safety certification.

## Limitations and prohibited claims

The model operates only within the calibrated task family and explicit
controller phases. It has no vision input, force sensing, uncertainty-aware
fallback, whole-arm planner or real-robot validation. Do not claim general
robot autonomy, transfer to arbitrary scenes, or safe physical deployment from
these results. A new camera, object, obstacle, workspace or robot requires
fresh calibration and safety evaluation.

## Artifacts

- Float64 checkpoint: `results/phase5b/gru_policy.npz`
- Float32 checkpoint: `results/phase6/gru_policy_float32.npz`
- Schema/provenance: `results/phase5b/dataset_schema.json`
- Fixed split manifest: `results/phase5b/split_manifest.json`
- Evaluation: `results/phase5b/evaluation.json`
- Derived motion priors: `assets/motion_priors/` (80 control points per route,
  endpoint anchors, geometry and hashes; no frame-level recordings)
- Public numerical trajectories: `data/public/` (36 episodes, 7,963 observations;
  no timestamps or source media; sufficient to refit the motion priors)
- Public smoke test: `make simulate-demo` (five fixed held-out cases)
