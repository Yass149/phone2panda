# Dataset card

## Summary

The private dataset contains 36 short overhead phone demonstrations of one
person moving a marked box from one of three start regions to a target by a
left or right route around a glasses case. The final validated split is 20
train, 4 validation and 12 test recordings; all 36 passed the Phase 3 quality
gate. Only aggregate reports, hashes without local paths, and a selected
privacy-reviewed derived plot are tracked publicly.

## Fields and derived data

Metadata supplies episode ID, filename, split, start region and route label.
Processing produces timestamps, direct/interpolated marker state, confidence,
homography-normalized `(x, y)`, calibration diagnostics and task checks. Phase
5 teacher records add state, task context, route one-hot values, gripper state,
bounded teacher action and source episode/confidence provenance.

## Collection and consent

The recordings were collected specifically for this project by its author.
They are not released because they contain personal recording content and
device metadata. Published media is a silent, metadata-stripped derived clip
showing only the task surface and hand; it should still be treated as a human
demonstration, not anonymous population data.

## Intended use

This dataset supports a single-person, single-camera proof of concept for
planar trajectory extraction, DMP retargeting and simulator policy
distillation. It is not a benchmark for human behaviour, identity, clinical
use, autonomous physical manipulation, or broad real-world generalization.

## Limitations and bias

There is one demonstrator, one phone viewpoint, one canvas design, one object
and one obstacle. The tracker observes a red point rather than full object
pose; height and rotation are not measured. Start/route combinations and
simulator geometry are controlled rather than representative of homes or
factories. The simulation results do not establish physical Panda safety.

## Access and retention

Raw MOV files and processed full trajectories remain local and are excluded
from Git history. `results/dataset_quality/source_manifest.json` provides
integrity hashes. Anyone receiving private recordings must establish separate
consent, access control, retention and deletion policies.
