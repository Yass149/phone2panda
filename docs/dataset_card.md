# Dataset card

## Summary

The private dataset contains 36 short overhead phone demonstrations of one
person moving a marked box from one of three start regions to a target by a
left or right route around a glasses case. The final validated split is 20
train, 4 validation and 12 test recordings; all 36 passed the Phase 3 quality
gate. The public release contains 7,963 numerical observations in 36 CSVs under
`data/public/trajectories/`, plus a schema and checksum manifest. Aggregate
reports, derived plots, teacher-training arrays and an 80-point DMP-prior bundle
are also tracked. Raw videos and the original timestamped processing outputs
remain private.

The recording split labels document collection. The DMP/GRU experiments use
all 36 accepted recordings as priors, with separate held-out simulator seeds.
They do not measure generalization to unseen human demonstrations.

## Fields and derived data

Metadata supplies episode ID, filename, split, start region and route label.
Processing produces timestamps, direct/interpolated marker state, confidence,
homography-normalized `(x, y)`, calibration diagnostics and task checks. Phase
5 teacher records add state, active DMP/phase targets, task context, route one-hot values, gripper state,
bounded teacher action and source episode/confidence provenance.

### Public numerical schema

| Fields | Meaning |
| --- | --- |
| `frame_index` | Original contiguous frame order; no recording date or time |
| `object_x`, `object_y` | Measured/interpolated object-marker coordinates in the normalized canvas |
| `smoothed_x`, `smoothed_y` | Previously processed object coordinates consumed by the DMP pipeline |
| `tracking_confidence` | Original tracker confidence in [0, 1] |
| `object_detected`, `object_interpolated` | Binary flags distinguishing direct observations and interpolation |
| `all_corners_direct`, `valid` | Binary calibration visibility and trajectory-validity flags |

CSV numerical strings and frame order are preserved exactly; export does not
round, resample, discard observations or change labels. Timestamps and repeated
episode/split/route columns are removed. Episode ID, collection split, start
zone and route live in `manifest.json`, alongside source and public CSV hashes.
The public validator rejects extra columns, non-finite numbers, bad frame order,
invalid flags, changed checksums and missing episodes.

## Collection and consent

The recordings were collected specifically for this project by its author.
Raw recordings are not released because they contain personal recording
content and device metadata. The author separately approved publication of the
sanitized numerical trajectories. Published media is a silent, metadata-stripped derived clip
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

Raw MOV files and original timestamped processing outputs remain local and are
excluded from Git history. `data/public/` permits reconstruction of the motion
priors and numerical experiments, but does not reconstruct original images,
audio or recording timestamps. `assets/motion_priors/` additionally contains
derived DMP control points, endpoint anchors and task geometry for a short
saved-model simulation without refitting.
`results/dataset_quality/source_manifest.json` provides
integrity hashes. Run `make verify-data` to validate the public release and
rebuild its 36 DMPs against the saved bundle. CSV checksums and original
numerical strings are exact. Refitted robot-space DMP points use an absolute
comparison tolerance of 1e-9 metres to accommodate cross-platform linear
algebra rounding; the audit reports the measured maximum difference.
Anyone receiving private recordings must establish separate consent, access
control, retention and deletion policies.
