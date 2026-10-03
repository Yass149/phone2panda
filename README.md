# Phone2Panda

**Research question:** can one person’s overhead phone demonstrations provide
path geometry that makes a Panda robot move an object around an obstacle more
safely than a direct controller?

**Contribution:** Phone2Panda turns 36 personal phone demonstrations into a
small library of calibrated left/right motion priors, scores both routes for a
new scene, and distils the selected safe behaviour into a 3,559-parameter GRU.
The demonstrations determine the robot's path; they are not labels or visual
decoration around a simulator policy.

## Demo

![Phone demonstration driving a Panda route](media/phone2panda_preview.gif)

[Watch the 70-second silent demo](media/phone2panda_demo.mp4) - phone
demonstration, calibrated path, DMP teacher, GRU policy, straight-line failure
and fixed-seed comparison. The published clip contains no audio or personal
device/location metadata.

## What is original here

This is not end-to-end behaviour cloning from a large public robot dataset. A
phone observes one person's 2D path around a real obstacle. Calibration turns
that path into embodiment-independent geometry; a route-aware DMP retargets it
to a Panda; and a compact recurrent policy learns the resulting closed-loop
actions. At evaluation time, both human-derived route families are considered
and the route/confidence gate keeps the safer candidate.

The useful result is not only 50/50 success. The same fixed scenarios reveal
which choices mattered: removing the human path reduced safe success to 0/50,
removing route filtering reduced it to 47/50, and naive image scaling reduced
the clearance margin and caused a contact.

## What changed my mind

| Initial assumption | Evidence | Decision |
| --- | --- | --- |
| Every calibration marker had to remain visible throughout a recording. | Object tracking stayed at 100%, while a forearm briefly hid one corner during otherwise valid motion. | Calibrate from stable start/end windows and separately gate camera drift and jitter. |
| A generic 140 mm obstacle was a reasonable first simulation proxy. | Contact-pair auditing found real Panda wrist collisions even when the object path was clear. | Measure the physical 40 mm obstacle, rebuild the geometry, and retain the failed setup as a stress test. |
| More demonstrations would automatically improve the policy. | The 5, 15 and 30 demonstration ablations all achieved 50/50 safe success. | Report saturation honestly; for this task, route selection and calibration mattered more than volume. |
| Float32 would be faster as well as smaller. | It cut checkpoint size by 44%, but was slightly slower in this CPU benchmark. | Keep the size result and make no speed claim. |

## Architecture

![Phone2Panda architecture](docs/architecture.svg)

Private HEVC recordings are decoded read-only, perspective-corrected from four
canvas markers and reduced to normalized red-marker trajectories. Accepted
paths drive a route-aware DMP Panda teacher; its state/action episodes train a
small normalized GRU. [Design details](docs/design.md).

## Calibrated headline results

<!-- BEGIN GENERATED RESULTS -->
The video dataset gate accepted **36/36 recordings**. On the same 50 held-out calibrated simulator scenarios:

| System | Safe success | Object collisions | Unintended contacts | Drops | Median placement error | Median clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Human-derived DMP teacher | 50/50 | 0 | 0 | 0 | 1.53 mm | 22.07 mm |
| 24-unit GRU | 50/50 | 0 | 0 | 0 | 1.95 mm | 21.72 mm |
| Straight line | 0/50 | 50 | 50 | 5 | 20.21 mm | -48.56 mm |

Ablations are one-factor-at-a-time on 50 fixed scenarios:

- Demonstration count did not change safe success: 50/50, 50/50 and 50/50 for 5, 15 and 30 demonstrations.
- Calibration improved the safety margin: 50/50 safe with 22.07 mm median clearance versus 49/50 and 14.35 mm for naive scaling.
- Smoothing did not improve success here: both settings achieved 50/50; median clearance was 22.07 mm enabled and 24.22 mm disabled.
- Route/confidence filtering improved safety: 50/50 safe when enabled versus 47/50, with 3 object-collision rollouts, 1 unintended-contact rollout and 1 drop when disabled.
- Float32 reduced the checkpoint from 31,942 to 17,795 bytes but was slightly slower in this CPU benchmark (0.0127 vs 0.0121 ms median).
<!-- END GENERATED RESULTS -->

These are simulator measurements for the physically calibrated **40 mm**
glasses-case obstacle. The earlier 140 mm Phase 4B–4D environment is an
**uncalibrated tall-obstacle stress test**, retained for failure analysis but
not used as a headline physical result. Machine-readable sources are
[`results/phase5b/evaluation.json`](results/phase5b/evaluation.json),
[`results/phase6/ablation_summary.json`](results/phase6/ablation_summary.json)
and [`results/phase6/precision_benchmark.json`](results/phase6/precision_benchmark.json).

## Quick start

The bootstrap installs uv and CPython 3.11.17 inside the checkout, creates
`.venv`, and never modifies global Anaconda.

```bash
make setup
make test                 # focused unit/integration tests
make smoke-demo           # decode and validate the committed 70 s demo
make evaluate-existing    # verify and print committed evaluation results
```

Full headless simulator reproduction, when the private processed trajectories
are available:

```bash
make phase4e   # calibrated four-controller comparison
make phase5b   # fixed train/validation/test GRU evaluation
make phase6    # fixed-seed ablations and precision comparison
```

These commands use pinned dependencies and fixed configuration/seed manifests.
They are compute-heavy and overwrite only reproducible result directories, not
raw recordings.

## How personal recordings change robot behaviour

The camera calibration maps the demonstrator’s route into a unit canvas. The
selected left/right curve is fit by a DMP, aligned to each seeded Panda
start/goal, and used during the transport phase; it is not merely a label. The
GRU then learns the DMP teacher’s state-to-action mapping while retaining route
and task context. Removing that human path produces the straight-line control:
0/50 safe successes and 50/50 object collisions on the fixed calibrated test.

Raw MOV files remain private. Only a short, silent, re-encoded task-surface clip
is published. See the [collection guide](docs/data_collection.md) and
[dataset card](docs/dataset_card.md).

## Limitations and failures

- This is simulation-only evidence in one calibrated task family, not proof of
  safe physical-robot transfer.
- One demonstrator, camera, object and obstacle limit population and scene
  generalization; the tracker observes a red point, not full 6D object pose.
- The 140 mm stress test exposed whole-arm contacts that a 2D object path could
  not prevent. The calibrated 40 mm route/confidence controller passed, but no
  general collision-aware planner was implemented.
- Demonstration count did not change safe success in this ablation, smoothing
  did not improve success, and float32 was smaller but slightly slower.
- A clean public clone cannot regenerate private vision trajectories. It can
  run tests, inspect fixed results, load checkpoints and validate the demo.

See [known issues](KNOWN_ISSUES.md) and the [model card](docs/model_card.md) for
the full scope and prohibited claims.

## Project map

- `src/phone2panda/pilot_validation/`: decode, calibration, tracking and gates
- `src/phone2panda/trajectories/`: smoothing and DMPs
- `src/phone2panda/sim/`: robosuite Panda environment and phased controller
- `src/phone2panda/policy/`: dependency-free normalized GRU
- `src/phone2panda/evaluation/`: fixed-scenario comparisons and ablations
- `configs/`: immutable experiment settings
- `results/`: selected machine-readable records and public-safe media

## Documentation, licence and citation

[Design](docs/design.md) · [Data collection](docs/data_collection.md) ·
[Dataset card](docs/dataset_card.md) · [Model card](docs/model_card.md) ·
[References and acknowledgements](docs/references.md)

Software is released under the [MIT License](LICENSE). Dependencies retain
their upstream licences; see [third-party notices](THIRD_PARTY_NOTICES.md).
Raw recordings are not distributed under this licence. Citation metadata is in
[`CITATION.cff`](CITATION.cff).
