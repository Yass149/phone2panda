# Standards hardening plan

This branch improves software, machine-learning, robotics, and computer-vision
assurance without changing committed experimental results or claiming physical
robot safety. Work is strictly sequential: each task starts with a failing test,
ends with focused and full-suite verification, and is committed before the next
task begins.

## Working rules

1. Write a small test that states the requirement and confirm that it fails for
   the expected reason.
2. Make the smallest cohesive implementation that passes the test.
3. Refactor only while the focused tests remain green.
4. Run the full test and lint gates, inspect the diff, and commit the task.
5. Do not alter result files unless the corresponding experiment is rerun. Do
   not tune against held-out seeds.

## Ordered tasks and acceptance gates

### 1. Unambiguous safety outcomes

Centralise outcome classification so object-task success and safe success cannot
be confused. Retain the historical `task_success` field as a documented legacy
alias so existing result readers remain compatible.

Gate:

- truth-table tests cover placement failure, object collision, unintended robot
  contact, drop, and lost grasp;
- calibrated evaluation uses the shared definition;
- existing JSON readers and all tests remain compatible;
- Ruff and the full test suite pass.

### 2. Corner-marker shape validation

Add an interpretable L-shape score to corner candidates instead of accepting a
dark component solely by area, extent, and location. Keep the score and threshold
observable in diagnostics.

Gate:

- synthetic true L markers pass;
- solid rectangles and corner-adjacent dark impostors fail;
- perspective ordering and existing tracking tests pass;
- a read-only sample of decoded recordings retains all four opening and closing
  markers before any full-dataset rerun is considered.

### 3. Reproducible run provenance

Provide one helper for recording the Git commit, dirty state, configuration hash,
and `uv.lock` SHA-256 in future run manifests. Historical result files remain
untouched.

Gate:

- deterministic unit tests cover clean, dirty, and non-Git environments;
- new manifests contain no absolute paths;
- privacy checks, Ruff, and all tests pass.

### 4. Repository and CI safeguards

Run the full-history privacy audit in CI, add dependency-update configuration,
and extend high-confidence secret detection with tests. Keep CI actions pinned.

Gate:

- fixture tests detect representative secrets without flagging documented
  placeholders;
- the public-history audit passes locally;
- workflow syntax and the existing submission check pass.

### 5. Maintainability refactors

Reduce the highest-complexity functions one at a time, starting with rollout
execution. Extract named phase helpers and immutable inputs instead of adding
new branches or changing controller behaviour.

Gate for each function:

- characterization tests are committed first;
- serialized outputs are unchanged for fixed fixtures;
- the function clears the configured complexity threshold;
- focused tests, the full suite, and Ruff pass before moving on.

### 6. Coverage and static-analysis gates

Add a modest, ratchetable coverage threshold and type-check the stable pure-code
modules first. Pin any new development dependency and update `uv.lock` once.

Gate:

- clean-clone setup succeeds from the lock file;
- coverage cannot fall below the recorded baseline;
- selected modules pass the type checker;
- CI remains within its current resource envelope.

## Explicitly deferred work

These items need evidence or scope beyond a code-hardening branch and must not be
represented as solved here:

- camera intrinsics, lens-distortion calibration, reprojection error, and manual
  tracking accuracy require calibration captures and labelled ground truth;
- broader human-subject generalisation, repeated training seeds, and domain
  randomisation require new experiments;
- full-arm collision-aware IK or motion planning is a separate robotics design;
- ISO 10218-style hazard analysis, safety-rated stops, guarding, and physical
  speed/force validation require the target robot, site, and responsible safety
  review.

