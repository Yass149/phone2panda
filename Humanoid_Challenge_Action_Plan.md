# Humanoid Internship Challenge Action Plan

## Project Decision

**Working title:** Phone2Panda: Efficient Robot Manipulation from Monocular Human Demonstrations

**Research question:** Can a small set of personally recorded smartphone demonstrations provide an obstacle-aware motion prior for a simulated Panda arm, and can successful retargeted rollouts be distilled into a compact policy without losing task success?

**Task:** Pick up a marked cube, carry it around a central obstacle, and place it in a marked target zone. Start positions, target positions, and obstacle placement vary between episodes.

The obstacle matters. Without it, a straight-line controller can solve the task and the human data adds little. With it, the shape and timing of the human demonstration become useful.

## What Success Looks Like

The minimum credible submission contains:

1. Personally collected phone videos and a documented collection protocol.
2. A reproducible pipeline that extracts trajectories from those videos.
3. A calibrated mapping from the human workspace to a simulated robot workspace.
4. A Panda arm completing the task in simulation using those trajectories.
5. Seeded evaluation against at least one simple baseline.
6. Honest failure analysis, runnable instructions, tests, and example videos.

The strong version additionally contains:

1. A Dynamic Movement Primitive (DMP) fitted to human paths and adapted to unseen starts and goals.
2. A compact imitation policy trained on successful retargeted simulator rollouts.
3. An inference benchmark comparing eager, compiled, and optionally ONNX or quantised execution.
4. Ablations showing exactly when the collected data helps.

Do not make SmolVLA or a world model part of the critical path. They are stretch work after the complete pipeline runs and has measured results.

## Manual Setup

### Materials

- Smartphone capable of 720p or 1080p at 30 FPS.
- Stable overhead or high-oblique mount. A tripod is ideal; stacked books are acceptable if the phone cannot move.
- A4 or A3 white sheet as the workspace.
- Four printed ArUco markers for workspace corners, IDs 0-3 from `DICT_4X4_50`.
- One marker on top of the movable cube, ID 4.
- One marker or clearly printed shape defining the target, ID 5.
- A cube or small box approximately 4-6 cm wide.
- A rectangular obstacle approximately 10-15 cm long and 5-8 cm wide.
- Even lighting with minimal glare and no moving shadows.

Keep faces, voices, addresses, screens, and personal information out of the recordings. Remove audio before publishing samples.

### Physical Layout

1. Put one corner marker near each corner of the white sheet.
2. Measure and record the real distances between marker centres in millimetres.
3. Place the obstacle near the middle so a direct start-to-target path is blocked.
4. Mark five possible cube starts and three possible targets with small removable dots.
5. Keep every marker visible throughout each recording.
6. Once recording begins, do not move the phone or corner markers.

### Pilot Before Full Collection

Record only five pilot demonstrations first. Run the tracking pipeline and check:

- All four corner markers are detected in at least 95% of frames.
- The cube marker is visible during approach, transport, and placement.
- The table homography is stable and trajectories do not jump.
- The hand does not cover the cube marker for long periods.
- The object path clearly passes around the obstacle.

If these fail, change lighting, camera angle, marker size, or hand grip before collecting the full dataset.

### Demonstration Protocol

Each episode should last roughly 4-8 seconds:

1. Start with hands outside the workspace for one second.
2. Reach toward the cube without rushing.
3. Grasp and lift it clearly.
4. Move around the obstacle, sometimes clockwise and sometimes anticlockwise.
5. Lower it into the target zone.
6. Release and remove the hand.
7. Hold the final scene for one second.

Record 36-50 successful demonstrations:

- Balance clockwise and anticlockwise routes.
- Cover all five start areas and three target areas.
- Vary speed and path curvature naturally.
- Reserve several start-target combinations entirely for testing.
- Optionally record five failures separately for failure-detection examples; never label them as successful training demonstrations.

Create `metadata.csv` while recording with:

```text
episode_id,video_file,start_id,target_id,route,success,split,notes
```

Split by start-target configuration, not by individual frames. Frames from one video must never appear in both training and testing.

## System Design

### 1. Data Ingestion and Validation

- Read video metadata and the manually maintained episode table.
- Strip audio and preserve the untouched originals outside Git.
- Compute checksums for provenance.
- Reject corrupt, truncated, or incorrectly labelled episodes.
- Cache decoded frame timestamps rather than repeatedly decoding entire videos.

### 2. Perception and Calibration

- Detect ArUco markers with OpenCV.
- Estimate a per-frame homography from image pixels to a normalised table frame.
- Track the cube marker and target marker with a confidence score.
- Optionally use MediaPipe Hands only for grasp-phase timing; do not make it a hard dependency.
- Interpolate only short tracking gaps. Mark long gaps invalid rather than inventing motion.

Output one processed record per timestep:

```text
episode_id, t, object_x, object_y, target_x, target_y,
route, phase, tracking_confidence, valid
```

### 3. Trajectory Processing

- Transform tracked points into the calibrated table frame.
- Smooth with a Savitzky-Golay filter or cubic spline.
- Resample to the simulator policy rate, initially 20 Hz.
- Infer simple phases: approach, grasp, lift, transport, lower, release.
- Use a phase-based height profile because monocular top-down video gives reliable x-y but weak depth.
- Fit one or more DMPs to the transport trajectories.
- Keep clockwise and anticlockwise motion primitives separate, then choose the route with greater predicted obstacle clearance.

### 4. Simulation and Retargeting

Primary simulator: `robosuite` with a Panda arm and `OSC_POSE` control. It has a current API and provides direct end-effector delta control.

Preferred scene:

- Panda arm.
- Cube matching the measured relative dimensions of the real object.
- Target zone.
- Central obstacle.
- One front camera and one overhead camera.
- Randomised start, goal, lighting, colours, and small camera perturbations.

Map normalised table coordinates into bounded robot workspace coordinates. Convert each desired DMP waypoint into an end-effector delta action and add a binary gripper command. Clip all actions and fail safely when tracking confidence is too low.

Use LIBERO only if a smoke test works within two hours. Its official installation pins an old Python and PyTorch stack, so it must not block the submission. A clean robosuite implementation satisfies the challenge's request for a simple simulation environment; a LIBERO adapter is a bonus.

### 5. Learned Policy

Once DMP retargeting succeeds:

1. Randomise starts, goals, and obstacle positions.
2. Execute the DMP controller.
3. Save successful simulated observations, robot states, language instruction, and actions.
4. Train a compact ACT policy through LeRobot, or a small GRU policy if ACT integration threatens the deadline.
5. Evaluate the learned policy on the same fixed seeds as all baselines.

SmolVLA is optional. Attempt it only if the core system is complete and it can overfit five episodes within two hours. Otherwise document why ACT was chosen: smaller training cost, faster iteration, and easier controlled comparison.

### 6. Runtime Optimisation

- Use `torch.inference_mode()` and fixed tensor shapes.
- Benchmark warm and cold inference separately.
- Compare PyTorch eager against `torch.compile` where supported.
- Optionally export to ONNX Runtime or apply dynamic quantisation on CPU.
- Report median and p95 latency, throughput, peak memory, model size, and task-success change.
- Never claim a speedup without the raw benchmark file and hardware details.

## Architecture

```mermaid
flowchart LR
    A[Personally recorded phone videos] --> B[Validation and metadata]
    B --> C[ArUco detection and table calibration]
    C --> D[Trajectory extraction and confidence]
    D --> E[Phase segmentation and smoothing]
    E --> F[DMP motion-prior library]
    F --> G[Human-to-Panda retargeter]
    G --> H[robosuite Panda simulation]
    H --> I[Successful rollout dataset]
    I --> J[Compact ACT or GRU policy]
    J --> K[Inference optimisation]
    H --> L[Seeded evaluator]
    J --> L
    K --> L
    L --> M[Metrics, plots, videos, failure analysis]
```

### Module Boundaries

```text
src/phone2panda/
  data/          video ingestion, metadata, schemas, checksums
  vision/        marker detection, homography, confidence, tracking
  trajectories/ smoothing, resampling, phases, DMP fitting
  retarget/      workspace transforms, route selection, action generation
  sim/           backend interface, robosuite environment, optional LIBERO adapter
  policies/      ACT adapter, GRU fallback, checkpoints
  evaluation/    seeded rollouts, metrics, ablations, failure taxonomy
  optimisation/  compile/export/quantisation and latency benchmarking
```

The simulator must sit behind an interface so the perception and trajectory code does not depend on robosuite or LIBERO internals.

## Baselines and Experiments

Run at least 50 fixed-seed episodes per method; use 100 if compute permits.

Methods:

1. Straight-line waypoint controller with no human data.
2. Nearest raw human trajectory replay.
3. DMP retargeting from human demonstrations.
4. DMP plus route selection and confidence filtering.
5. Compact learned policy distilled from successful DMP rollouts.
6. Optimised learned policy with identical weights and evaluation seeds.

Metrics:

- Task success rate.
- Obstacle collision rate.
- Object drop rate.
- Mean completion steps.
- Path-length efficiency.
- Minimum obstacle clearance.
- Median and p95 inference latency.
- Peak memory and checkpoint size.

Ablations:

- 5 vs 15 vs 30 human demonstrations.
- Calibration enabled vs naive pixel scaling.
- Smoothing enabled vs disabled.
- Confidence filtering enabled vs disabled.
- Human-derived obstacle path vs straight-line path.
- Eager vs compiled or exported inference.

Report confidence intervals for success rates. Use the same seeds and environment settings for fair comparisons.

## Failure Taxonomy

Every failed rollout should be assigned one primary category:

- Perception or calibration failure.
- Retargeting/workspace mismatch.
- Grasp failure.
- Collision.
- Object dropped during transport.
- Placement outside target.
- Policy drift or compounding error.
- Simulator or infrastructure failure.

Include representative videos of at least two failures. Honest negative results will improve the submission.

## Repository Layout

```text
phone2panda/
  README.md
  LICENSE
  pyproject.toml
  Makefile
  .github/workflows/ci.yml
  configs/
  data/
    README.md
    metadata.csv
    samples/
    processed/
  docs/
    design.md
    data_collection.md
    experiments.md
    model_card.md
  notebooks/
    exploration.ipynb
  scripts/
    make_calibration_board.py
    validate_recordings.py
    extract_trajectories.py
    run_retargeted_demo.py
    collect_sim_rollouts.py
    train_policy.py
    evaluate.py
    benchmark_inference.py
  src/phone2panda/
  tests/
  assets/
    demo.mp4
    architecture.png
    results.png
  results/
    metrics.json
    rollouts.csv
    latency.json
```

Notebooks are for exploration only. The complete pipeline must run through scripts or a Makefile.

## Acceptance Tests

- Calibration transforms each corner marker to the expected table coordinate within tolerance.
- Trajectory resampling preserves endpoints and produces monotonic timestamps.
- Phase segmentation handles a synthetic perfect episode.
- Retargeted actions remain within configured workspace and action bounds.
- Simulator reset is deterministic for a fixed seed.
- A smoke test runs one complete episode headlessly.
- Evaluation writes both per-rollout and aggregate metrics.
- A fresh environment can reproduce one demonstration using documented commands.

## Seven-Day Schedule

### Day 0: Environment and Pilot

- Create the public repository privately first, then make it public only when safe.
- Bootstrap packaging, tests, linting, configuration, and experiment logs.
- Make the calibration board.
- Run a simulator smoke test with random actions.
- Record five pilot videos and validate marker detection.

**Exit condition:** one video becomes a stable table-frame trajectory and one Panda environment runs headlessly.

### Day 1: Full Data Collection

- Fix any pilot problems.
- Record 36-50 demonstrations and complete metadata while recording.
- Extract and visualise every trajectory.
- Reject poor demonstrations explicitly.

**Exit condition:** processed dataset with train/validation/test splits and a quality report.

### Day 2: Retargeting

- Implement smoothing, phase segmentation, workspace mapping, and gripper timing.
- Fit DMPs and execute one full simulated task.
- Save success and failure videos.

**Exit condition:** at least one deterministic successful Panda rollout driven by personal data.

### Day 3: Generalisation and Baselines

- Add route selection and environment randomisation.
- Run straight-line, replay, and DMP baselines.
- Fix the evaluation seeds and produce initial metrics.

**Exit condition:** repeatable baseline table and at least 50 evaluated episodes per method.

### Day 4: Policy Distillation

- Generate successful randomised simulator rollouts.
- Train ACT or the GRU fallback.
- Verify the model can overfit a tiny subset before a full run.

**Exit condition:** learned policy completes at least some held-out tasks and has a saved training curve.

### Day 5: Performance and Ablations

- Benchmark learned policy latency and memory.
- Try compile/export/quantisation one at a time.
- Run the highest-value ablations.

**Exit condition:** measured performance table with raw machine-readable outputs.

### Day 6: Presentation

- Build concise figures and a 60-90 second demo video.
- Write README around the research question, method, results, and failures.
- Add architecture and data-collection documentation.
- Remove secrets, personal metadata, large accidental files, and generated clutter.

### Day 7: Reproduction and Submission

- Clone into a clean directory and follow the README exactly.
- Run tests and one end-to-end smoke path.
- Check every number in README against committed result files.
- Make the repository public.
- Submit several hours before the deadline.

## Stop Rules

- If LIBERO is not running within two hours, use robosuite and continue.
- If pilot marker coverage is below 95%, do not collect the full dataset yet.
- If a policy cannot overfit five episodes, debug that before any long training run.
- If ACT integration consumes more than four hours without a successful rollout, use the GRU fallback.
- If SmolVLA cannot overfit five episodes within two hours, stop and document the decision.
- If the learned policy underperforms DMPs, report that result and make the DMP system the main contribution.
- Do not start a world-model extension before the core evaluation and README are complete.

## Presentation Rules

- Lead with one sentence, one diagram, one demo GIF, and one results table.
- State exactly how personal data influenced robot actions.
- Separate measured results from future work.
- Include failed approaches and why they failed.
- Avoid unsupported adjectives such as "novel", "state of the art", or "production ready".
- Cite external repositories, models, papers, and copied configuration.
- Make every chart reproducible from a committed script and raw result file.

## Final Submission Checklist

- Public GitHub repository opens without authentication.
- README includes setup, quick start, design choices, results, limitations, and acknowledgements.
- At least one sample of personally collected data is visible.
- Demo video clearly shows source data and corresponding robot behaviour.
- No faces, audio, addresses, API keys, tokens, or private paths are present.
- Dependencies are pinned.
- Tests pass.
- One-command smoke demo works.
- Metrics have seeds, hardware, sample counts, and confidence intervals.
- CV link and application fields are ready.

## Technical References

- LIBERO official repository: https://github.com/Lifelong-Robot-Learning/LIBERO
- robosuite controller documentation: https://robosuite.ai/docs/modules/controllers.html
- LeRobot repository and agent guide: https://github.com/huggingface/lerobot
- SmolVLA technical introduction: https://huggingface.co/blog/smolvla
- OpenCV ArUco documentation: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

