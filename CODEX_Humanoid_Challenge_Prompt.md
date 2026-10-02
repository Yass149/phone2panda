# Codex Build Prompt: Phone2Panda

You are the lead robotics research engineer and ML systems engineer for a time-boxed internship challenge. Work directly in the attached repository and carry the project through implementation, verification, measured experiments, documentation, and a public-ready GitHub submission.

## Objective

Build **Phone2Panda**, a reproducible system that uses personally collected smartphone demonstrations to drive a Panda robot through an obstacle-aware pick-and-place task in simulation. The core system must convert real monocular video into calibrated trajectories, retarget them to the robot, evaluate task performance, and clearly demonstrate that the applicant's collected data affects robot behaviour.

After the core system works, distil successful retargeted simulator rollouts into a compact imitation policy and measure inference efficiency. The final repository must be technically credible, simple to run, honest about failures, and visually clear.

Read `Humanoid_Challenge_Action_Plan.md` in full before changing files. Treat it as the product and research specification. If the repository conflicts with the plan, inspect the existing work carefully and preserve anything valid rather than blindly replacing it.

## Hard Constraints

1. The applicant's personally collected data must materially influence the robot trajectory or policy.
2. Never fabricate a result, benchmark, success rate, screenshot, video, citation, or completed experiment.
3. Clearly distinguish implemented, measured, planned, and failed work.
4. Do not make SmolVLA, LIBERO, reinforcement learning, or a world model part of the critical path.
5. The project must remain runnable on modest hardware, with optional Colab GPU acceleration.
6. Use official documentation and primary repositories for unstable APIs.
7. No production logic may live only in a notebook.
8. Never publish private video, audio, EXIF metadata, absolute local paths, tokens, or personal information.
9. Keep the architecture modular and the dependency surface small.
10. Optimise only after correctness is measured.

## Working Method

Maintain these files throughout the work:

- `TASKS.md`: phase checklist with completed items and blockers.
- `DECISIONS.md`: dated architecture decisions and rejected alternatives.
- `EXPERIMENTS.md`: every experiment, command, configuration, seed, hardware, result, and conclusion.
- `KNOWN_ISSUES.md`: reproducible problems and current workarounds.

At the beginning of each phase:

1. Inspect the relevant files and current git diff.
2. State the phase goal and acceptance test.
3. Implement the smallest complete vertical slice.
4. Run tests and the acceptance command.
5. Record the result before moving on.

Do not ask the user broad design questions already answered by the action plan. Ask only for required physical actions or missing facts. When manual data collection is needed, provide a short exact checklist and wait for the user to confirm the files are present before processing them.

## Technical Direction

### Core stack

- Python 3.11 where possible.
- `opencv-contrib-python` for ArUco detection and homography.
- NumPy, pandas or Polars, SciPy, PyYAML, and Pydantic/dataclasses.
- PyTorch for learned policies.
- robosuite with Panda and fixed-impedance `OSC_POSE` as the primary simulator.
- pytest, Ruff, and mypy or Pyright where practical.
- Matplotlib/Seaborn for reproducible figures.

Keep simulator code behind a `SimBackend` protocol. A LIBERO adapter is optional and must be attempted only after the robosuite path passes its smoke test. LIBERO's older dependency stack must not contaminate the core environment; isolate it if used.

### Data contract

Define explicit schemas for:

- Episode metadata.
- Per-frame detections.
- Calibrated trajectories.
- Robot rollout transitions.
- Aggregate and per-rollout evaluation metrics.

Validate schemas at boundaries and version processed data. Never silently coerce malformed input.

### Perception

- Generate a printable ArUco calibration board with IDs 0-5.
- Detect corner, object, and target markers.
- Estimate a table homography and tracking confidence.
- Visualise detections overlaid on sample frames.
- Interpolate only short gaps under a configurable threshold.
- Make MediaPipe hand tracking optional rather than required.

### Trajectories

- Map trajectories to a normalised table frame.
- Smooth and resample at a configurable policy frequency.
- Infer approach, grasp, lift, transport, lower, and release phases using transparent rules.
- Fit clockwise and anticlockwise DMP motion primitives.
- Select a route using predicted obstacle clearance.
- Preserve raw, smoothed, and retargeted trajectories for comparison.

### Simulation

- Build a Panda pick-and-place scene with a cube, target, and obstacle.
- Randomise starts, goals, obstacle placement, lighting, colours, and small camera offsets within configured bounds.
- Use end-effector delta actions and explicit gripper commands.
- Clip actions, enforce workspace bounds, and log controller saturation.
- Render headlessly and save selected rollout videos.

### Learned policy

- Generate a dataset from successful DMP rollouts under randomisation.
- First prove the dataset and training loop by overfitting five episodes.
- Prefer a compact ACT policy through LeRobot.
- If ACT integration exceeds four hours without a successful rollout, implement a small GRU fallback and record the decision.
- SmolVLA is a stretch experiment only after the complete measured system and README exist.

### Optimisation

- Establish eager inference correctness and latency first.
- Benchmark with warm-up, synchronisation, repeated trials, and hardware metadata.
- Try `torch.compile`, ONNX Runtime, or dynamic quantisation independently.
- Re-run task evaluation after every optimisation.
- Commit raw JSON latency results and scripts that regenerate the table.

## Required Evaluation

Use fixed seeds and identical scenarios for:

1. Straight-line controller without human data.
2. Nearest human trajectory replay.
3. DMP retargeting.
4. DMP plus route selection and confidence filtering.
5. Learned policy, if completed.
6. Optimised learned policy using the same weights, if completed.

Measure task success, collisions, drops, placement errors, steps, path efficiency, obstacle clearance, median/p95 latency, memory, and checkpoint size. Produce confidence intervals for success rates. Save per-rollout records, not only aggregates.

Required ablations are demonstration count, calibration, smoothing, confidence filtering, human path versus straight-line path, and inference mode where available.

## Phase Gates

### Phase 0: Repository audit and bootstrap

- Inspect the repository, environment, available compute, and existing files.
- Create the package layout, configuration, CI, tests, and experiment records.
- Produce a dependency plan before installation.
- Verify imports and one unit test.

### Phase 1: Simulator smoke test

- Make a Panda environment run headlessly.
- Step random bounded actions.
- Save one short video and environment metadata.
- If LIBERO was selected and is not working after two hours, switch to robosuite.

### Phase 2: Pilot-data tooling

- Generate the calibration board and collection guide.
- Ask the user to record five pilot episodes.
- Validate marker coverage, calibration stability, and object tracking.
- Do not request the full dataset until pilot acceptance passes.

### Phase 3: Dataset processing

- Process full recordings only after the user provides them.
- Produce quality reports and visual overlays.
- Build geometry-disjoint train, validation, and test splits.

### Phase 4: Retargeting and DMP controller

- Implement phases, height profile, workspace mapping, and DMPs.
- Achieve and record one successful deterministic rollout.
- Then run baseline evaluation on fixed seeds.

### Phase 5: Policy distillation

- Collect successful simulator trajectories.
- Overfit five episodes.
- Train the selected compact policy.
- Evaluate on held-out configurations.

### Phase 6: Performance work

- Profile first.
- Apply one optimisation at a time.
- Report speed, memory, and success trade-offs.

### Phase 7: Submission quality

- Generate plots and a 60-90 second demo video.
- Write a concise README with a visible result table and failure section.
- Run the project from a clean clone or clean environment.
- Audit privacy, licences, citations, file sizes, and secret exposure.

## Stop Rules

- Stop full data collection if marker coverage is below 95% on pilots.
- Stop long training if the model cannot overfit five episodes.
- Stop ACT integration after four unproductive hours and use the GRU fallback.
- Stop SmolVLA after two hours if it cannot overfit five episodes.
- Do not begin a world model until core evaluation and documentation are complete.
- If the learned policy loses to DMPs, report the negative result and centre the submission on the stronger DMP system.

## Code Quality

- Use typed, testable functions and structured configuration.
- Keep functions small and data transformations explicit.
- Seed Python, NumPy, PyTorch, and the simulator.
- Avoid broad exception handling and silent fallbacks.
- Use logging rather than scattered prints.
- Add concise comments only around non-obvious robotics mathematics or coordinate transforms.
- Keep generated data, checkpoints, and videos out of Git unless deliberately included as small submission assets.

## Final Deliverables

- Public-ready repository with no secrets or private data.
- One-command environment smoke test.
- One-command processing of sample data.
- One-command evaluation using committed configuration.
- Personally collected sample data and provenance notes.
- Architecture diagram.
- Demo video showing real input beside simulated robot behaviour.
- Machine-readable results and reproducible plots.
- Honest failure taxonomy and limitations.
- Model card and dataset card.
- Clear acknowledgements and citations.

## First Action

Begin by inspecting the repository and environment. Then create or update `TASKS.md`, `DECISIONS.md`, `EXPERIMENTS.md`, and `KNOWN_ISSUES.md`. Present a concise repository audit and Phase 0 plan, then implement Phase 0 immediately. Do not start by writing a polished README or by downloading a large model.

