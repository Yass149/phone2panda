# Known issues

- The supplied pilots use custom L markers rather than printed ArUco markers. The implementation supports the supplied layout only; future recordings should either preserve this exact layout or migrate together to a versioned ArUco configuration.
- The camera must remain fixed. Static-reference calibration fills frames where a corner is occluded, but those frames fail the direct-visibility coverage metric.
- Normalised clearance uses a configured conservative circular footprint around the tracked red marker. Exact box orientation and height are not observable from the single overhead RGB view.
- Generated overlays intentionally omit the source AAC audio and timed metadata.
- robosuite 1.5.1 is incompatible with MuJoCo 3.14.0 in this environment; `mujoco==3.2.7` is pinned.
- The agent-view camera partially occludes the obstacle behind the Panda during parts of the rollout; state-based collision and clearance metrics remain available.
- Phase 4B retargets two-dimensional object paths and does not solve whole-arm motion planning. Exact Phase 4C attribution found 687 link-7/obstacle events plus one gripper-hand/obstacle event across 39/50 winning-method runs. A conservative height and goal-side posture still failed the five-seed safety gate, with links 5–7 contacting the obstacle; whole-arm motion planning or a validated collision-aware IK posture is required before policy distillation.
- Phase 4D confirmed that route-side lateral staging keeps transport collision-free but does not resolve the final manipulation posture: all tested offsets contacted the obstacle during lowering, placement, or release, and larger offsets reduced placement success.
- Phases 4B–4D used an uncalibrated 0.14 m obstacle and should be interpreted as a tall-obstacle stress test, not a faithful model of the approximately 0.04 m glasses case. Phase 4E calibrates the obstacle to 0.04 m; the route/confidence DMP was safe in 50/50 fixed-seed rollouts, while raw replay and unfiltered DMP each had one gripper-finger/obstacle contact at seed 4140.
