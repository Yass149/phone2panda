# Known issues

- The supplied pilots use custom L markers rather than printed ArUco markers. The implementation supports the supplied layout only; future recordings should either preserve this exact layout or migrate together to a versioned ArUco configuration.
- The camera must remain fixed. Static-reference calibration fills frames where a corner is occluded, but those frames fail the direct-visibility coverage metric.
- Normalised clearance uses a configured conservative circular footprint around the tracked red marker. Exact box orientation and height are not observable from the single overhead RGB view.
- Generated overlays intentionally omit the source AAC audio and timed metadata.
- robosuite 1.5.1 is incompatible with MuJoCo 3.14.0 in this environment; `mujoco==3.2.7` is pinned.
- Phase 4A measures one fixed-seed rollout only. Generalisation, baseline success rates, and confidence intervals remain unmeasured.
- The agent-view camera partially occludes the obstacle behind the Panda during parts of the rollout; state-based collision and clearance metrics remain available.
