# Related work and positioning

Phone2Panda combines established ideas rather than claiming a new foundational
robot-learning algorithm. Its contribution is the small, reproducible system
around those ideas and the controlled evidence showing where the personal
demonstrations matter.

| System | Human input | Robot transfer | Main distinction from Phone2Panda |
| --- | --- | --- | --- |
| DMP Simulation | Robot joint demonstrations recorded through simulator sliders | DMP replay and goal adaptation on a simulated Panda | Demonstrates Panda DMPs, but does not begin with phone video, compare route alternatives, audit contact safety, or distil a policy. |
| WHIRL | One unconstrained third-person RGB video | Human priors followed by real-world interaction and policy improvement | More general and physically validated; Phone2Panda is a lower-compute, explicit-geometry study with deterministic safety ablations. |
| VRB | Large-scale egocentric internet video | Learned contact and trajectory affordances support several robot-learning paradigms | Learns broad visual affordances from public-scale data; Phone2Panda uses 36 task-specific personal recordings and no pretrained affordance model. |
| Dobb-E | iPhone mounted on an instrumented reacher-grabber, recording RGB-D and motion | Behaviour cloning on a real Stretch robot | Uses purpose-built collection hardware and real-robot demonstrations; Phone2Panda uses an ordinary fixed overhead phone and marker-calibrated 2D object motion. |
| Human2Sim2Robot | One RGB-D human demonstration with object and hand reconstruction | Object-centric simulation reward, RL, then physical execution | Solves a harder 3D sim-to-real problem; Phone2Panda deliberately studies a simpler 2D route prior and compact policy on modest hardware. |
| OKAMI | One RGB-D video with object, hand, and body reconstruction | Object-aware retargeting to a physical humanoid and policy learning | Handles open-world objects and full humanoid motion; Phone2Panda instead isolates route geometry and contact-free Panda transport. |
| VideoManip | Monocular RGB human videos with 4D hand-object reconstruction | Contact-optimized trajectories train dexterous-hand policies | Recovers rich 3D contact and object geometry; Phone2Panda trades that generality for transparent 2D calibration and lightweight evaluation. |
| RoboWheel | Monocular RGB or RGB-D hand-object videos | Physics-refined data retargeted across arms, dexterous hands, and humanoids | A large cross-embodiment data engine; Phone2Panda is a single-task study that can be reproduced on modest hardware. |
| SPIDER | Human-video datasets reconstructed into hand-object trajectories | Physics-informed retargeting across several dexterous embodiments | Targets contact-rich dexterity at scale; Phone2Panda focuses on object-path safety for a parallel-jaw Panda. |

## What is specific to this project

- A complete path from personally recorded ordinary-phone video to closed-loop
  robot evaluation, without depth sensing, motion capture, or robot hardware.
- Counterfactual scoring of both left and right human-derived route families,
  rather than replaying only the route associated with a test scenario.
- Contact-pair auditing that separates intended grasp contact from object,
  robot, table, and self collisions.
- A measured correction from an uncalibrated 140 mm obstacle to the physical
  40 mm object, while preserving the failed experiment rather than replacing it.
- Distillation of safe DMP rollouts into a 3,559-parameter recurrent controller
  followed by closed-loop, fixed-seed evaluation and one-factor ablations.

## Sources

- [DMP Simulation](https://github.com/Cheems-JH/DMP_Simulation)
- [WHIRL: Human-to-Robot Imitation in the Wild](https://human2robot.github.io/)
- [VRB: Affordances from Human Videos](https://robo-affordances.github.io/)
- [Dobb-E](https://www.dobb-e.com/)
- [Human2Sim2Robot](https://proceedings.mlr.press/v305/lum25a.html)
- [OKAMI](https://ut-austin-rpl.github.io/OKAMI/)
- [VideoManip](https://videomanip.github.io/)
- [RoboWheel](https://github.com/zhangyuhong01/Robowheel-Toolkits)
- [SPIDER](https://github.com/facebookresearch/spider)

These projects operate at substantially greater scale and, in several cases,
on physical robots. The comparison is included to define scope, not to imply
state-of-the-art performance.
