# Third-party notices

Phone2Panda is MIT-licensed. Its pinned dependencies are not relicensed by this
repository; each remains under its upstream licence. The runtime includes
PyAV/FFmpeg, OpenCV, NumPy, SciPy, Matplotlib, MuJoCo, robosuite, ImageIO and
PyYAML. Development uses pytest, Ruff, uv and Hatchling.

The repository does not vendor those projects. See `uv.lock` for the exact
resolved packages and versions, and review upstream licences before
redistributing a bundled environment or binary. MuJoCo model assets and
robosuite assets retain their upstream notices.

The private source recordings belong to the project author and are not covered
by the software licence or distributed in Git. The selected, silent derived
demonstration media under `media/` is published with this repository.
