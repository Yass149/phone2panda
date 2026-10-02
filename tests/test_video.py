from __future__ import annotations

from phone2panda.pilot_validation.video import estimate_dropped_frames


def test_dropped_frame_estimation() -> None:
    dropped, gaps = estimate_dropped_frames([0.0, 1 / 30, 3 / 30, 4 / 30], 30.0)
    assert dropped == 1
    assert gaps == [1]
