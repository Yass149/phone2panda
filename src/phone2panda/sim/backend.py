from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class SimBackend(Protocol):
    def reset(self) -> dict[str, Any]: ...

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]: ...

    def close(self) -> None: ...
