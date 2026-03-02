from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from rl.env.masks import ActionMask


@dataclass
class NoReconfigPolicy:
    """Always choose u=0, v=0."""

    def act(self, obs: np.ndarray, mask: ActionMask) -> Tuple[Tuple[int, int], float, float]:
        return (0, 0), 0.0, 0.0


@dataclass
class ActivateAtFirstChancePolicy:
    """Activate once at the first period where activation is allowed; never withdraw."""
    activated: bool = False

    def act(self, obs: np.ndarray, mask: ActionMask) -> Tuple[Tuple[int, int], float, float]:
        u = 0
        if (not self.activated) and mask.u_mask[1] == 1:
            u = 1
            self.activated = True
        v = 0
        return (u, v), 0.0, 0.0
