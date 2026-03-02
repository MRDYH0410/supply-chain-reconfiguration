from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    strategy: str = "B"  # "B" or "C"
    seed: int = 0
    iterations: int = 50
    episodes_per_iter: int = 20
    hidden: int = 64
    device: str = "cpu"
