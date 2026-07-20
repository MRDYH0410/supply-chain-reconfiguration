from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class EpisodeSummary:
    total_cost: float
    total_reward: float
    cost_breakdown_sum: Dict[str, float]
    activated: int
    withdrawn: int


def summarize_episode(infos: List[Dict[str, Any]]) -> EpisodeSummary:
    total_cost = 0.0
    total_reward = 0.0
    breakdown = {"C_in": 0.0, "C_out": 0.0, "C_fix": 0.0, "C_qual": 0.0, "C_loss": 0.0, "Salvage": 0.0}
    activated = 0
    withdrawn = 0

    for info in infos:
        cb = info.get("cost_breakdown", None)
        if cb is not None:
            total_cost += float(cb["C_total"])
            total_reward += float(cb["reward"])
            for k in breakdown:
                breakdown[k] += float(cb[k])
        activated += int(info.get("u_t", 0))
        withdrawn += int(info.get("v_t", 0))

    return EpisodeSummary(
        total_cost=total_cost,
        total_reward=total_reward,
        cost_breakdown_sum=breakdown,
        activated=activated,
        withdrawn=withdrawn,
    )
