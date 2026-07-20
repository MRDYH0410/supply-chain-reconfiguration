from __future__ import annotations

from typing import Dict, Tuple

from scenarios.scenario import SupplyChainScenario


def compute_potential_demand(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a_candidate: int,
) -> Dict[str, float]:
    """Return potential demand bar d_{k,t} for all markets."""
    return scenario.potential_demand(t=t, xi_t=xi_t, a_candidate=a_candidate)


def compute_realised_demand(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a: Dict[str, int],
    age: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return realised demand d_{k,t} and demand loss L_{k,t}."""
    return scenario.realised_demand(t=t, xi_t=xi_t, a=a, age=age)