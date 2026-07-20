from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from scenarios.scenario import SupplyChainScenario
from problem.costs import is_full_relocation_ready


@dataclass(frozen=True)
class ActionMask:
    u_mask: Tuple[int, int]
    v_mask: Tuple[int, int]


def compute_action_mask(
    scenario: SupplyChainScenario,
    strategy: str,
    t: int,
    xi_t: int,
    a: Dict[str, int],
    age: int,
) -> ActionMask:
    """Compute admissible action masks for (u_t, v_t).

    Activation is not allowed in the final period because it only becomes
    effective at t+1.

    For Strategy C, withdrawal becomes mandatory once the candidate plant has
    ramped to the legacy plant's effective capacity. This matches the paper
    rule requested by the user.
    """
    M1 = scenario.legacy_plant_id
    M2 = scenario.candidate_plant_id

    a_m2 = int(a.get(M2, 0))
    a_m1 = int(a.get(M1, 0))

    if a_m2 == 0 and t < int(scenario.H):
        u_mask = (1, 1)
    else:
        u_mask = (1, 0)

    v_mask = (1, 0)

    if strategy.upper() == "C":
        if is_full_relocation_ready(
            scenario=scenario,
            t=t,
            xi_t=xi_t,
            a=a,
            age=age,
        ):
            v_mask = (0, 1)
        else:
            v_mask = (1, 0)

        if a_m1 == 0:
            v_mask = (1, 0)

    return ActionMask(u_mask=u_mask, v_mask=v_mask)
