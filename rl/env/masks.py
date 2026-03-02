from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from scenarios.scenario import SupplyChainScenario


@dataclass(frozen=True)
class ActionMask:
    # each mask is (allow0, allow1) for the binary action
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

    Two-phase structure
    - Phase I: candidate inactive. u can be 0 or 1, v must be 0.
    - Phase II: candidate active. u must be 0.
      Strategy B: v must be 0.
      Strategy C: v can be 0 or 1 only if withdrawal is feasible and legacy still active.

    Withdrawal feasibility follows Chapter 3.5:
      v_t is allowed only if Cap_M2,t >= sum_k d_{k,t}.
    """
    M1 = scenario.legacy_plant_id
    M2 = scenario.candidate_plant_id

    a_m2 = int(a.get(M2, 0))
    a_m1 = int(a.get(M1, 0))

    # u mask
    if a_m2 == 0:
        u_mask = (1, 1)  # allow 0 and 1
    else:
        u_mask = (1, 0)  # only 0

    # v mask default
    v_mask = (1, 0)  # only 0 by default

    if strategy.upper() == "C":
        if a_m2 == 1 and a_m1 == 1:
            # compute feasibility: Cap_M2,t >= sum_k d_{k,t}
            d, _ = scenario.realised_demand(t=t, xi_t=xi_t, a=a, age=age)
            cap_m2 = scenario.effective_capacity(
                plant_id=M2,
                t=t,
                xi_t=xi_t,
                a_jt=1,
                age=age,
            )
            feasible = float(cap_m2) >= sum(float(v) for v in d.values())
            if feasible:
                v_mask = (1, 1)  # allow withdrawal
            else:
                v_mask = (1, 0)

    # Strategy A and B: v always 0
    return ActionMask(u_mask=u_mask, v_mask=v_mask)
