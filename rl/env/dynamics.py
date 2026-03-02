from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from scenarios.scenario import SupplyChainScenario


@dataclass(frozen=True)
class StructuralState:
    t: int
    xi_t: int
    a: Dict[str, int]     # plant availability
    age: int              # candidate plant age (0 if inactive)
    u_prev: int           # u_{t-1}, needed for fixed cost timing


def next_structural_state(
    scenario: SupplyChainScenario,
    state: StructuralState,
    u_t: int,
    v_t: int,
    xi_next: int,
) -> StructuralState:
    """Apply Chapter 3 structural dynamics with lag=1.

    - activation: if a_M2,t == 0 and u_t == 1, then a_M2,t+1 = 1 and age_{t+1} = 1
    - age: if a_M2,t == 1 then age_{t+1} = age_t + 1
           else age_{t+1} = 0 unless activation happens
    - withdrawal: if v_t == 1 then a_M1,t+1 = 0
    """
    M1 = scenario.legacy_plant_id
    M2 = scenario.candidate_plant_id

    a_next = dict(state.a)
    age_next = state.age

    # withdrawal affects next period availability of legacy
    if int(v_t) == 1:
        a_next[M1] = 0

    # activation affects next period availability of candidate
    if int(state.a.get(M2, 0)) == 0 and int(u_t) == 1:
        a_next[M2] = 1
        age_next = 1
    else:
        if int(a_next.get(M2, 0)) == 1:
            # candidate is active next period, age increments if it was active already
            if int(state.a.get(M2, 0)) == 1:
                age_next = state.age + 1
            # else activation case already handled
        else:
            age_next = 0

    t_next = state.t + 1
    return StructuralState(
        t=t_next,
        xi_t=xi_next,
        a=a_next,
        age=age_next,
        u_prev=int(u_t),
    )
