from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from scenarios.scenario import SupplyChainScenario
from .operational_lp import OperationalSolution


@dataclass(frozen=True)
class CostBreakdown:
    t: int
    xi_t: int

    C_in: float
    C_out: float
    C_fix: float
    C_qual: float
    C_loss: float
    Salvage: float
    V_terminal: float

    C_total: float
    C_objective: float
    reward: float


def _tail_annuity_factor(gamma: float, tail_horizon: int) -> float:
    H_tail = max(0, int(tail_horizon))
    if H_tail <= 0:
        return 0.0
    return float(sum((float(gamma) ** m) for m in range(1, H_tail + 1)))


def is_full_relocation_ready(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a: Dict[str, int],
    age: int,
) -> bool:
    """Return True when Strategy C must withdraw the legacy plant."""
    M1 = scenario.legacy_plant_id
    M2 = scenario.candidate_plant_id

    if int(a.get(M1, 0)) != 1 or int(a.get(M2, 0)) != 1:
        return False

    cap_m1 = scenario.effective_capacity(
        plant_id=M1,
        t=t,
        xi_t=xi_t,
        a_jt=1,
        age=0,
    )
    cap_m2 = scenario.effective_capacity(
        plant_id=M2,
        t=t,
        xi_t=xi_t,
        a_jt=1,
        age=age,
    )
    return float(cap_m2) >= float(cap_m1) - 1.0e-9


def compute_terminal_value(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a_post: Dict[str, int],
    age_post: int,
    op: Optional[OperationalSolution] = None,
    tail_horizon: Optional[int] = None,
    tail_weight: Optional[float] = None,
) -> float:
    """Compute a conservative continuation-value credit at the window endpoint."""
    M2 = scenario.candidate_plant_id
    M1 = scenario.legacy_plant_id
    kc = scenario.core_market_id

    if int(a_post.get(M2, 0)) != 1:
        return 0.0

    cfg = scenario.reconfig
    H_tail = int(tail_horizon) if tail_horizon is not None else int(getattr(cfg, "terminal_tail_horizon", 0))
    w_tail = float(tail_weight) if tail_weight is not None else float(getattr(cfg, "terminal_tail_weight", 1.0))
    if H_tail <= 0 or w_tail <= 0.0:
        return 0.0

    annuity = w_tail * _tail_annuity_factor(float(scenario.gamma), H_tail)
    if annuity <= 0.0:
        return 0.0

    d_post, _ = scenario.realised_demand(t=t, xi_t=xi_t, a=a_post, age=age_post)
    cap_m2 = scenario.effective_capacity(
        plant_id=M2,
        t=t,
        xi_t=xi_t,
        a_jt=1,
        age=age_post,
    )
    q_ref = min(float(cap_m2), float(d_post.get(kc, 0.0)))
    if q_ref <= 1.0e-12:
        return 0.0

    legacy_unit = scenario.approximate_unit_service_cost(
        t=t,
        plant_id=M1,
        market_id=kc,
        xi_t=xi_t,
        age=scenario.H_rel + 1,
    )
    candidate_unit = scenario.approximate_unit_service_cost(
        t=t,
        plant_id=M2,
        market_id=kc,
        xi_t=xi_t,
        age=age_post,
    )
    delta_c_plus = max(0.0, float(legacy_unit) - float(candidate_unit))
    if delta_c_plus <= 1.0e-12:
        return 0.0

    return float(annuity * q_ref * delta_c_plus)


def compute_period_cost(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a: Dict[str, int],
    age: int,
    u_prev: int,
    v_t: int,
    op: OperationalSolution,
    a_post: Optional[Dict[str, int]] = None,
    age_post: Optional[int] = None,
    add_terminal_credit: bool = False,
) -> CostBreakdown:
    """Assemble per-period cost components and the terminal-value-adjusted objective."""
    C_in = float(op.C_in)
    C_out = float(op.C_out)
    C_loss = float(op.C_loss)

    # Fixed investment paid when candidate becomes operational.
    C_fix = float(scenario.reconfig.F) * (1 if int(u_prev) == 1 else 0)

    # Qualification cost now uses the scenario-defined front-loaded schedule.
    a_cand = int(a.get(scenario.candidate_plant_id, 0))
    C_qual = float(scenario.qualification_cost(a_candidate=a_cand, age=age))

    # Salvage credit under withdrawal.
    Salvage = float(scenario.reconfig.S_salv) * (1 if int(v_t) == 1 else 0)

    C_total = C_in + C_out + C_fix + C_qual + C_loss - Salvage

    if add_terminal_credit and a_post is not None and age_post is not None:
        V_terminal = compute_terminal_value(
            scenario=scenario,
            t=t,
            xi_t=xi_t,
            a_post=a_post,
            age_post=age_post,
            op=op,
        )
    else:
        V_terminal = 0.0

    C_objective = C_total - V_terminal
    reward = -C_objective

    return CostBreakdown(
        t=t,
        xi_t=xi_t,
        C_in=C_in,
        C_out=C_out,
        C_fix=C_fix,
        C_qual=C_qual,
        C_loss=C_loss,
        Salvage=Salvage,
        V_terminal=float(V_terminal),
        C_total=float(C_total),
        C_objective=float(C_objective),
        reward=float(reward),
    )
