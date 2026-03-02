from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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

    C_total: float
    reward: float


def compute_period_cost(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a: Dict[str, int],
    age: int,
    u_prev: int,
    v_t: int,
    op: OperationalSolution,
) -> CostBreakdown:
    """Assemble per period cost components following Chapter 3.5."""
    C_in = float(op.C_in)
    C_out = float(op.C_out)
    C_loss = float(op.C_loss)

    # fixed investment paid when candidate becomes operational
    C_fix = float(scenario.reconfig.F) * (1 if int(u_prev) == 1 else 0)

    # qualification and deposit cost during early post-activation window
    a_cand = int(a.get(scenario.candidate_plant_id, 0))
    if a_cand == 1 and age > 0 and age <= int(scenario.reconfig.qual_ell):
        C_qual = float(scenario.reconfig.qual_G)
    else:
        C_qual = 0.0

    # salvage credit under withdrawal
    Salvage = float(scenario.reconfig.S_salv) * (1 if int(v_t) == 1 else 0)

    C_total = C_in + C_out + C_fix + C_qual + C_loss - Salvage
    reward = -C_total

    return CostBreakdown(
        t=t,
        xi_t=xi_t,
        C_in=C_in,
        C_out=C_out,
        C_fix=C_fix,
        C_qual=C_qual,
        C_loss=C_loss,
        Salvage=Salvage,
        C_total=float(C_total),
        reward=float(reward),
    )