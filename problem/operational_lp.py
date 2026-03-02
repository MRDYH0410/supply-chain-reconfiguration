from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from scenarios.scenario import SupplyChainScenario, at
from .demand import compute_realised_demand


@dataclass(frozen=True)
class OperationalSolution:
    t: int
    xi_t: int
    status: str
    message: str

    d: Dict[str, float]
    L: Dict[str, float]

    q: Dict[Tuple[str, str], float]  # (plant, market) -> shipment
    x: Dict[Tuple[str, str], float]  # (supplier, plant) -> procurement
    y: Dict[str, float]             # plant -> production

    C_out: float
    C_in: float
    C_loss: float


def solve_operational_lp(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a: Dict[str, int],
    age: int,
) -> OperationalSolution:
    """Solve the single period operational LP in Chapter 4."""
    try:
        from scipy.optimize import linprog
    except Exception as e:
        raise ImportError(
            "scipy is required for the operational LP solver. "
            "Install scipy or replace this solver with another LP package."
        ) from e

    plants: List[str] = sorted(list(scenario.plants.keys()))
    markets: List[str] = sorted(list(scenario.markets.keys()))
    suppliers: List[str] = sorted(list(scenario.suppliers.keys()))

    d, L = compute_realised_demand(scenario=scenario, t=t, xi_t=xi_t, a=a, age=age)

    Cap: Dict[str, float] = {}
    for j in plants:
        cap_j = scenario.effective_capacity(
            plant_id=j,
            t=t,
            xi_t=xi_t,
            a_jt=int(a.get(j, 0)),
            age=(age if j == scenario.candidate_plant_id else 0),
        )
        Cap[j] = float(cap_j)

    q_idx: Dict[Tuple[str, str], int] = {}
    x_idx: Dict[Tuple[str, str], int] = {}
    y_idx: Dict[str, int] = {}

    idx = 0
    for j in plants:
        for k in markets:
            q_idx[(j, k)] = idx
            idx += 1
    for i in suppliers:
        for j in plants:
            x_idx[(i, j)] = idx
            idx += 1
    for j in plants:
        y_idx[j] = idx
        idx += 1

    n_vars = idx
    c = [0.0] * n_vars

    for j in plants:
        for k in markets:
            c[q_idx[(j, k)]] = scenario.tariff_inclusive_delivered_cost(j, k, xi_t)

    for i in suppliers:
        for j in plants:
            mat = scenario.material_unit_cost(
                i, j, candidate_age=(age if j == scenario.candidate_plant_id else scenario.H_rel)
            )
            in_cost = scenario.inbound_unit_cost(i, j)
            c[x_idx[(i, j)]] = mat + in_cost

    A_eq: List[List[float]] = []
    b_eq: List[float] = []

    # demand satisfaction
    for k in markets:
        row = [0.0] * n_vars
        for j in plants:
            row[q_idx[(j, k)]] = 1.0
        A_eq.append(row)
        b_eq.append(float(d[k]))

    # shipment balance y_j = sum_k q_jk
    for j in plants:
        row = [0.0] * n_vars
        row[y_idx[j]] = 1.0
        for k in markets:
            row[q_idx[(j, k)]] = -1.0
        A_eq.append(row)
        b_eq.append(0.0)

    # material balance sum_i x_ij = alpha_j * y_j
    for j in plants:
        row = [0.0] * n_vars
        for i in suppliers:
            row[x_idx[(i, j)]] = 1.0
        alpha = float(scenario.alpha_by_plant[j])
        row[y_idx[j]] = -alpha
        A_eq.append(row)
        b_eq.append(0.0)

    A_ub: List[List[float]] = []
    b_ub: List[float] = []

    # supplier capacity
    for i in suppliers:
        row = [0.0] * n_vars
        for j in plants:
            row[x_idx[(i, j)]] = 1.0
        A_ub.append(row)
        b_ub.append(float(at(scenario.suppliers[i].W_bar, t)))

    # plant capacity
    for j in plants:
        row = [0.0] * n_vars
        row[y_idx[j]] = 1.0
        A_ub.append(row)
        b_ub.append(float(Cap[j]))

    bounds = [(0.0, None)] * n_vars
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    status = "optimal" if res.success else "infeasible_or_failed"
    message = res.message if isinstance(res.message, str) else str(res.message)

    q: Dict[Tuple[str, str], float] = {}
    x: Dict[Tuple[str, str], float] = {}
    y: Dict[str, float] = {}

    if res.success and res.x is not None:
        sol = res.x.tolist()
        for j in plants:
            for k in markets:
                q[(j, k)] = float(sol[q_idx[(j, k)]])
        for i in suppliers:
            for j in plants:
                x[(i, j)] = float(sol[x_idx[(i, j)]])
        for j in plants:
            y[j] = float(sol[y_idx[j]])
    else:
        for j in plants:
            for k in markets:
                q[(j, k)] = 0.0
        for i in suppliers:
            for j in plants:
                x[(i, j)] = 0.0
        for j in plants:
            y[j] = 0.0

    C_out = 0.0
    for j in plants:
        for k in markets:
            C_out += scenario.tariff_inclusive_delivered_cost(j, k, xi_t) * q[(j, k)]

    C_in = 0.0
    for i in suppliers:
        for j in plants:
            mat = scenario.material_unit_cost(
                i, j, candidate_age=(age if j == scenario.candidate_plant_id else scenario.H_rel)
            )
            in_cost = scenario.inbound_unit_cost(i, j)
            C_in += (mat + in_cost) * x[(i, j)]

    C_loss = 0.0
    for k in markets:
        C_loss += float(scenario.markets[k].cu) * float(L[k])

    return OperationalSolution(
        t=t,
        xi_t=xi_t,
        status=status,
        message=message,
        d=d,
        L=L,
        q=q,
        x=x,
        y=y,
        C_out=float(C_out),
        C_in=float(C_in),
        C_loss=float(C_loss),
    )