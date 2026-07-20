from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

from scenarios.scenario import SupplyChainScenario
from problem.operational_lp import solve_operational_lp
from problem.costs import compute_period_cost, is_full_relocation_ready


@dataclass(frozen=True)
class TriggerConfig:
    H_look: int = 6
    n_paths: int = 50
    seed: int = 0
    gamma: float = 1.0


@dataclass(frozen=True)
class TriggerResult:
    J_NR: float
    J_ALT: float
    psi: float
    best_theta: Optional[int] = None


def _sample_future_regimes(
    scenario: SupplyChainScenario,
    xi_t: int,
    t: int,
    L: int,
    rng: random.Random,
) -> List[int]:
    """Return a regime list for periods t..t+L-1 inclusive."""
    regimes = [xi_t]

    if L == 1:
        return regimes

    if t == 1 and scenario.xi_2_forced is not None:
        regimes.append(int(scenario.xi_2_forced))
    else:
        regimes.append(_sample_next_regime(scenario, xi_t, rng))

    while len(regimes) < L:
        regimes.append(_sample_next_regime(scenario, regimes[-1], rng))

    return regimes


def _sample_next_regime(scenario: SupplyChainScenario, xi: int, rng: random.Random) -> int:
    idx = scenario.regimes.index(xi)
    probs = scenario.P[idx]
    u = rng.random()
    s = 0.0
    for j, p in enumerate(probs):
        s += p
        if u <= s:
            return int(scenario.regimes[j])
    return int(scenario.regimes[-1])


def _post_structure_after_action(
    scenario: SupplyChainScenario,
    a: Dict[str, int],
    age: int,
    u_t: int,
    v_t: int,
) -> Tuple[Dict[str, int], int]:
    M1 = scenario.legacy_plant_id
    M2 = scenario.candidate_plant_id

    a_next = dict(a)
    age_next = int(age)

    if int(v_t) == 1:
        a_next[M1] = 0

    if int(a.get(M2, 0)) == 0 and int(u_t) == 1:
        a_next[M2] = 1
        age_next = 1
    else:
        if int(a_next.get(M2, 0)) == 1:
            if int(a.get(M2, 0)) == 1:
                age_next = int(age) + 1
            else:
                age_next = 1
        else:
            age_next = 0

    return a_next, int(age_next)


def _earliest_full_relocation_theta(
    scenario: SupplyChainScenario,
    t0: int,
    xi_t0: int,
    a0: Dict[str, int],
    age0: int,
    regimes: List[int],
    activate_now: bool,
) -> Optional[int]:
    """Find the earliest period when Strategy C must withdraw M1.

    Paper rule implemented here:
    C must build very early, and once the candidate plant has ramped to the
    legacy plant's effective capacity, the legacy plant must be withdrawn
    immediately.
    """
    a = dict(a0)
    age = int(age0)

    for step, xi in enumerate(regimes):
        t = t0 + step

        if step >= 1 and is_full_relocation_ready(
            scenario=scenario,
            t=t,
            xi_t=xi,
            a=a,
            age=age,
        ):
            return int(t)

        u_t = 1 if (step == 0 and activate_now) else 0
        a, age = _post_structure_after_action(
            scenario=scenario,
            a=a,
            age=age,
            u_t=u_t,
            v_t=0,
        )

    return None


def _simulate_cost_over_window(
    scenario: SupplyChainScenario,
    t0: int,
    a0: Dict[str, int],
    age0: int,
    u_prev0: int,
    regimes: List[int],
    activate_now: bool,
    withdraw_at: Optional[int],
) -> float:
    """Simulate discounted objective cost over the window.

    The last period receives a terminal-value credit so that activation
    decisions are not over-penalised by the truncated look-ahead horizon.
    """
    a = dict(a0)
    age = int(age0)
    u_prev = int(u_prev0)

    total = 0.0

    for step, xi in enumerate(regimes):
        t = t0 + step

        u_t = 1 if (step == 0 and activate_now) else 0
        v_t = 1 if (withdraw_at is not None and t == withdraw_at) else 0

        op = solve_operational_lp(scenario=scenario, t=t, xi_t=xi, a=a, age=age)
        a_post, age_post = _post_structure_after_action(
            scenario=scenario,
            a=a,
            age=age,
            u_t=u_t,
            v_t=v_t,
        )

        cb = compute_period_cost(
            scenario=scenario,
            t=t,
            xi_t=xi,
            a=a,
            age=age,
            u_prev=u_prev,
            v_t=v_t,
            op=op,
            a_post=a_post,
            age_post=age_post,
            add_terminal_credit=(step == len(regimes) - 1),
        )

        total += (scenario.gamma ** step) * float(cb.C_objective)

        if step == len(regimes) - 1:
            break

        u_prev = int(u_t)
        a = a_post
        age = int(age_post)

    return float(total)


def trigger_score_strategy_b(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a: Dict[str, int],
    age: int,
    u_prev: int,
    cfg: TriggerConfig,
) -> TriggerResult:
    """Psi_PR(s_t) = J_NR - J_PR with terminal value at the window end."""
    L = min(cfg.H_look, scenario.H - t + 1)
    rng = random.Random(cfg.seed + 100000 * t + 17)

    J_NR_sum = 0.0
    J_PR_sum = 0.0

    for _ in range(cfg.n_paths):
        regimes = _sample_future_regimes(scenario, xi_t=xi_t, t=t, L=L, rng=rng)

        J_NR_sum += _simulate_cost_over_window(
            scenario=scenario,
            t0=t,
            a0=a,
            age0=age,
            u_prev0=u_prev,
            regimes=regimes,
            activate_now=False,
            withdraw_at=None,
        )

        J_PR_sum += _simulate_cost_over_window(
            scenario=scenario,
            t0=t,
            a0=a,
            age0=age,
            u_prev0=u_prev,
            regimes=regimes,
            activate_now=True,
            withdraw_at=None,
        )

    J_NR = J_NR_sum / cfg.n_paths
    J_PR = J_PR_sum / cfg.n_paths
    psi = J_NR - J_PR

    return TriggerResult(J_NR=J_NR, J_ALT=J_PR, psi=psi, best_theta=None)


def trigger_score_strategy_c(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a: Dict[str, int],
    age: int,
    u_prev: int,
    cfg: TriggerConfig,
) -> TriggerResult:
    """Psi_FR(s_t) = J_NR - J_FR with mandatory earliest full-relocation withdrawal.

    Strategy C no longer optimises over a late withdrawal date.
    After activation, withdrawal is fixed at the earliest period when the
    candidate plant reaches the legacy plant's effective capacity.
    """
    L = min(cfg.H_look, scenario.H - t + 1)
    rng = random.Random(cfg.seed + 200000 * t + 29)

    J_NR_sum = 0.0
    J_FR_sum = 0.0
    theta_best_count: Dict[int, int] = {}

    for _ in range(cfg.n_paths):
        regimes = _sample_future_regimes(scenario, xi_t=xi_t, t=t, L=L, rng=rng)

        J_NR_path = _simulate_cost_over_window(
            scenario=scenario,
            t0=t,
            a0=a,
            age0=age,
            u_prev0=u_prev,
            regimes=regimes,
            activate_now=False,
            withdraw_at=None,
        )
        J_NR_sum += J_NR_path

        theta = _earliest_full_relocation_theta(
            scenario=scenario,
            t0=t,
            xi_t0=xi_t,
            a0=a,
            age0=age,
            regimes=regimes,
            activate_now=True,
        )

        J_FR_path = _simulate_cost_over_window(
            scenario=scenario,
            t0=t,
            a0=a,
            age0=age,
            u_prev0=u_prev,
            regimes=regimes,
            activate_now=True,
            withdraw_at=theta,
        )
        J_FR_sum += J_FR_path

        if theta is not None:
            theta_best_count[theta] = theta_best_count.get(theta, 0) + 1

    J_NR = J_NR_sum / cfg.n_paths
    J_FR = J_FR_sum / cfg.n_paths
    psi = J_NR - J_FR

    theta_mode = None
    if theta_best_count:
        theta_mode = max(theta_best_count.items(), key=lambda x: x[1])[0]

    return TriggerResult(J_NR=J_NR, J_ALT=J_FR, psi=psi, best_theta=theta_mode)
