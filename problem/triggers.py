from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

from scenarios.scenario import SupplyChainScenario
from problem.operational_lp import solve_operational_lp
from problem.costs import compute_period_cost


@dataclass(frozen=True)
class TriggerConfig:
    H_look: int = 6              # look-ahead window length
    n_paths: int = 50            # Monte Carlo regime paths
    seed: int = 0
    gamma: float = 1.0           # discount factor


@dataclass(frozen=True)
class TriggerResult:
    J_NR: float
    J_ALT: float
    psi: float
    best_theta: Optional[int] = None   # only meaningful for Strategy C


def _sample_future_regimes(
    scenario: SupplyChainScenario,
    xi_t: int,
    t: int,
    L: int,
    rng: random.Random,
) -> List[int]:
    """
    Return a list of regimes of length L for periods t..t+L-1, starting at xi_t.
    Keep consistency with xi_2_forced when t==1 and L>=2.
    """
    regimes = [xi_t]

    if L == 1:
        return regimes

    # enforce early escalation if needed and this is period 1
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


def _withdrawal_feasible(
    scenario: SupplyChainScenario,
    t: int,
    xi_t: int,
    a: Dict[str, int],
    age: int,
) -> bool:
    """
    Same feasibility rule used in rl/env/masks.py for Strategy C
    Allow withdrawal only if Cap_M2,t >= sum_k d_{k,t}
    d is realised demand under the current structure a
    """
    M2 = scenario.candidate_plant_id
    d, _ = scenario.realised_demand(t=t, xi_t=xi_t, a=a, age=age)
    cap_m2 = scenario.effective_capacity(
        plant_id=M2, t=t, xi_t=xi_t, a_jt=1, age=age
    )
    return float(cap_m2) >= sum(float(v) for v in d.values())


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
    """
    Simulate discounted total cost over the given regimes list.
    regimes is for periods t0..t0+L-1 inclusive.

    activate_now means u_{t0}=1 (effective at t0+1).
    withdraw_at means v_{withdraw_at}=1 (effective at withdraw_at+1).
    """
    M1 = scenario.legacy_plant_id
    M2 = scenario.candidate_plant_id

    a = dict(a0)
    age = int(age0)
    u_prev = int(u_prev0)

    total = 0.0

    for step, xi in enumerate(regimes):
        t = t0 + step

        u_t = 1 if (step == 0 and activate_now) else 0
        v_t = 1 if (withdraw_at is not None and t == withdraw_at) else 0

        # operational LP under current structure
        op = solve_operational_lp(scenario=scenario, t=t, xi_t=xi, a=a, age=age)

        # total cost assembly
        cb = compute_period_cost(
            scenario=scenario,
            t=t,
            xi_t=xi,
            a=a,
            age=age,
            u_prev=u_prev,
            v_t=v_t,
            op=op,
        )

        total += (scenario.gamma ** step) * float(cb.C_total)

        # transition to next period structural state
        if step == len(regimes) - 1:
            break

        a_next = dict(a)

        # withdrawal affects legacy availability next period
        if v_t == 1:
            a_next[M1] = 0

        # activation affects candidate availability next period
        if int(a.get(M2, 0)) == 0 and u_t == 1:
            a_next[M2] = 1
            age_next = 1
        else:
            if int(a_next.get(M2, 0)) == 1:
                if int(a.get(M2, 0)) == 1:
                    age_next = age + 1
                else:
                    age_next = 1
            else:
                age_next = 0

        u_prev = int(u_t)
        a = a_next
        age = int(age_next)

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
    """
    Psi_PR(s_t) = J_NR - J_PR
    J_PR assumes activation decision is taken now and candidate becomes available at t+1
    No withdrawal is considered in Strategy B
    """
    L = min(cfg.H_look, scenario.H - t + 1)
    rng = random.Random(cfg.seed + 100000 * t + 17)

    J_NR_sum = 0.0
    J_PR_sum = 0.0

    for p in range(cfg.n_paths):
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
    """
    Psi_FR(s_t) = J_NR - J_FR

    J_FR assumes activation now, and then chooses the best withdrawal time theta
    within the look-ahead window by minimising the total discounted cost.
    """
    L = min(cfg.H_look, scenario.H - t + 1)
    rng = random.Random(cfg.seed + 200000 * t + 29)

    J_NR_sum = 0.0
    J_FR_sum = 0.0
    theta_best_count: Dict[int, int] = {}

    for p in range(cfg.n_paths):
        regimes = _sample_future_regimes(scenario, xi_t=xi_t, t=t, L=L, rng=rng)

        # stay in legacy only
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

        # activate now and then choose the best theta
        # build a candidate set of theta within the window
        # theta must be >= t+1 since candidate is effective from t+1
        best_val = None
        best_theta = None

        # option 1 no withdrawal within window
        val_no_withdraw = _simulate_cost_over_window(
            scenario=scenario,
            t0=t,
            a0=a,
            age0=age,
            u_prev0=u_prev,
            regimes=regimes,
            activate_now=True,
            withdraw_at=None,
        )
        best_val = val_no_withdraw
        best_theta = None

        # options with withdrawal at theta
        # we test feasibility using the same rule as mask, at the theta period before withdrawal is applied
        for step in range(1, L):  # step=1 corresponds to period t+1
            theta = t + step

            # reconstruct structural state at theta under activation only
            # simulate forward up to theta to get a_theta and age_theta
            # simplest is to simulate step by step with zero withdrawal, then test feasibility
            a_tmp = dict(a)
            age_tmp = int(age)
            u_prev_tmp = int(u_prev)

            for s in range(step):
                tt = t + s
                u_s = 1 if (s == 0) else 0
                # update to next
                a_next = dict(a_tmp)
                if int(a_tmp.get(scenario.candidate_plant_id, 0)) == 0 and u_s == 1:
                    a_next[scenario.candidate_plant_id] = 1
                    age_next = 1
                else:
                    if int(a_next.get(scenario.candidate_plant_id, 0)) == 1:
                        if int(a_tmp.get(scenario.candidate_plant_id, 0)) == 1:
                            age_next = age_tmp + 1
                        else:
                            age_next = 1
                    else:
                        age_next = 0
                a_tmp = a_next
                age_tmp = int(age_next)
                u_prev_tmp = int(u_s)

            # now at theta we have a_tmp and age_tmp
            # withdrawal feasible only if legacy still active and candidate active
            if int(a_tmp.get(scenario.legacy_plant_id, 0)) != 1:
                continue
            if int(a_tmp.get(scenario.candidate_plant_id, 0)) != 1:
                continue

            xi_theta = regimes[step]
            if not _withdrawal_feasible(scenario, t=theta, xi_t=xi_theta, a=a_tmp, age=age_tmp):
                continue

            val_theta = _simulate_cost_over_window(
                scenario=scenario,
                t0=t,
                a0=a,
                age0=age,
                u_prev0=u_prev,
                regimes=regimes,
                activate_now=True,
                withdraw_at=theta,
            )

            if best_val is None or val_theta < best_val:
                best_val = val_theta
                best_theta = theta

        J_FR_sum += float(best_val)
        if best_theta is not None:
            theta_best_count[best_theta] = theta_best_count.get(best_theta, 0) + 1

    J_NR = J_NR_sum / cfg.n_paths
    J_FR = J_FR_sum / cfg.n_paths
    psi = J_NR - J_FR

    # report the most frequently selected theta across paths, for debugging
    theta_mode = None
    if theta_best_count:
        theta_mode = max(theta_best_count.items(), key=lambda x: x[1])[0]

    return TriggerResult(J_NR=J_NR, J_ALT=J_FR, psi=psi, best_theta=theta_mode)