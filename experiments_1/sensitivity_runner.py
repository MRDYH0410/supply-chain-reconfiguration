from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import copy
import random

from scenarios.story_case import build_story_case
from scenarios.scenario import SupplyChainScenario

from rl.env.sc_reconfig_env import SCReconfigEnv, Strategy
from problem.costs import is_full_relocation_ready


@dataclass
class TrainEvalConfig:
    hidden: int = 32
    device: str = "cpu"
    iterations: int = 10
    episodes_per_iter: int = 10
    eval_episode_seed: int = 20260325


@dataclass
class StrategyStats:
    mean_discounted_total_cost: float
    std_discounted_total_cost: float
    mean_total_cost: float
    std_total_cost: float
    mean_breakdown: Dict[str, float]
    std_breakdown: Dict[str, float]
    mean_cost_by_t: List[float]
    std_cost_by_t: List[float]
    mean_discounted_cost_by_t: List[float]
    std_discounted_cost_by_t: List[float]
    mean_cumulative_discounted_cost_by_t: List[float]
    std_cumulative_discounted_cost_by_t: List[float]
    mean_demand_by_t: List[float]
    std_demand_by_t: List[float]
    mean_cumulative_demand_by_t: List[float]
    std_cumulative_demand_by_t: List[float]
    mean_total_demand: float
    std_total_demand: float
    mean_u_by_t: List[float]
    mean_v_by_t: List[float]
    mean_a_M1_by_t: List[float]
    mean_a_M2_by_t: List[float]
    mean_age_M2_by_t: List[float]
    mean_rho_M2_by_t: List[float]
    activation_periods: List[int]
    withdrawal_periods: List[int]
    mean_terminal_value: float


def _first_action_period(trace: List[Dict[str, Any]], key: str) -> int:
    for info in trace:
        if int(info.get(key, 0)) == 1:
            return int(info.get("t", 0))
    return 0


def _summarize_trace(trace: List[Dict[str, Any]], gamma: float) -> Dict[str, Any]:
    H = len(trace)
    cost_by_t: List[float] = []
    discounted_cost_by_t: List[float] = []
    cumulative_discounted_cost_by_t: List[float] = []
    demand_by_t: List[float] = []
    cumulative_demand_by_t: List[float] = []
    u_by_t: List[float] = []
    v_by_t: List[float] = []
    a_m1_by_t: List[float] = []
    a_m2_by_t: List[float] = []
    age_m2_by_t: List[float] = []
    rho_m2_by_t: List[float] = []

    breakdown = {
        "C_in": 0.0,
        "C_out": 0.0,
        "C_fix": 0.0,
        "C_qual": 0.0,
        "C_loss": 0.0,
        "Salvage": 0.0,
        "V_terminal": 0.0,
    }

    discounted_total = 0.0
    total = 0.0
    running = 0.0
    terminal_value = 0.0
    running_demand = 0.0

    for idx, info in enumerate(trace):
        cb = info.get("cost_breakdown", {}) or {}
        c_obj = float(cb.get("C_objective", cb.get("C_total", 0.0)))
        dc = (gamma ** idx) * c_obj
        running += dc
        total += c_obj
        discounted_total += dc

        demand_t = float(sum((info.get("d", {}) or {}).values()))
        running_demand += demand_t

        cost_by_t.append(c_obj)
        discounted_cost_by_t.append(dc)
        cumulative_discounted_cost_by_t.append(running)
        demand_by_t.append(demand_t)
        cumulative_demand_by_t.append(running_demand)
        u_by_t.append(float(info.get("u_t", 0.0)))
        v_by_t.append(float(info.get("v_t", 0.0)))
        a_m1_by_t.append(float(info.get("a_M1_pre", info.get("a_M1", 0.0))))
        a_m2_by_t.append(float(info.get("a_M2_pre", info.get("a_M2", 0.0))))
        age_m2_by_t.append(float(info.get("age_M2_pre", info.get("age_M2", 0.0))))
        rho_m2_by_t.append(float(info.get("rho_M2_pre", info.get("rho_M2", 0.0))))

        for k in breakdown:
            breakdown[k] += float(cb.get(k, 0.0))

        terminal_value += float(cb.get("V_terminal", 0.0))

    if H == 0:
        return {
            "total_cost": 0.0,
            "discounted_total_cost": 0.0,
            "terminal_value": 0.0,
            "breakdown": breakdown,
            "cost_by_t": [],
            "discounted_cost_by_t": [],
            "cumulative_discounted_cost_by_t": [],
            "demand_by_t": [],
            "cumulative_demand_by_t": [],
            "total_demand": 0.0,
            "u_by_t": [],
            "v_by_t": [],
            "a_M1_by_t": [],
            "a_M2_by_t": [],
            "age_M2_by_t": [],
            "rho_M2_by_t": [],
            "activation_period": 0,
            "withdrawal_period": 0,
        }

    return {
        "total_cost": float(total),
        "discounted_total_cost": float(discounted_total),
        "terminal_value": float(terminal_value),
        "breakdown": breakdown,
        "cost_by_t": cost_by_t,
        "discounted_cost_by_t": discounted_cost_by_t,
        "cumulative_discounted_cost_by_t": cumulative_discounted_cost_by_t,
        "demand_by_t": demand_by_t,
        "cumulative_demand_by_t": cumulative_demand_by_t,
        "total_demand": float(running_demand),
        "u_by_t": u_by_t,
        "v_by_t": v_by_t,
        "a_M1_by_t": a_m1_by_t,
        "a_M2_by_t": a_m2_by_t,
        "age_M2_by_t": age_m2_by_t,
        "rho_M2_by_t": rho_m2_by_t,
        "activation_period": _first_action_period(trace, "u_t"),
        "withdrawal_period": _first_action_period(trace, "v_t"),
    }


def _pack_single_summary_as_stats(summary: Dict[str, Any]) -> StrategyStats:
    breakdown = dict(summary["breakdown"])
    H = len(summary["cost_by_t"])
    zero_breakdown = {k: 0.0 for k in breakdown.keys()}
    return StrategyStats(
        mean_discounted_total_cost=float(summary["discounted_total_cost"]),
        std_discounted_total_cost=0.0,
        mean_total_cost=float(summary["total_cost"]),
        std_total_cost=0.0,
        mean_breakdown=breakdown,
        std_breakdown=zero_breakdown,
        mean_cost_by_t=[float(v) for v in summary["cost_by_t"]],
        std_cost_by_t=[0.0] * H,
        mean_discounted_cost_by_t=[float(v) for v in summary["discounted_cost_by_t"]],
        std_discounted_cost_by_t=[0.0] * H,
        mean_cumulative_discounted_cost_by_t=[float(v) for v in summary["cumulative_discounted_cost_by_t"]],
        std_cumulative_discounted_cost_by_t=[0.0] * H,
        mean_demand_by_t=[float(v) for v in summary.get("demand_by_t", [0.0] * H)],
        std_demand_by_t=[0.0] * H,
        mean_cumulative_demand_by_t=[float(v) for v in summary.get("cumulative_demand_by_t", [0.0] * H)],
        std_cumulative_demand_by_t=[0.0] * H,
        mean_total_demand=float(summary.get("total_demand", 0.0)),
        std_total_demand=0.0,
        mean_u_by_t=[float(v) for v in summary["u_by_t"]],
        mean_v_by_t=[float(v) for v in summary["v_by_t"]],
        mean_a_M1_by_t=[float(v) for v in summary["a_M1_by_t"]],
        mean_a_M2_by_t=[float(v) for v in summary["a_M2_by_t"]],
        mean_age_M2_by_t=[float(v) for v in summary["age_M2_by_t"]],
        mean_rho_M2_by_t=[float(v) for v in summary["rho_M2_by_t"]],
        activation_periods=[int(summary["activation_period"])],
        withdrawal_periods=[int(summary["withdrawal_period"])],
        mean_terminal_value=float(summary.get("terminal_value", 0.0)),
    )


FIXED_HORIZON = 30


def _build_fixed_core_tariff_series(horizon: int = FIXED_HORIZON) -> Tuple[List[float], List[float]]:
    """Core-market tariffs for the fixed 30-period baseline.

    Keep the sharp early observed shock, then let the last 18 periods fluctuate
    mildly around their post-shock levels so that demand does not flatten into a
    completely constant plateau after period 12.
    """
    if horizon < 12:
        raise ValueError("Fixed baseline horizon must be at least 12 periods")

    jo_head = [1.00, 1.10, 1.20, 2.00, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10]
    # jo_head = [0.85, 0.85, 0.85, 0.85, 0.55, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    jn_head = [0.025, 0.025, 0.025, 0.35, 0.25, 0.10, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]

    jo_tail_pattern = [1.13, 1.15, 1.16, 1.14, 1.17, 1.15, 1.14, 1.16, 1.15, 1.13, 1.17, 1.16, 1.14, 1.15, 1.16, 1.14, 1.17, 1.15]
    jn_tail_pattern = [0.23, 0.25, 0.26, 0.24, 0.27, 0.25, 0.24, 0.26, 0.25, 0.23, 0.27, 0.26, 0.24, 0.25, 0.26, 0.24, 0.27, 0.25]

    jo_kc = jo_head + jo_tail_pattern[: max(horizon - 12, 0)]
    jn_kc = jn_head + jn_tail_pattern[: max(horizon - 12, 0)]

    if len(jo_kc) != horizon or len(jn_kc) != horizon:
        raise ValueError(f"Core tariff series must both have length {horizon}")
    return jo_kc, jn_kc


def _build_fixed_additional_tariff_series(horizon: int = FIXED_HORIZON, seed: int = 20260325) -> Tuple[List[float], List[float]]:
    """Additional-market tariffs with a wider 0.01--0.30 range.

    The series follows a reproducible seasonal-style base pattern with small
    seeded jitter instead of near-zero noise. This gives the D2 lanes a visible
    role in total demand and realised-cost dynamics.
    """
    rng = random.Random(seed)
    jo_base = [0.05, 0.08, 0.12, 0.18, 0.15, 0.10, 0.07, 0.11, 0.16, 0.22, 0.14, 0.09]
    jn_base = [0.08, 0.12, 0.18, 0.24, 0.20, 0.24, 0.10, 0.16, 0.22, 0.28, 0.19, 0.13]
    # jn_base = [0.18, 0.22, 0.28, 0.34, 0.30, 0.34, 0.20, 0.26, 0.32, 0.38, 0.29, 0.23]


    def _series_from_base(base: List[float]) -> List[float]:
        out: List[float] = []
        idx = 0
        while len(out) < horizon:
            level = base[idx % len(base)]
            jitter = rng.uniform(-0.02, 0.02)
            out.append(round(min(0.30, max(0.01, level + jitter)), 3))
            idx += 1
        return out

    jo_ka = _series_from_base(jo_base)
    jn_ka = _series_from_base(jn_base)
    return jo_ka, jn_ka


def make_fixed_baseline_scenario(validate: bool = False) -> SupplyChainScenario:
    base = build_story_case(validate=False)
    scenario = copy.deepcopy(base)

    scenario.H = FIXED_HORIZON
    scenario.regimes = [1]
    scenario.P = [[1.0]]
    scenario.xi_1 = 1
    scenario.xi_2_forced = 1

    jo = scenario.legacy_plant_id
    jn = scenario.candidate_plant_id
    kc = scenario.core_market_id
    ka = scenario.additional_market_id

    jo_kc, jn_kc = _build_fixed_core_tariff_series(horizon=int(scenario.H))
    jo_ka, jn_ka = _build_fixed_additional_tariff_series(horizon=int(scenario.H), seed=20260325)

    scenario.tau_time_series = {
        jo: {kc: jo_kc, ka: jo_ka},
        jn: {kc: jn_kc, ka: jn_ka},
    }

    def _fixed_sample_regime_path(seed: Optional[int] = None):
        return [1] * int(scenario.H)

    scenario.sample_regime_path = _fixed_sample_regime_path  # type: ignore[attr-defined]

    if validate:
        scenario.validate_assumptions()
    return scenario


def get_fixed_tariff_schedule(scenario: SupplyChainScenario) -> Dict[str, Dict[str, List[float]]]:
    schedule = getattr(scenario, "tau_time_series", None)
    if schedule is None:
        raise ValueError("Fixed baseline scenario does not define tau_time_series")
    return schedule


def _candidate_activation_times_for_b(scenario: SupplyChainScenario) -> List[Optional[int]]:
    return [None] + list(range(1, int(scenario.H)))


def _candidate_activation_times_for_c(scenario: SupplyChainScenario) -> List[int]:
    H = int(scenario.H)
    return [t for t in [1, 2] if t < H]


def _planned_action_for_period(
    strategy: Strategy,
    t: int,
    activation_time: Optional[int],
) -> Tuple[int, int]:
    u_t = 1 if activation_time is not None and t == int(activation_time) else 0
    v_t = 0
    if strategy == Strategy.A:
        u_t, v_t = 0, 0
    if strategy == Strategy.B:
        v_t = 0
    return int(u_t), int(v_t)


def _sanitize_action_with_mask(mask, action: Tuple[int, int], strategy: Strategy) -> Tuple[int, int]:
    u_t, v_t = int(action[0]), int(action[1])

    if u_t == 1 and int(mask.u_mask[1]) == 0:
        u_t = 0
    if u_t == 0 and int(mask.u_mask[0]) == 0:
        u_t = 1 if int(mask.u_mask[1]) == 1 else 0

    if strategy == Strategy.B:
        v_t = 0
    else:
        if v_t == 1 and int(mask.v_mask[1]) == 0:
            v_t = 0
        if v_t == 0 and int(mask.v_mask[0]) == 0:
            v_t = 1 if int(mask.v_mask[1]) == 1 else 0

    return int(u_t), int(v_t)


def _rollout_trace_with_plan(
    scenario: SupplyChainScenario,
    strategy: Strategy,
    activation_time: Optional[int],
    episode_seed: int,
) -> List[Dict[str, Any]]:
    env = SCReconfigEnv(
        scenario=scenario,
        strategy=strategy,
        seed=0,
        episode_seed=int(episode_seed),
        activation_mode="rl",
    )
    env.reset(episode_seed=int(episode_seed))

    infos: List[Dict[str, Any]] = []
    done = False
    t = 1
    while not done:
        mask = env.get_action_mask()
        action = _planned_action_for_period(
            strategy=strategy,
            t=t,
            activation_time=activation_time,
        )
        action = _sanitize_action_with_mask(mask, action, strategy)
        out = env.step(action)
        infos.append(out.info)
        done = bool(out.done)
        t += 1
    return infos[: int(scenario.H)]


def _record_key_for_b(record: Dict[str, Any], H: int) -> Tuple[float, int, int]:
    cost = float(record["adjusted_discounted_total_cost"])
    actual_activation = int(record["actual_activation_period"])
    no_act_flag = 0 if actual_activation == 0 else 1
    act_rank = actual_activation if actual_activation > 0 else (H + 1)
    return (cost, no_act_flag, act_rank)


def _record_key_for_c(record: Dict[str, Any], H: int) -> Tuple[float, int, int]:
    cost = float(record["adjusted_discounted_total_cost"])
    actual_activation = int(record["actual_activation_period"])
    actual_withdrawal = int(record["actual_withdrawal_period"])
    act_rank = actual_activation if actual_activation > 0 else (H + 1)
    wd_rank = actual_withdrawal if actual_withdrawal > 0 else (H + 1)
    return (cost, act_rank, wd_rank)


def _search_best_plan_for_b(
    scenario: SupplyChainScenario,
    cfg: TrainEvalConfig,
) -> Tuple[StrategyStats, List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    best_summary: Optional[Dict[str, Any]] = None
    best_trace: Optional[List[Dict[str, Any]]] = None
    best_record: Optional[Dict[str, Any]] = None

    for activation_time in _candidate_activation_times_for_b(scenario):
        trace = _rollout_trace_with_plan(
            scenario=scenario,
            strategy=Strategy.B,
            activation_time=activation_time,
            episode_seed=int(cfg.eval_episode_seed),
        )
        summary = _summarize_trace(trace, gamma=float(scenario.gamma))
        rec = {
            "strategy": "B",
            "planned_activation_period": 0 if activation_time is None else int(activation_time),
            "planned_withdrawal_period": 0,
            "actual_activation_period": int(summary["activation_period"]),
            "actual_withdrawal_period": int(summary["withdrawal_period"]),
            "adjusted_discounted_total_cost": float(summary["discounted_total_cost"]),
            "objective_total_cost": float(summary["total_cost"]),
            "terminal_value": float(summary.get("terminal_value", 0.0)),
        }
        records.append(rec)

        if best_record is None or _record_key_for_b(rec, int(scenario.H)) < _record_key_for_b(best_record, int(scenario.H)):
            best_record = rec
            best_summary = summary
            best_trace = trace

    assert best_summary is not None
    assert best_trace is not None
    return _pack_single_summary_as_stats(best_summary), best_trace, records


def _search_best_plan_for_c(
    scenario: SupplyChainScenario,
    cfg: TrainEvalConfig,
) -> Tuple[StrategyStats, List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    best_summary: Optional[Dict[str, Any]] = None
    best_trace: Optional[List[Dict[str, Any]]] = None
    best_record: Optional[Dict[str, Any]] = None

    for activation_time in _candidate_activation_times_for_c(scenario):
        trace = _rollout_trace_with_plan(
            scenario=scenario,
            strategy=Strategy.C,
            activation_time=activation_time,
            episode_seed=int(cfg.eval_episode_seed),
        )
        summary = _summarize_trace(trace, gamma=float(scenario.gamma))
        rec = {
            "strategy": "C",
            "planned_activation_period": int(activation_time),
            "planned_withdrawal_period": 0,
            "actual_activation_period": int(summary["activation_period"]),
            "actual_withdrawal_period": int(summary["withdrawal_period"]),
            "adjusted_discounted_total_cost": float(summary["discounted_total_cost"]),
            "objective_total_cost": float(summary["total_cost"]),
            "terminal_value": float(summary.get("terminal_value", 0.0)),
        }
        records.append(rec)

        if int(summary["activation_period"]) == 0:
            continue
        if int(summary["withdrawal_period"]) == 0:
            continue

        if best_record is None or _record_key_for_c(rec, int(scenario.H)) < _record_key_for_c(best_record, int(scenario.H)):
            best_record = rec
            best_summary = summary
            best_trace = trace

    if best_summary is None or best_trace is None:
        raise ValueError(
            "No feasible Strategy C plan was found under the current scenario. "
            "This means no candidate plan both activated early and completed the mandatory full-relocation withdrawal within the horizon."
        )

    return _pack_single_summary_as_stats(best_summary), best_trace, records


def _evaluate_strategy_a(
    scenario: SupplyChainScenario,
    cfg: TrainEvalConfig,
) -> Tuple[StrategyStats, List[Dict[str, Any]], List[Dict[str, Any]]]:
    trace = _rollout_trace_with_plan(
        scenario=scenario,
        strategy=Strategy.A,
        activation_time=None,
        episode_seed=int(cfg.eval_episode_seed),
    )
    summary = _summarize_trace(trace, gamma=float(scenario.gamma))
    record = {
        "strategy": "A",
        "planned_activation_period": 0,
        "planned_withdrawal_period": 0,
        "actual_activation_period": 0,
        "actual_withdrawal_period": 0,
        "adjusted_discounted_total_cost": float(summary["discounted_total_cost"]),
        "objective_total_cost": float(summary["total_cost"]),
        "terminal_value": float(summary.get("terminal_value", 0.0)),
    }
    return _pack_single_summary_as_stats(summary), trace, [record]


def evaluate_three_strategies(
    scenario: SupplyChainScenario,
    seeds: List[int],
    cfg: TrainEvalConfig,
    *,
    return_traces: bool = False,
    return_search: bool = False,
):
    stats_a, trace_a, search_a = _evaluate_strategy_a(scenario, cfg)
    stats_b, trace_b, search_b = _search_best_plan_for_b(scenario, cfg)
    stats_c, trace_c, search_c = _search_best_plan_for_c(scenario, cfg)

    stats = {"A": stats_a, "B": stats_b, "C": stats_c}
    traces = {"A": trace_a, "B": trace_b, "C": trace_c}
    search = {"A": search_a, "B": search_b, "C": search_c}

    if not return_traces and not return_search:
        return stats
    if return_traces and not return_search:
        return stats, traces
    if not return_traces and return_search:
        return stats, search
    return stats, traces, search
