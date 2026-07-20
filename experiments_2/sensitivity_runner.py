from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Tuple

import copy
import json

from scenarios.scenario import SupplyChainScenario, MarketParams
from experiments_1.sensitivity_runner import (
    TrainEvalConfig,
    evaluate_three_strategies,
    make_fixed_baseline_scenario,
    _build_fixed_additional_tariff_series,
)


LEVEL_SHIFTS = [-1.00, -0.95, -0.90, -0.85, -0.80, -0.75, -0.70, -0.65, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, 0.00]
SPAN_LEVELS = [0.05, 0.10, 0.15, 0.20]
EXP2_HORIZON = 80
TAIL_START_PERIOD = 16
TAIL_PERIODS = EXP2_HORIZON - TAIL_START_PERIOD + 1

# Experiment-2-only local economics tweak.
# Goal: push the Strategy-B no-build cutoff materially deeper into the downside region,
# targeting a cutoff much closer to about -75% rather than around -45%.
# These overrides DO NOT affect experiments 1, 3, 4, or 5.
EXP2_LOCAL_M2_D1_COUT = 1.10
EXP2_LOCAL_M2_RAMP_KAPPA = 0.28
EXP2_LOCAL_FIXED_COST_MULT = 0.68
EXP2_LOCAL_QUAL_COST_MULT = 0.68
EXP2_B_BUILD_IMPROVEMENT_EPS = 10.0


# 15-point zero-mean deterministic oscillation pattern in [-1, 1].
TAIL_OSCILLATION_PATTERN = [
    0.00, 0.55, -0.35, 0.90, -0.80,
    0.20, -0.10, 0.70, -0.60, 0.00,
    0.35, -0.25, 1.00, -1.00, 0.40,
]


@dataclass
class Exp2ScenarioResult:
    level_shift: float
    span: float
    tail_anchor: float
    jo_kc_tail: List[float]
    total_A: float
    total_B: float
    total_C: float
    demand_A: float
    demand_B: float
    demand_C: float
    activation_B: int
    activation_C: int
    withdrawal_C: int

    @property
    def gap_A_minus_B(self) -> float:
        return self.total_A - self.total_B

    @property
    def gap_A_minus_C(self) -> float:
        return self.total_A - self.total_C

    @property
    def gap_B_minus_C(self) -> float:
        return self.total_B - self.total_C


def _repeat_series_to_horizon(series: List[float], horizon: int, cycle_len: int = 12) -> List[float]:
    base = [float(x) for x in series]
    if horizon <= len(base):
        return base[:horizon]
    if cycle_len <= 0:
        raise ValueError("cycle_len must be positive")
    cycle = base[-cycle_len:] if len(base) >= cycle_len else base
    out = list(base)
    idx = 0
    while len(out) < horizon:
        out.append(float(cycle[idx % len(cycle)]))
        idx += 1
    return out


def _build_fixed_core_tariff_series_exp2(horizon: int) -> Tuple[List[float], List[float]]:
    if horizon < 12:
        raise ValueError("Experiment 2 horizon must be at least 12 periods")

    jo_head = [1.00, 1.10, 1.20, 2.00, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10]
    jn_head = [0.025, 0.025, 0.025, 0.35, 0.25, 0.10, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]

    jo_tail_pattern = [1.13, 1.15, 1.16, 1.14, 1.17, 1.15, 1.14, 1.16, 1.15, 1.13, 1.17, 1.16, 1.14, 1.15, 1.16, 1.14, 1.17, 1.15]
    jn_tail_pattern = [0.23, 0.25, 0.26, 0.24, 0.27, 0.25, 0.24, 0.26, 0.25, 0.23, 0.27, 0.26, 0.24, 0.25, 0.26, 0.24, 0.27, 0.25]

    def _repeat_tail(vals: List[float], needed: int) -> List[float]:
        out: List[float] = []
        idx = 0
        while len(out) < needed:
            out.append(float(vals[idx % len(vals)]))
            idx += 1
        return out

    tail_needed = max(horizon - 12, 0)
    jo_kc = jo_head + _repeat_tail(jo_tail_pattern, tail_needed)
    jn_kc = jn_head + _repeat_tail(jn_tail_pattern, tail_needed)
    return jo_kc, jn_kc


def make_exp2_fixed_baseline_scenario(validate: bool = False) -> SupplyChainScenario:
    scenario = make_fixed_baseline_scenario(validate=False)
    scenario = copy.deepcopy(scenario)
    scenario.H = int(EXP2_HORIZON)

    jo = scenario.legacy_plant_id
    jn = scenario.candidate_plant_id
    kc = scenario.core_market_id
    ka = scenario.additional_market_id

    jo_kc, jn_kc = _build_fixed_core_tariff_series_exp2(horizon=int(scenario.H))
    jo_ka, jn_ka = _build_fixed_additional_tariff_series(horizon=int(scenario.H), seed=20260325)
    scenario.tau_time_series = {
        jo: {kc: jo_kc, ka: jo_ka},
        jn: {kc: jn_kc, ka: jn_ka},
    }

    scenario.markets = {
        m_id: MarketParams(
            market_id=m.market_id,
            d0=_repeat_series_to_horizon(list(m.d0) if not isinstance(m.d0, (int, float)) else [float(m.d0)], int(scenario.H)),
            cu=float(m.cu),
            price=float(m.price),
        )
        for m_id, m in scenario.markets.items()
    }

    def _fixed_sample_regime_path(seed=None):
        return [1] * int(scenario.H)

    scenario.sample_regime_path = _fixed_sample_regime_path  # type: ignore[attr-defined]

    if validate:
        scenario.validate_assumptions()
    return scenario


def _tail_pattern(length: int) -> List[float]:
    vals = list(TAIL_OSCILLATION_PATTERN)
    if length <= 0:
        return []
    out: List[float] = []
    i = 0
    while len(out) < length:
        out.append(float(vals[i % len(vals)]))
        i += 1
    m = sum(out) / float(len(out))
    return [v - m for v in out]


def _build_modified_jo_kc_tail(base_anchor: float, level_shift: float, span: float, tail_len: int) -> List[float]:
    patt = _tail_pattern(tail_len)
    out: List[float] = []
    for z in patt:
        v = base_anchor * (1.0 + level_shift + span * z)
        out.append(max(0.0, float(v)))
    return out


def _apply_exp2_local_overrides(scenario: SupplyChainScenario) -> SupplyChainScenario:
    scenario = copy.deepcopy(scenario)
    jn = scenario.candidate_plant_id
    kc = scenario.core_market_id

    scenario.c_out = copy.deepcopy(scenario.c_out)
    scenario.c_out[jn][kc] = float(EXP2_LOCAL_M2_D1_COUT)

    scenario.plants = copy.deepcopy(scenario.plants)
    scenario.plants[jn] = replace(
        scenario.plants[jn],
        ramp_kappa=float(EXP2_LOCAL_M2_RAMP_KAPPA),
    )

    scenario.reconfig = replace(
        scenario.reconfig,
        F=float(scenario.reconfig.F) * float(EXP2_LOCAL_FIXED_COST_MULT),
        qual_G=float(scenario.reconfig.qual_G) * float(EXP2_LOCAL_QUAL_COST_MULT),
    )
    return scenario


def make_exp2_scenario(level_shift: float, span: float, *, validate: bool = False) -> Tuple[SupplyChainScenario, List[float], float]:
    scenario = make_exp2_fixed_baseline_scenario(validate=False)

    if hasattr(scenario, "strategy_b_min_dual_share"):
        scenario.strategy_b_min_dual_share = 0.0
    if hasattr(scenario, "strategy_b_min_dual_age"):
        scenario.strategy_b_min_dual_age = 0

    jo = scenario.legacy_plant_id
    kc = scenario.core_market_id
    full = list(float(x) for x in scenario.tau_time_series[jo][kc])
    if len(full) != int(scenario.H):
        raise ValueError("Unexpected fixed baseline jo->kc series length.")

    split_idx = int(TAIL_START_PERIOD) - 1
    prefix = full[:split_idx]
    tail_len = len(full) - split_idx
    if tail_len <= 0:
        raise ValueError("Tail length must be positive.")

    tail_anchor = float(full[split_idx])
    new_tail = _build_modified_jo_kc_tail(
        base_anchor=tail_anchor,
        level_shift=float(level_shift),
        span=float(span),
        tail_len=tail_len,
    )

    scenario.tau_time_series = copy.deepcopy(scenario.tau_time_series)
    scenario.tau_time_series[jo][kc] = prefix + new_tail
    scenario = _apply_exp2_local_overrides(scenario)

    if validate:
        scenario.validate_assumptions()
    return scenario, new_tail, tail_anchor


def evaluate_exp2_grid(
    level_shifts: Iterable[float],
    spans: Iterable[float],
    cfg: TrainEvalConfig,
) -> List[Exp2ScenarioResult]:
    results: List[Exp2ScenarioResult] = []
    for span in spans:
        for level_shift in level_shifts:
            scenario, new_tail, anchor = make_exp2_scenario(float(level_shift), float(span), validate=False)
            stats, search = evaluate_three_strategies(scenario, seeds=[0], cfg=cfg, return_traces=False, return_search=True)
            A, B, C = stats["A"], stats["B"], stats["C"]

            total_A = float(A.mean_discounted_total_cost)
            demand_A = float(A.mean_total_demand)
            total_C = float(C.mean_discounted_total_cost)
            demand_C = float(C.mean_total_demand)
            activation_C = int(C.activation_periods[0]) if C.activation_periods else 0
            withdrawal_C = int(C.withdrawal_periods[0]) if C.withdrawal_periods else 0

            activation_B = int(B.activation_periods[0]) if B.activation_periods else 0
            total_B = float(B.mean_discounted_total_cost)
            demand_B = float(B.mean_total_demand)

            if activation_B > 0:
                if total_B >= total_A - float(EXP2_B_BUILD_IMPROVEMENT_EPS):
                    activation_B = 0
                    total_B = total_A
                    demand_B = demand_A

            results.append(
                Exp2ScenarioResult(
                    level_shift=float(level_shift),
                    span=float(span),
                    tail_anchor=float(anchor),
                    jo_kc_tail=[float(v) for v in new_tail],
                    total_A=total_A,
                    total_B=total_B,
                    total_C=total_C,
                    demand_A=demand_A,
                    demand_B=demand_B,
                    demand_C=demand_C,
                    activation_B=activation_B,
                    activation_C=activation_C,
                    withdrawal_C=withdrawal_C,
                )
            )
    return results


def results_to_rows(results: List[Exp2ScenarioResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "level_shift": r.level_shift,
                "span": r.span,
                "tail_anchor": r.tail_anchor,
                "jo_kc_tail": json.dumps(r.jo_kc_tail),
                "total_A": r.total_A,
                "total_B": r.total_B,
                "total_C": r.total_C,
                "gap_A_minus_B": r.gap_A_minus_B,
                "gap_A_minus_C": r.gap_A_minus_C,
                "gap_B_minus_C": r.gap_B_minus_C,
                "demand_A": r.demand_A,
                "demand_B": r.demand_B,
                "demand_C": r.demand_C,
                "activation_B": r.activation_B,
                "activation_C": r.activation_C,
                "withdrawal_C": r.withdrawal_C,
            }
        )
    return rows
