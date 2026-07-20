from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import copy
from dataclasses import replace

from scenarios.scenario import SupplyChainScenario, MarketParams
from scenarios.story_case import build_story_case, _monthly_baseline_series
from experiments_1.sensitivity_runner import TrainEvalConfig, evaluate_three_strategies


HORIZON_LEVELS = [30, 40, 50, 60, 70, 80]
CORE_HEAD_LEN = 12


@dataclass
class Exp3HorizonResult:
    horizon: int
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

    @property
    def avg_discounted_cost_A(self) -> float:
        return self.total_A / float(self.horizon)

    @property
    def avg_discounted_cost_B(self) -> float:
        return self.total_B / float(self.horizon)

    @property
    def avg_discounted_cost_C(self) -> float:
        return self.total_C / float(self.horizon)


def _repeat_to_length(values: List[float], length: int) -> List[float]:
    if not values:
        raise ValueError("values must be non-empty")
    out: List[float] = []
    i = 0
    while len(out) < length:
        out.append(float(values[i % len(values)]))
        i += 1
    return out


def _build_extended_fixed_core_tariff_series(horizon: int) -> tuple[List[float], List[float]]:
    if horizon < CORE_HEAD_LEN:
        raise ValueError("Horizon sensitivity study requires horizon >= 12")

    jo_head = [1.00, 1.10, 1.20, 2.00, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10]
    jn_head = [0.025, 0.025, 0.025, 0.35, 0.25, 0.10, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]

    jo_tail_pattern = [1.13, 1.15, 1.16, 1.14, 1.17, 1.15, 1.14, 1.16, 1.15, 1.13, 1.17, 1.16, 1.14, 1.15, 1.16, 1.14, 1.17, 1.15]
    jn_tail_pattern = [0.23, 0.25, 0.26, 0.24, 0.27, 0.25, 0.24, 0.26, 0.25, 0.23, 0.27, 0.26, 0.24, 0.25, 0.26, 0.24, 0.27, 0.25]

    tail_len = horizon - CORE_HEAD_LEN
    jo_kc = jo_head + _repeat_to_length(jo_tail_pattern, tail_len)
    jn_kc = jn_head + _repeat_to_length(jn_tail_pattern, tail_len)
    return jo_kc, jn_kc


def _build_extended_fixed_additional_tariff_series(horizon: int) -> tuple[List[float], List[float]]:
    jo_base = [0.05, 0.08, 0.12, 0.18, 0.15, 0.10, 0.07, 0.11, 0.16, 0.22, 0.14, 0.09]
    jn_base = [0.08, 0.12, 0.18, 0.24, 0.20, 0.24, 0.10, 0.16, 0.22, 0.28, 0.19, 0.13]
    return _repeat_to_length(jo_base, horizon), _repeat_to_length(jn_base, horizon)


def _extend_market_demands(scenario: SupplyChainScenario, horizon: int) -> None:
    d1_cycle = [0.90, 0.94, 0.98, 1.03, 1.08, 1.05, 1.00, 0.96, 0.92, 0.90, 0.95, 1.00]
    d2_cycle = [0.85, 0.90, 0.96, 1.04, 1.12, 1.08, 1.01, 0.95, 0.89, 0.85, 0.92, 1.00]
    d1_series = _monthly_baseline_series(mean_level=43.5, cycle=d1_cycle, horizon=horizon)
    d2_series = _monthly_baseline_series(mean_level=20.0, cycle=d2_cycle, horizon=horizon)

    old_d1 = scenario.markets[scenario.core_market_id]
    old_d2 = scenario.markets[scenario.additional_market_id]
    scenario.markets = dict(scenario.markets)
    scenario.markets[scenario.core_market_id] = MarketParams(
        market_id=old_d1.market_id,
        d0=d1_series,
        cu=old_d1.cu,
        price=old_d1.price,
    )
    scenario.markets[scenario.additional_market_id] = MarketParams(
        market_id=old_d2.market_id,
        d0=d2_series,
        cu=old_d2.cu,
        price=old_d2.price,
    )


def make_horizon_sensitivity_scenario(horizon: int, *, validate: bool = False) -> SupplyChainScenario:
    base = build_story_case(validate=False)
    scenario = copy.deepcopy(base)

    scenario.H = int(horizon)
    scenario.regimes = [1]
    scenario.P = [[1.0]]
    scenario.xi_1 = 1
    scenario.xi_2_forced = 1

    _extend_market_demands(scenario, int(horizon))

    jo = scenario.legacy_plant_id
    jn = scenario.candidate_plant_id
    kc = scenario.core_market_id
    ka = scenario.additional_market_id

    jo_kc, jn_kc = _build_extended_fixed_core_tariff_series(int(horizon))
    jo_ka, jn_ka = _build_extended_fixed_additional_tariff_series(int(horizon))
    scenario.tau_time_series = {
        jo: {kc: jo_kc, ka: jo_ka},
        jn: {kc: jn_kc, ka: jn_ka},
    }

    def _fixed_sample_regime_path(seed=None):
        return [1] * int(scenario.H)

    scenario.sample_regime_path = _fixed_sample_regime_path  # type: ignore[attr-defined]

    if validate:
        scenario.validate_assumptions()
    return scenario


def evaluate_exp3_horizon_grid(
    horizons: Iterable[int],
    cfg: TrainEvalConfig,
) -> List[Exp3HorizonResult]:
    results: List[Exp3HorizonResult] = []
    for horizon in horizons:
        scenario = make_horizon_sensitivity_scenario(int(horizon), validate=False)
        stats = evaluate_three_strategies(scenario, seeds=[0], cfg=cfg, return_traces=False, return_search=False)
        A, B, C = stats["A"], stats["B"], stats["C"]
        results.append(
            Exp3HorizonResult(
                horizon=int(horizon),
                total_A=float(A.mean_discounted_total_cost),
                total_B=float(B.mean_discounted_total_cost),
                total_C=float(C.mean_discounted_total_cost),
                demand_A=float(A.mean_total_demand),
                demand_B=float(B.mean_total_demand),
                demand_C=float(C.mean_total_demand),
                activation_B=int(B.activation_periods[0]) if B.activation_periods else 0,
                activation_C=int(C.activation_periods[0]) if C.activation_periods else 0,
                withdrawal_C=int(C.withdrawal_periods[0]) if C.withdrawal_periods else 0,
            )
        )
    return results


def results_to_rows(results: List[Exp3HorizonResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "horizon": r.horizon,
                "total_A": r.total_A,
                "total_B": r.total_B,
                "total_C": r.total_C,
                "gap_A_minus_B": r.gap_A_minus_B,
                "gap_A_minus_C": r.gap_A_minus_C,
                "gap_B_minus_C": r.gap_B_minus_C,
                "avg_discounted_cost_A": r.avg_discounted_cost_A,
                "avg_discounted_cost_B": r.avg_discounted_cost_B,
                "avg_discounted_cost_C": r.avg_discounted_cost_C,
                "demand_A": r.demand_A,
                "demand_B": r.demand_B,
                "demand_C": r.demand_C,
                "activation_B": r.activation_B,
                "activation_C": r.activation_C,
                "withdrawal_C": r.withdrawal_C,
            }
        )
    return rows
