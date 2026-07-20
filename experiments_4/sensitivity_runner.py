from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import copy
import json
import random

from scenarios.scenario import SupplyChainScenario
from experiments_1.sensitivity_runner import (
    TrainEvalConfig,
    evaluate_three_strategies,
    get_fixed_tariff_schedule,
    make_fixed_baseline_scenario,
)


T1_LOW_TARIFF = 0.05
TARIFF_FLOOR = 0.01
STABLE_BAND = 0.10
T2_TARIFF_GRID = [round(x / 100.0, 2) for x in range(5, 101, 5)]
BASE_PATH_SEED = 20260330

# Experiment 4 only
# This local c_out override is applied only inside make_exp4_scenario().
# Other experiments continue to use the global baseline c_out from story_case.
EXP4_C_OUT_OVERRIDE: Dict[str, Dict[str, float]] = {
    "M1": {"D1": 1.0, "D2": 6.0},
    "M2": {"D1": 2.2, "D2": 1.8},
}


@dataclass
class Exp4ScenarioResult:
    t2_tariff: float
    path_seed: int
    jo_kc_series: List[float]
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


def _stable_next(prev: float, rng: random.Random) -> float:
    lo = max(TARIFF_FLOOR, float(prev) - STABLE_BAND)
    hi = float(prev) + STABLE_BAND
    return round(rng.uniform(lo, hi), 3)


def _build_stable_jo_kc_series(horizon: int, t2_tariff: float, path_seed: int) -> List[float]:
    if horizon < 2:
        raise ValueError("Experiment 4 requires horizon >= 2.")
    rng = random.Random(int(path_seed))
    series = [round(T1_LOW_TARIFF, 3), round(float(t2_tariff), 3)]
    while len(series) < horizon:
        series.append(_stable_next(series[-1], rng))
    return series


def _apply_exp4_local_cout_override(scenario: SupplyChainScenario) -> None:
    scenario.c_out = copy.deepcopy(scenario.c_out)
    for plant_id, market_map in EXP4_C_OUT_OVERRIDE.items():
        if plant_id not in scenario.c_out:
            continue
        for market_id, value in market_map.items():
            scenario.c_out[plant_id][market_id] = float(value)


def make_exp4_scenario(t2_tariff: float, *, validate: bool = False) -> SupplyChainScenario:
    scenario = make_fixed_baseline_scenario(validate=False)
    scenario = copy.deepcopy(scenario)

    fixed = get_fixed_tariff_schedule(scenario)
    jo = scenario.legacy_plant_id
    kc = scenario.core_market_id

    scenario.tau_time_series = copy.deepcopy(fixed)
    path_seed = BASE_PATH_SEED + int(round(float(t2_tariff) * 1000.0))
    scenario.tau_time_series[jo][kc] = _build_stable_jo_kc_series(
        horizon=int(scenario.H),
        t2_tariff=float(t2_tariff),
        path_seed=path_seed,
    )

    _apply_exp4_local_cout_override(scenario)

    if validate:
        scenario.validate_assumptions()
    return scenario


def evaluate_exp4_grid(
    t2_tariff_grid: Iterable[float],
    cfg: TrainEvalConfig,
) -> List[Exp4ScenarioResult]:
    results: List[Exp4ScenarioResult] = []
    for t2_tariff in t2_tariff_grid:
        scenario = make_exp4_scenario(float(t2_tariff), validate=False)
        stats = evaluate_three_strategies(
            scenario,
            seeds=[0],
            cfg=cfg,
            return_traces=False,
            return_search=False,
        )
        A, B, C = stats["A"], stats["B"], stats["C"]
        jo = scenario.legacy_plant_id
        kc = scenario.core_market_id
        results.append(
            Exp4ScenarioResult(
                t2_tariff=float(t2_tariff),
                path_seed=BASE_PATH_SEED + int(round(float(t2_tariff) * 1000.0)),
                jo_kc_series=[float(v) for v in scenario.tau_time_series[jo][kc]],
                total_A=float(A.mean_discounted_total_cost),
                total_B=float(B.mean_discounted_total_cost),
                total_C=float(C.mean_discounted_total_cost),
                demand_A=float(A.mean_total_demand),
                demand_B=float(B.mean_total_demand),
                demand_C=float(C.mean_total_demand),
                activation_B=int(ACT) if (ACT := (B.activation_periods[0] if B.activation_periods else 0)) else 0,
                activation_C=int(ACT) if (ACT := (C.activation_periods[0] if C.activation_periods else 0)) else 0,
                withdrawal_C=int(WD) if (WD := (C.withdrawal_periods[0] if C.withdrawal_periods else 0)) else 0,
            )
        )
    return results


def results_to_rows(results: List[Exp4ScenarioResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "t2_tariff": r.t2_tariff,
                "path_seed": r.path_seed,
                "exp4_c_out_override": json.dumps(EXP4_C_OUT_OVERRIDE),
                "jo_kc_series": json.dumps(r.jo_kc_series),
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
