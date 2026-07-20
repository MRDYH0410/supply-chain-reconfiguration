from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
RESET_THRESHOLD = 2.50
RESET_STREAK = 3
T2_TARIFF_GRID = [round(x / 100.0, 2) for x in range(5, 101, 5)]
BASE_PATH_SEED = 20260331
DEFAULT_N_REPLICATIONS = 1000

RISK_BUCKETS: List[Tuple[float, float, float]] = [
    (0.20, 0.00, 0.10),
    (0.30, 0.10, 0.20),
    (0.40, 0.20, 0.50),
    (0.08, 0.50, 1.00),
    (0.02, 1.00, 1.50),
]


@dataclass
class Exp5ScenarioResult:
    t2_tariff: float
    n_replications: int
    first_path_seed: int
    mean_reset_count: float
    std_reset_count: float
    jo_kc_series_mean: List[float]
    jo_kc_series_std: List[float]
    total_A: float
    total_B: float
    total_C: float
    std_total_A: float
    std_total_B: float
    std_total_C: float
    demand_A: float
    demand_B: float
    demand_C: float
    std_demand_A: float
    std_demand_B: float
    std_demand_C: float
    activation_B: float
    activation_C: float
    withdrawal_C: float
    std_activation_B: float
    std_activation_C: float
    std_withdrawal_C: float

    @property
    def gap_A_minus_B(self) -> float:
        return self.total_A - self.total_B

    @property
    def gap_A_minus_C(self) -> float:
        return self.total_A - self.total_C

    @property
    def gap_B_minus_C(self) -> float:
        return self.total_B - self.total_C


def _mean(values: Sequence[float]) -> float:
    return float(fmean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _mean_series(series_list: Sequence[Sequence[float]]) -> List[float]:
    if not series_list:
        return []
    H = len(series_list[0])
    return [_mean([float(series[t]) for series in series_list]) for t in range(H)]


def _std_series(series_list: Sequence[Sequence[float]]) -> List[float]:
    if not series_list:
        return []
    H = len(series_list[0])
    return [_std([float(series[t]) for series in series_list]) for t in range(H)]


def _draw_bucket(rng: random.Random) -> Tuple[float, float]:
    u = rng.random()
    acc = 0.0
    for p, lo, hi in RISK_BUCKETS:
        acc += p
        if u <= acc + 1e-12:
            return lo, hi
    return RISK_BUCKETS[-1][1], RISK_BUCKETS[-1][2]


def _risk_next(prev: float, rng: random.Random) -> float:
    lo, hi = _draw_bucket(rng)
    mag = rng.uniform(lo, hi)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return round(max(TARIFF_FLOOR, float(prev) + sign * mag), 3)


def _build_risk_jo_kc_series(horizon: int, t2_tariff: float, path_seed: int) -> Tuple[List[float], int]:
    if horizon < 2:
        raise ValueError("Experiment 5 requires horizon >= 2.")
    rng = random.Random(int(path_seed))
    anchor = round(float(t2_tariff), 3)
    series = [round(T1_LOW_TARIFF, 3), anchor]
    reset_count = 0

    while len(series) < horizon:
        if len(series) >= RESET_STREAK and all(v > RESET_THRESHOLD for v in series[-RESET_STREAK:]):
            series.append(anchor)
            reset_count += 1
            continue
        series.append(_risk_next(series[-1], rng))

    return series, reset_count


def _path_seed_for(t2_tariff: float, replication_idx: int) -> int:
    t2_key = int(round(float(t2_tariff) * 1000.0))
    return int(BASE_PATH_SEED + 100000 * t2_key + replication_idx)


def make_exp5_scenario(
    t2_tariff: float,
    *,
    path_seed: int | None = None,
    validate: bool = False,
) -> Tuple[SupplyChainScenario, int, int]:
    scenario = make_fixed_baseline_scenario(validate=False)
    scenario = copy.deepcopy(scenario)

    fixed = get_fixed_tariff_schedule(scenario)
    jo = scenario.legacy_plant_id
    kc = scenario.core_market_id

    scenario.tau_time_series = copy.deepcopy(fixed)
    if path_seed is None:
        path_seed = _path_seed_for(float(t2_tariff), 0)
    jo_series, reset_count = _build_risk_jo_kc_series(
        horizon=int(scenario.H),
        t2_tariff=float(t2_tariff),
        path_seed=int(path_seed),
    )
    scenario.tau_time_series[jo][kc] = jo_series

    if validate:
        scenario.validate_assumptions()
    return scenario, reset_count, int(path_seed)


def evaluate_exp5_grid(
    t2_tariff_grid: Iterable[float],
    cfg: TrainEvalConfig,
    *,
    n_replications: int = DEFAULT_N_REPLICATIONS,
) -> List[Exp5ScenarioResult]:
    results: List[Exp5ScenarioResult] = []

    for t2_tariff in t2_tariff_grid:
        tariff_series_samples: List[List[float]] = []
        reset_counts: List[float] = []
        total_A_list: List[float] = []
        total_B_list: List[float] = []
        total_C_list: List[float] = []
        demand_A_list: List[float] = []
        demand_B_list: List[float] = []
        demand_C_list: List[float] = []
        activation_B_list: List[float] = []
        activation_C_list: List[float] = []
        withdrawal_C_list: List[float] = []
        first_seed: int | None = None

        for rep in range(int(n_replications)):
            path_seed = _path_seed_for(float(t2_tariff), rep)
            if first_seed is None:
                first_seed = path_seed
            scenario, reset_count, _ = make_exp5_scenario(float(t2_tariff), path_seed=path_seed, validate=False)
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

            tariff_series_samples.append([float(v) for v in scenario.tau_time_series[jo][kc]])
            reset_counts.append(float(reset_count))
            total_A_list.append(float(A.mean_discounted_total_cost))
            total_B_list.append(float(B.mean_discounted_total_cost))
            total_C_list.append(float(C.mean_discounted_total_cost))
            demand_A_list.append(float(A.mean_total_demand))
            demand_B_list.append(float(B.mean_total_demand))
            demand_C_list.append(float(C.mean_total_demand))
            activation_B_list.append(float(B.activation_periods[0]) if B.activation_periods else 0.0)
            activation_C_list.append(float(C.activation_periods[0]) if C.activation_periods else 0.0)
            withdrawal_C_list.append(float(C.withdrawal_periods[0]) if C.withdrawal_periods else 0.0)

        results.append(
            Exp5ScenarioResult(
                t2_tariff=float(t2_tariff),
                n_replications=int(n_replications),
                first_path_seed=int(first_seed or _path_seed_for(float(t2_tariff), 0)),
                mean_reset_count=_mean(reset_counts),
                std_reset_count=_std(reset_counts),
                jo_kc_series_mean=_mean_series(tariff_series_samples),
                jo_kc_series_std=_std_series(tariff_series_samples),
                total_A=_mean(total_A_list),
                total_B=_mean(total_B_list),
                total_C=_mean(total_C_list),
                std_total_A=_std(total_A_list),
                std_total_B=_std(total_B_list),
                std_total_C=_std(total_C_list),
                demand_A=_mean(demand_A_list),
                demand_B=_mean(demand_B_list),
                demand_C=_mean(demand_C_list),
                std_demand_A=_std(demand_A_list),
                std_demand_B=_std(demand_B_list),
                std_demand_C=_std(demand_C_list),
                activation_B=_mean(activation_B_list),
                activation_C=_mean(activation_C_list),
                withdrawal_C=_mean(withdrawal_C_list),
                std_activation_B=_std(activation_B_list),
                std_activation_C=_std(activation_C_list),
                std_withdrawal_C=_std(withdrawal_C_list),
            )
        )
    return results


def results_to_rows(results: List[Exp5ScenarioResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "t2_tariff": r.t2_tariff,
                "n_replications": r.n_replications,
                "first_path_seed": r.first_path_seed,
                "mean_reset_count": r.mean_reset_count,
                "std_reset_count": r.std_reset_count,
                "jo_kc_series_mean": json.dumps(r.jo_kc_series_mean),
                "jo_kc_series_std": json.dumps(r.jo_kc_series_std),
                "total_A": r.total_A,
                "std_total_A": r.std_total_A,
                "total_B": r.total_B,
                "std_total_B": r.std_total_B,
                "total_C": r.total_C,
                "std_total_C": r.std_total_C,
                "gap_A_minus_B": r.gap_A_minus_B,
                "gap_A_minus_C": r.gap_A_minus_C,
                "gap_B_minus_C": r.gap_B_minus_C,
                "demand_A": r.demand_A,
                "std_demand_A": r.std_demand_A,
                "demand_B": r.demand_B,
                "std_demand_B": r.std_demand_B,
                "demand_C": r.demand_C,
                "std_demand_C": r.std_demand_C,
                "activation_B": r.activation_B,
                "std_activation_B": r.std_activation_B,
                "activation_C": r.activation_C,
                "std_activation_C": r.std_activation_C,
                "withdrawal_C": r.withdrawal_C,
                "std_withdrawal_C": r.std_withdrawal_C,
            }
        )
    return rows
