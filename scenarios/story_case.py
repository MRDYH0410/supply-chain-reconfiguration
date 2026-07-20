from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

from .scenario import (
    SupplyChainScenario,
    PlantParams,
    SupplierParams,
    MarketParams,
    ReconfigParams,
    RouteParams,
)


DEFAULT_TARIFF_DEMAND_CURVE = {
    "curve_model": "exponential",
    "curve_params": {"eta": 0.13653203011845347},
}


MONTHLY_GAMMA = 0.9963
DEFAULT_HORIZON = 30


def _load_tariff_demand_curve() -> dict:
    """Load the fitted annual tariff -> demand curve produced by
    paras/annual_tariff_demand_curve.py.

    If the calibration artifacts are unavailable, fall back to the currently
    selected exponential curve so that the scenario remains runnable.
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    json_path = repo_root / "paras" / "tariff_demand_curve_outputs" / "tariff_demand_function.json"
    grid_path = repo_root / "paras" / "tariff_demand_curve_outputs" / "tariff_curve_grid.csv"

    if not json_path.exists():
        return dict(DEFAULT_TARIFF_DEMAND_CURVE)

    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return dict(DEFAULT_TARIFF_DEMAND_CURVE)

    curve = {
        "curve_model": str(payload.get("selected_curve_model", DEFAULT_TARIFF_DEMAND_CURVE["curve_model"])),
        "curve_params": payload.get("curve_params", DEFAULT_TARIFF_DEMAND_CURVE["curve_params"]),
    }

    if grid_path.exists():
        try:
            tariff_grid = []
            retention_grid = []
            with grid_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tariff_grid.append(float(row["tariff_decimal"]))
                    retention_grid.append(float(row["predicted_retention_factor"]))
            if len(tariff_grid) >= 2:
                curve["tariff_grid"] = tariff_grid
                curve["retention_grid"] = retention_grid
        except Exception:
            pass

    return curve


def _repeat_cycle(cycle: Iterable[float], horizon: int) -> List[float]:
    vals = [float(x) for x in cycle]
    if not vals:
        raise ValueError("cycle must be non-empty")
    out = []
    idx = 0
    while len(out) < horizon:
        out.append(vals[idx % len(vals)])
        idx += 1
    return out


def _monthly_baseline_series(mean_level: float, cycle: Iterable[float], horizon: int) -> List[float]:
    weights = [float(x) for x in cycle]
    if not weights:
        raise ValueError("monthly demand cycle must be non-empty")
    avg = sum(weights) / len(weights)
    if avg <= 0.0:
        raise ValueError("monthly demand cycle must have positive mean")
    normalised = [w / avg for w in weights]
    return [mean_level * m for m in _repeat_cycle(normalised, horizon)]


def build_story_case(validate: bool = True) -> SupplyChainScenario:
    # ------------------------------------------------------------------
    # Quantity and time units
    # ------------------------------------------------------------------
    # 1 model period = 1 month
    # H = 30 months
    # Demand cycles are intentionally stronger than before so that the total
    # demand and cost traces retain visible curvature even when tariffs are frozen.
    # ------------------------------------------------------------------

    automotive_cost_per_vehicle_usd = 28_551.0
    material_share_mid = 0.725
    conversion_share_mid = 1.0 - material_share_mid

    conversion_cost_per_vehicle_usd = automotive_cost_per_vehicle_usd * conversion_share_mid
    model_money_unit_usd_per_vehicle = conversion_cost_per_vehicle_usd / 4.0
    model_money_unit_musd_per_ku = model_money_unit_usd_per_vehicle / 1000.0

    candidate_capex_musd = 5_000.0
    qualification_cost_share_of_capex = 0.01
    salvage_rate = 0.067

    F_model = candidate_capex_musd / model_money_unit_musd_per_ku
    qual_G_model = (candidate_capex_musd * qualification_cost_share_of_capex) / model_money_unit_musd_per_ku
    S_salv_model = (candidate_capex_musd * salvage_rate) / model_money_unit_musd_per_ku

    horizon = DEFAULT_HORIZON

    plants = {
        "M1": PlantParams(
            plant_id="M1",
            base_type="legacy",
            K_bar=70.92,
            c_mfg=3.3,
            chi_by_regime={1: 1.0, 2: 1.0, 3: 1.0},
            ramp_kappa=0.0,
        ),
        "M2": PlantParams(
            plant_id="M2",
            base_type="candidate",
            K_bar=73.33,
            c_mfg=3,
            chi_by_regime={1: 1.0, 2: 1.0, 3: 1.0},
            ramp_kappa=0.192,
        ),
    }

    suppliers = {
        "S1": SupplierParams("S1", W_bar=40.0, c_mat_base=2.74, delta_rel=0.48),
        "S2": SupplierParams("S2", W_bar=40.0, c_mat_base=2.63, delta_rel=0.46),
        "S3": SupplierParams("S3", W_bar=40.0, c_mat_base=2.75, delta_rel=0.48),
        "S4": SupplierParams("S4", W_bar=40.0, c_mat_base=2.43, delta_rel=0.43),
    }

    # Stronger but still feasible monthly demand cycles.
    # Peak total demand remains below M1 standalone monthly capacity.
    d1_cycle = [0.90, 0.94, 0.98, 1.03, 1.08, 1.05, 1.00, 0.96, 0.92, 0.90, 0.95, 1.00]
    d2_cycle = [0.85, 0.90, 0.96, 1.04, 1.12, 1.08, 1.01, 0.95, 0.89, 0.85, 0.92, 1.00]
    d1_series = _monthly_baseline_series(mean_level=43.5, cycle=d1_cycle, horizon=horizon)
    d2_series = _monthly_baseline_series(mean_level=20.0, cycle=d2_cycle, horizon=horizon)

    markets = {
        "D1": MarketParams("D1", d0=d1_series, cu=15.0, price=30.0),
        "D2": MarketParams("D2", d0=d2_series, cu=10.0, price=28.0),
    }

    tariff_demand_curve = _load_tariff_demand_curve()

    c_in = {
        "S1": {"M1": 0.6, "M2": 0.8},
        "S2": {"M1": 0.5, "M2": 0.7},
        "S3": {"M1": 0.7, "M2": 0.9},
        "S4": {"M1": 0.4, "M2": 0.6},
    }

    # Keep the static logic clean:
    # - M1 remains intrinsically better on D1 before tariff wedges
    # - M2 has a clear logistics advantage on D2
    c_out = {
        "M1": {"D1": 1.0, "D2": 1.5},
        "M2": {"D1": 2.0, "D2": 1.8},
    }

    tau = {
        "M1": {
            "D1": {1: 0.10, 2: 0.90, 3: 0.25},
            "D2": {1: 0.05, 2: 0.12, 3: 0.08},
        },
        "M2": {
            "D1": {1: 0.05, 2: 0.05, 3: 0.12},
            "D2": {1: 0.04, 2: 0.08, 3: 0.12},
        },
    }

    reconfig = ReconfigParams(
        activation_plant="M2",
        F=F_model,
        qual_G=qual_G_model,
        qual_ell=12,
        withdrawal_plant="M1",
        S_salv=S_salv_model,
        terminal_tail_horizon=8,
        terminal_tail_weight=1.0,
    )

    pickup_routes = {
        "M1": [
            RouteParams(
                route_id="M1_pick_multi",
                visits=["S1", "S2", "S3"],
                dist_km=120.0,
                vehicle_cap=300.0,
                cost_per_km=0.02,
            ),
        ],
        "M2": [
            RouteParams(
                route_id="M2_pick_multi",
                visits=["S2", "S4"],
                dist_km=160.0,
                vehicle_cap=300.0,
                cost_per_km=0.02,
            ),
        ],
    }

    delivery_routes = {
        "M1": [
            RouteParams(
                route_id="M1_del_multi",
                visits=["D1", "D2"],
                dist_km=900.0,
                vehicle_cap=220.0,
                cost_per_km=0.03,
            ),
        ],
        "M2": [
            RouteParams(
                route_id="M2_del_multi",
                visits=["D1", "D2"],
                dist_km=850.0,
                vehicle_cap=220.0,
                cost_per_km=0.03,
            ),
        ],
    }

    scenario = SupplyChainScenario(
        H=horizon,
        gamma=MONTHLY_GAMMA,
        suppliers=suppliers,
        plants=plants,
        markets=markets,
        legacy_plant_id="M1",
        candidate_plant_id="M2",
        core_market_id="D1",
        additional_market_id="D2",
        regimes=[1, 2, 3],
        P=[
            [0.55, 0.40, 0.05],
            [0.02, 0.97, 0.01],
            [0.20, 0.20, 0.60],
        ],
        xi_1=1,
        xi_2_forced=2,
        activation_lag=1,
        initial_a={"M1": 1, "M2": 0},
        initial_age=0,
        alpha_by_plant={"M1": 1.0, "M2": 1.0},
        c_in=c_in,
        c_out=c_out,
        tau=tau,
        H_rel=30,
        reconfig=reconfig,
        pickup_routes=pickup_routes,
        delivery_routes=delivery_routes,
        ramp_full_age=12,
        ramp_full_threshold=0.90,
        tariff_demand_curve=tariff_demand_curve,
    )

    if validate:
        scenario.validate_assumptions()
    return scenario


if __name__ == "__main__":
    sc = build_story_case()
    print(sc.summary())
    sc.print_diagnose()
    print("ok: all story and Chapter 3.1 checks passed")
