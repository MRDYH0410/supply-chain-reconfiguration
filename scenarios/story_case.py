from __future__ import annotations

from .scenario import (
    SupplyChainScenario,
    PlantParams,
    SupplierParams,
    MarketParams,
    ReconfigParams,
    RouteParams,
)


def build_story_case(validate: bool = True) -> SupplyChainScenario:
    # ---------- plants ----------
    plants = {
        "M1": PlantParams(
            plant_id="M1",
            base_type="legacy",
            K_bar=220.0,
            c_mfg=4.0,
            chi_by_regime={1: 1.0, 2: 1.0, 3: 1.0},
            ramp_kappa=0.0,
        ),
        "M2": PlantParams(
            plant_id="M2",
            base_type="candidate",
            K_bar=220.0,
            c_mfg=5.0,
            chi_by_regime={1: 1.0, 2: 1.0, 3: 1.0},
            # choose kappa so that rho(4) >= 0.9  -> kappa >= ln(10)/4 ≈ 0.576
            ramp_kappa=0.7,
        ),
    }

    # ---------- suppliers ----------
    suppliers = {
        "S1": SupplierParams("S1", W_bar=120.0, c_mat_base=1.2, delta_rel=0.10),
        "S2": SupplierParams("S2", W_bar=120.0, c_mat_base=1.1, delta_rel=0.10),
        "S3": SupplierParams("S3", W_bar=120.0, c_mat_base=1.3, delta_rel=0.10),
        "S4": SupplierParams("S4", W_bar=120.0, c_mat_base=1.0, delta_rel=0.10),
    }

    # ---------- markets ----------
    markets = {
        "D1": MarketParams("D1", d0=140.0, eta=0.2, cu=15.0, price=30.0, demand_floor=0.90),
        "D2": MarketParams("D2", d0=80.0, eta=0.1, cu=10.0, price=28.0, demand_floor=0.92),
    }

    # ---------- inbound / outbound costs ----------
    c_in = {
        "S1": {"M1": 0.6, "M2": 0.8},
        "S2": {"M1": 0.5, "M2": 0.7},
        "S3": {"M1": 0.7, "M2": 0.9},
        "S4": {"M1": 0.4, "M2": 0.6},
    }

    c_out = {
        "M1": {"D1": 1.0, "D2": 2.5},
        "M2": {"D1": 1.8, "D2": 1.9},
    }

    # ---------- tariffs on four edges ----------
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

    # ---------- reconfiguration costs ----------
    reconfig = ReconfigParams(
        activation_plant="M2",
        F=300.0,
        qual_G=30.0,
        qual_ell=3,
        withdrawal_plant="M1",
        S_salv=1500.0,
    )

    # ---------- route-based tours ----------
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
        H=20,
        gamma=1.0,
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
        H_rel=10,
        reconfig=reconfig,
        pickup_routes=pickup_routes,
        delivery_routes=delivery_routes,
        ramp_full_age=4,
        ramp_full_threshold=0.90,
    )

    if validate:
        scenario.validate_assumptions()
    return scenario


if __name__ == "__main__":
    sc = build_story_case()
    print(sc.summary())
    sc.print_diagnose()
    print("ok: all story and Chapter 3.1 checks passed")