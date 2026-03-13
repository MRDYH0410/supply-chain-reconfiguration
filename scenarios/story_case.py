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
    # ------------------------------------------------------------------
    # Quantity and time units
    # ------------------------------------------------------------------
    # 1 model period = 1 quarter
    # Quantity unit = thousand vehicles per quarter
    #
    # Horizon:
    #   H = 20 quarters = 5 years
    #
    # Discount factor:
    #   quarterly gamma = 0.989
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Real-data anchors used for the compact-scale calibration
    # ------------------------------------------------------------------
    # 1) Legacy plant quarterly capacity
    #    Tesla Shanghai 2025 deliveries ≈ 851,000 vehicles
    #    => quarterly average ≈ 212,750 vehicles
    #    => K_bar(M1) = 212.75 in thousand vehicles per quarter
    #
    # 2) Unit automotive cost decomposition
    #    Shanghai 2025 automotive cost proxy ≈ 28,551 USD / vehicle
    #    material share midpoint = 72.5%
    #    => material cost ≈ 20,699.475 USD / vehicle
    #    => conversion cost ≈ 7,851.525 USD / vehicle
    #
    # 3) Compact money scale
    #    Keep c_mfg(M1)=4.0 as the conversion-cost anchor
    #    => 1 model money unit ≈ 7,851.525 / 4 = 1,962.88125 USD / vehicle
    #    => equivalently 1 model money unit ≈ 1.96288125 million USD
    #       per thousand vehicles
    #
    # 4) Material pool under the compact scale
    #    total material cost in model units
    #    = 20,699.475 / 1,962.88125 ≈ 10.545
    #
    # 5) Early relationship premium
    #    empirical range 15%–20%
    #    here we keep the additive model form and approximate a 17.5% uplift
    #
    # 6) Relationship horizon
    #    H_rel = 10 quarters = 30 months
    #
    # 7) Qualification horizon
    #    qual_ell = 4 quarters = 12 months
    #
    # 8) Candidate plant investment
    #    capex proxy ≈ 5,000 million USD
    #
    # 9) Qualification fixed cost
    #    no direct amount given yet
    #    use 1% of capex as a compact baseline proxy
    #
    # 10) Salvage
    #     salvage rate ≈ 6.7%
    #     use the same 5,000 million USD asset-value proxy for now
    #
    # 11) Demand anchor
    #     2025 Q1 domestic sales ≈ 137.2 thousand vehicles
    #     use that to anchor D1 baseline demand
    # ------------------------------------------------------------------

    automotive_cost_per_vehicle_usd = 28_551.0
    material_share_mid = 0.725
    conversion_share_mid = 1.0 - material_share_mid

    conversion_cost_per_vehicle_usd = automotive_cost_per_vehicle_usd * conversion_share_mid
    material_cost_per_vehicle_usd = automotive_cost_per_vehicle_usd * material_share_mid

    model_money_unit_usd_per_vehicle = conversion_cost_per_vehicle_usd / 4.0
    model_money_unit_musd_per_ku = model_money_unit_usd_per_vehicle / 1000.0

    candidate_capex_musd = 5_000.0
    qualification_cost_share_of_capex = 0.01
    salvage_rate = 0.067

    F_model = candidate_capex_musd / model_money_unit_musd_per_ku
    qual_G_model = (candidate_capex_musd * qualification_cost_share_of_capex) / model_money_unit_musd_per_ku
    S_salv_model = (candidate_capex_musd * salvage_rate) / model_money_unit_musd_per_ku

    # ------------------------------------------------------------------
    # plants
    # ------------------------------------------------------------------
    plants = {
        "M1": PlantParams(
            plant_id="M1",
            base_type="legacy",
            K_bar=212.75,
            c_mfg=4.0,
            chi_by_regime={1: 1.0, 2: 1.0, 3: 1.0},
            ramp_kappa=0.0,
        ),
        "M2": PlantParams(
            plant_id="M2",
            base_type="candidate",
            K_bar=220.0,
            # no direct plant-specific real manufacturing-cost anchor for M2 yet
            # keep the previous compact premium relative to M1
            c_mfg=5.0,
            chi_by_regime={1: 1.0, 2: 1.0, 3: 1.0},
            # age 4 quarters should reach about 90% effective capacity
            # kappa ≈ ln(10) / 4 ≈ 0.576
            ramp_kappa=0.576,
        ),
    }

    # ------------------------------------------------------------------
    # suppliers
    # ------------------------------------------------------------------
    # total c_mat_base target ≈ 10.545 model units
    # choose a mild heterogeneous split across four suppliers
    # additive delta_rel values approximate a 17.5% early-stage uplift
    suppliers = {
        "S1": SupplierParams("S1", W_bar=120.0, c_mat_base=2.74, delta_rel=0.48),
        "S2": SupplierParams("S2", W_bar=120.0, c_mat_base=2.63, delta_rel=0.46),
        "S3": SupplierParams("S3", W_bar=120.0, c_mat_base=2.75, delta_rel=0.48),
        "S4": SupplierParams("S4", W_bar=120.0, c_mat_base=2.43, delta_rel=0.43),
    }

    # ------------------------------------------------------------------
    # markets
    # ------------------------------------------------------------------
    markets = {
        "D1": MarketParams("D1", d0=137.2, eta=0.2, cu=15.0, price=30.0, demand_floor=0.90),
        "D2": MarketParams("D2", d0=60.0, eta=0.1, cu=10.0, price=28.0, demand_floor=0.92),
    }

    # ------------------------------------------------------------------
    # external tariff-related demand shock estimated from paras/traiff_demand.py
    # conservative version
    # ------------------------------------------------------------------
    tariff_demand_shock_cycle = [
        0.711912,  # FQ1
        0.854261,  # FQ2
        1.000000,  # FQ3
        0.805200,  # FQ4
    ]

    # ------------------------------------------------------------------
    # inbound and outbound logistics costs
    # ------------------------------------------------------------------
    # No direct route-specific real logistics data has been passed in yet.
    # Keep these on the compact model scale for now.
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

    # ------------------------------------------------------------------
    # tariffs
    # ------------------------------------------------------------------
    # tau is dimensionless and should not be money-scaled
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

    # ------------------------------------------------------------------
    # reconfiguration costs
    # ------------------------------------------------------------------
    reconfig = ReconfigParams(
        activation_plant="M2",
        F=F_model,
        qual_G=qual_G_model,
        qual_ell=4,
        withdrawal_plant="M1",
        S_salv=S_salv_model,
    )

    # ------------------------------------------------------------------
    # route-based tours
    # ------------------------------------------------------------------
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
        gamma=0.989,
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
        tariff_demand_shock_cycle=tariff_demand_shock_cycle,
    )

    if validate:
        scenario.validate_assumptions()
    return scenario


if __name__ == "__main__":
    sc = build_story_case()
    print(sc.summary())
    sc.print_diagnose()
    print("ok: all story and Chapter 3.1 checks passed")