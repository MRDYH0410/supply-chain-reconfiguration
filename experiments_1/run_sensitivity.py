from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Dict, List

from experiments_1.sensitivity_plots import (
    ActionTracePlotSpec,
    CapacityTracePlotSpec,
    DemandDashboardPlotSpec,
    FixedScenarioPlotSpec,
    plot_action_trace,
    plot_capacity_trace,
    plot_demand_dashboard,
    plot_fixed_baseline_dashboard,
)
from experiments_1.sensitivity_runner import (
    TrainEvalConfig,
    evaluate_three_strategies,
    get_fixed_tariff_schedule,
    make_fixed_baseline_scenario,
)


PRINT_TIMELINE_TRACE = True


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cb_get(info: dict, key: str, default: float = 0.0) -> float:
    cb = info.get("cost_breakdown", {}) or {}
    return float(cb.get(key, default))


def _print_model_timeline_trace(H: int, traces: Dict[str, List[dict]]) -> None:
    A = traces.get("A", [])
    B = traces.get("B", [])
    C = traces.get("C", [])
    print("\nDeterministic timeline under the fixed 30-period tariff scenario")
    print("  t |   A_cost |   B_cost |   C_cost | B_u | C_u | C_v | B_aM2 | C_aM2 | B_capM1 | B_capM2 | B_yM1 | B_yM2")
    print("----+----------+----------+----------+-----+-----+-----+-------+-------+---------+---------+-------+------")
    for t in range(H):
        ia = A[t] if t < len(A) else {}
        ib = B[t] if t < len(B) else {}
        ic = C[t] if t < len(C) else {}
        ca = _cb_get(ia, "C_total")
        cb = _cb_get(ib, "C_total")
        cc = _cb_get(ic, "C_total")
        bu = int(ib.get("u_t", 0))
        cu = int(ic.get("u_t", 0))
        cv = int(ic.get("v_t", 0))
        ba = int(ib.get("a_M2_pre", ib.get("a_M2", 0)))
        ca2 = int(ic.get("a_M2_pre", ic.get("a_M2", 0)))
        bcap1 = float(ib.get("cap_M1_pre", 0.0))
        bcap2 = float(ib.get("cap_M2_pre", 0.0))
        by1 = float(ib.get("y_M1", 0.0))
        by2 = float(ib.get("y_M2", 0.0))
        print(
            f"{t + 1:>3d} | {ca:>8.1f} | {cb:>8.1f} | {cc:>8.1f} | {bu:>3d} | {cu:>3d} | {cv:>3d} | "
            f"{ba:>5d} | {ca2:>5d} | {bcap1:>7.1f} | {bcap2:>7.1f} | {by1:>5.1f} | {by2:>4.1f}"
        )


def _winner(costA: float, costB: float, costC: float) -> str:
    vals = {"A": costA, "B": costB, "C": costC}
    return min(vals, key=vals.get)


def _write_summary_csv(path: str, stats: Dict[str, any]) -> None:
    fieldnames = [
        "strategy",
        "mean_discounted_total_cost",
        "std_discounted_total_cost",
        "mean_total_cost",
        "std_total_cost",
        "mean_total_demand",
        "std_total_demand",
        "activation_periods",
        "withdrawal_periods",
        "C_in",
        "C_out",
        "C_fix",
        "C_qual",
        "C_loss",
        "Salvage",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in ["A", "B", "C"]:
            s = stats[name]
            writer.writerow({
                "strategy": name,
                "mean_discounted_total_cost": float(s.mean_discounted_total_cost),
                "std_discounted_total_cost": float(s.std_discounted_total_cost),
                "mean_total_cost": float(s.mean_total_cost),
                "std_total_cost": float(s.std_total_cost),
                "mean_total_demand": float(s.mean_total_demand),
                "std_total_demand": float(s.std_total_demand),
                "activation_periods": ";".join(str(int(v)) for v in s.activation_periods),
                "withdrawal_periods": ";".join(str(int(v)) for v in s.withdrawal_periods),
                "C_in": float(s.mean_breakdown.get("C_in", 0.0)),
                "C_out": float(s.mean_breakdown.get("C_out", 0.0)),
                "C_fix": float(s.mean_breakdown.get("C_fix", 0.0)),
                "C_qual": float(s.mean_breakdown.get("C_qual", 0.0)),
                "C_loss": float(s.mean_breakdown.get("C_loss", 0.0)),
                "Salvage": float(s.mean_breakdown.get("Salvage", 0.0)),
            })


def _write_period_trace_csv(path: str, periods: List[int], tariff_schedule: Dict[str, Dict[str, List[float]]], traces: Dict[str, List[dict]], stats: Dict[str, any]) -> None:
    jo = list(tariff_schedule.keys())[0]
    jn = list(tariff_schedule.keys())[1]
    kc = list(tariff_schedule[jo].keys())[0]
    ka = list(tariff_schedule[jo].keys())[1]

    fieldnames = [
        "t",
        "tau_jo_kc", "tau_jn_kc", "tau_jo_ka", "tau_jn_ka",
        "A_period_cost", "B_period_cost", "C_period_cost",
        "A_cum_discounted", "B_cum_discounted", "C_cum_discounted",
        "A_period_demand", "B_period_demand", "C_period_demand",
        "A_cum_demand", "B_cum_demand", "C_cum_demand",
        "B_u", "C_u", "C_v",
        "B_a_M2_pre", "C_a_M2_pre", "B_age_M2_pre", "C_age_M2_pre",
        "A_cap_M1_pre", "A_cap_M2_pre", "A_y_M1", "A_y_M2",
        "B_cap_M1_pre", "B_cap_M2_pre", "B_y_M1", "B_y_M2",
        "C_cap_M1_pre", "C_cap_M2_pre", "C_y_M1", "C_y_M2",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        A = traces["A"]
        B = traces["B"]
        C = traces["C"]
        for idx, t in enumerate(periods):
            writer.writerow({
                "t": int(t),
                "tau_jo_kc": float(tariff_schedule[jo][kc][idx]),
                "tau_jn_kc": float(tariff_schedule[jn][kc][idx]),
                "tau_jo_ka": float(tariff_schedule[jo][ka][idx]),
                "tau_jn_ka": float(tariff_schedule[jn][ka][idx]),
                "A_period_cost": float(stats["A"].mean_cost_by_t[idx]),
                "B_period_cost": float(stats["B"].mean_cost_by_t[idx]),
                "C_period_cost": float(stats["C"].mean_cost_by_t[idx]),
                "A_cum_discounted": float(stats["A"].mean_cumulative_discounted_cost_by_t[idx]),
                "B_cum_discounted": float(stats["B"].mean_cumulative_discounted_cost_by_t[idx]),
                "C_cum_discounted": float(stats["C"].mean_cumulative_discounted_cost_by_t[idx]),
                "A_period_demand": float(stats["A"].mean_demand_by_t[idx]),
                "B_period_demand": float(stats["B"].mean_demand_by_t[idx]),
                "C_period_demand": float(stats["C"].mean_demand_by_t[idx]),
                "A_cum_demand": float(stats["A"].mean_cumulative_demand_by_t[idx]),
                "B_cum_demand": float(stats["B"].mean_cumulative_demand_by_t[idx]),
                "C_cum_demand": float(stats["C"].mean_cumulative_demand_by_t[idx]),
                "B_u": int(B[idx].get("u_t", 0)),
                "C_u": int(C[idx].get("u_t", 0)),
                "C_v": int(C[idx].get("v_t", 0)),
                "B_a_M2_pre": int(B[idx].get("a_M2_pre", B[idx].get("a_M2", 0))),
                "C_a_M2_pre": int(C[idx].get("a_M2_pre", C[idx].get("a_M2", 0))),
                "B_age_M2_pre": int(B[idx].get("age_M2_pre", B[idx].get("age_M2", 0))),
                "C_age_M2_pre": int(C[idx].get("age_M2_pre", C[idx].get("age_M2", 0))),
                "A_cap_M1_pre": float(A[idx].get("cap_M1_pre", 0.0)),
                "A_cap_M2_pre": float(A[idx].get("cap_M2_pre", 0.0)),
                "A_y_M1": float(A[idx].get("y_M1", 0.0)),
                "A_y_M2": float(A[idx].get("y_M2", 0.0)),
                "B_cap_M1_pre": float(B[idx].get("cap_M1_pre", 0.0)),
                "B_cap_M2_pre": float(B[idx].get("cap_M2_pre", 0.0)),
                "B_y_M1": float(B[idx].get("y_M1", 0.0)),
                "B_y_M2": float(B[idx].get("y_M2", 0.0)),
                "C_cap_M1_pre": float(C[idx].get("cap_M1_pre", 0.0)),
                "C_cap_M2_pre": float(C[idx].get("cap_M2_pre", 0.0)),
                "C_y_M1": float(C[idx].get("y_M1", 0.0)),
                "C_y_M2": float(C[idx].get("y_M2", 0.0)),
            })


def _write_tariff_json(path: str, tariff_schedule: Dict[str, Dict[str, List[float]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tariff_schedule, f, indent=2)


def run_fixed_baseline_experiment(seeds: List[int], cfg: TrainEvalConfig) -> None:
    out_dir = os.path.join("outputs", "experiment_1_fixed_scenario")
    _ensure_dir(out_dir)

    scenario = make_fixed_baseline_scenario(validate=False)
    tariff_schedule = get_fixed_tariff_schedule(scenario)
    stats, traces = evaluate_three_strategies(scenario, seeds=seeds, cfg=cfg, return_traces=True)

    A, B, C = stats["A"], stats["B"], stats["C"]
    best = _winner(A.mean_discounted_total_cost, B.mean_discounted_total_cost, C.mean_discounted_total_cost)

    print("\n=== Experiment 1  fixed single tariff scenario baseline ===")
    print(f"H = {scenario.H}")
    print(f"Seeds = {seeds}")
    print("Comparison metric = cumulative discounted total cost")
    print(
        f"A={A.mean_discounted_total_cost:.1f}, "
        f"B={B.mean_discounted_total_cost:.1f}, "
        f"C={C.mean_discounted_total_cost:.1f} | best={best}"
    )
    print(f"B activation periods = {B.activation_periods}")
    print(f"C activation periods = {C.activation_periods}")
    print(f"C withdrawal periods = {C.withdrawal_periods}")

    if PRINT_TIMELINE_TRACE:
        _print_model_timeline_trace(H=int(scenario.H), traces=traces)

    periods = list(range(1, int(scenario.H) + 1))
    dashboard_spec = FixedScenarioPlotSpec(
        periods=periods,
        tariff_schedule=tariff_schedule,
        mean_period_cost_A=[float(v) for v in A.mean_cost_by_t],
        mean_period_cost_B=[float(v) for v in B.mean_cost_by_t],
        mean_period_cost_C=[float(v) for v in C.mean_cost_by_t],
        mean_cum_discounted_A=[float(v) for v in A.mean_cumulative_discounted_cost_by_t],
        mean_cum_discounted_B=[float(v) for v in B.mean_cumulative_discounted_cost_by_t],
        mean_cum_discounted_C=[float(v) for v in C.mean_cumulative_discounted_cost_by_t],
        std_period_cost_A=[float(v) for v in A.std_cost_by_t],
        std_period_cost_B=[float(v) for v in B.std_cost_by_t],
        std_period_cost_C=[float(v) for v in C.std_cost_by_t],
        std_cum_discounted_A=[float(v) for v in A.std_cumulative_discounted_cost_by_t],
        std_cum_discounted_B=[float(v) for v in B.std_cumulative_discounted_cost_by_t],
        std_cum_discounted_C=[float(v) for v in C.std_cumulative_discounted_cost_by_t],
        total_discounted_A=float(A.mean_discounted_total_cost),
        total_discounted_B=float(B.mean_discounted_total_cost),
        total_discounted_C=float(C.mean_discounted_total_cost),
        activation_period_B=int(B.activation_periods[0]) if B.activation_periods else 0,
        activation_period_C=int(C.activation_periods[0]) if C.activation_periods else 0,
        withdrawal_period_C=int(C.withdrawal_periods[0]) if C.withdrawal_periods else 0,
    )
    demand_spec = DemandDashboardPlotSpec(
        periods=periods,
        tariff_schedule=tariff_schedule,
        mean_period_demand_A=[float(v) for v in A.mean_demand_by_t],
        mean_period_demand_B=[float(v) for v in B.mean_demand_by_t],
        mean_period_demand_C=[float(v) for v in C.mean_demand_by_t],
        mean_cum_demand_A=[float(v) for v in A.mean_cumulative_demand_by_t],
        mean_cum_demand_B=[float(v) for v in B.mean_cumulative_demand_by_t],
        mean_cum_demand_C=[float(v) for v in C.mean_cumulative_demand_by_t],
        std_period_demand_A=[float(v) for v in A.std_demand_by_t],
        std_period_demand_B=[float(v) for v in B.std_demand_by_t],
        std_period_demand_C=[float(v) for v in C.std_demand_by_t],
        std_cum_demand_A=[float(v) for v in A.std_cumulative_demand_by_t],
        std_cum_demand_B=[float(v) for v in B.std_cumulative_demand_by_t],
        std_cum_demand_C=[float(v) for v in C.std_cumulative_demand_by_t],
        total_demand_A=float(A.mean_total_demand),
        total_demand_B=float(B.mean_total_demand),
        total_demand_C=float(C.mean_total_demand),
        activation_period_B=int(B.activation_periods[0]) if B.activation_periods else 0,
        activation_period_C=int(C.activation_periods[0]) if C.activation_periods else 0,
        withdrawal_period_C=int(C.withdrawal_periods[0]) if C.withdrawal_periods else 0,
    )

    action_spec = ActionTracePlotSpec(
        periods=periods,
        u_B=[float(info.get("u_t", 0.0)) for info in traces["B"]],
        u_C=[float(info.get("u_t", 0.0)) for info in traces["C"]],
        v_C=[float(info.get("v_t", 0.0)) for info in traces["C"]],
        a_M2_B=[float(info.get("a_M2_pre", info.get("a_M2", 0.0))) for info in traces["B"]],
        a_M2_C=[float(info.get("a_M2_pre", info.get("a_M2", 0.0))) for info in traces["C"]],
        age_M2_B=[float(info.get("age_M2_pre", info.get("age_M2", 0.0))) for info in traces["B"]],
        age_M2_C=[float(info.get("age_M2_pre", info.get("age_M2", 0.0))) for info in traces["C"]],
    )
    capacity_spec = CapacityTracePlotSpec(
        periods=periods,
        cap_M1_A=[float(info.get("cap_M1_pre", 0.0)) for info in traces["A"]],
        cap_M2_A=[float(info.get("cap_M2_pre", 0.0)) for info in traces["A"]],
        y_M1_A=[float(info.get("y_M1", 0.0)) for info in traces["A"]],
        y_M2_A=[float(info.get("y_M2", 0.0)) for info in traces["A"]],
        cap_M1_B=[float(info.get("cap_M1_pre", 0.0)) for info in traces["B"]],
        cap_M2_B=[float(info.get("cap_M2_pre", 0.0)) for info in traces["B"]],
        y_M1_B=[float(info.get("y_M1", 0.0)) for info in traces["B"]],
        y_M2_B=[float(info.get("y_M2", 0.0)) for info in traces["B"]],
        cap_M1_C=[float(info.get("cap_M1_pre", 0.0)) for info in traces["C"]],
        cap_M2_C=[float(info.get("cap_M2_pre", 0.0)) for info in traces["C"]],
        y_M1_C=[float(info.get("y_M1", 0.0)) for info in traces["C"]],
        y_M2_C=[float(info.get("y_M2", 0.0)) for info in traces["C"]],
    )

    dashboard_png = os.path.join(out_dir, "fig_exp1_fixed_scenario_dashboard.png")
    dashboard_pdf = os.path.join(out_dir, "fig_exp1_fixed_scenario_dashboard.pdf")
    demand_png = os.path.join(out_dir, "fig_exp1_fixed_scenario_demand.png")
    demand_pdf = os.path.join(out_dir, "fig_exp1_fixed_scenario_demand.pdf")
    action_png = os.path.join(out_dir, "fig_exp1_fixed_scenario_actions.png")
    action_pdf = os.path.join(out_dir, "fig_exp1_fixed_scenario_actions.pdf")
    capacity_png = os.path.join(out_dir, "fig_exp1_fixed_scenario_capacity.png")
    capacity_pdf = os.path.join(out_dir, "fig_exp1_fixed_scenario_capacity.pdf")
    summary_csv = os.path.join(out_dir, "exp1_strategy_summary.csv")
    trace_csv = os.path.join(out_dir, "exp1_period_trace.csv")
    tariff_json = os.path.join(out_dir, "exp1_fixed_tariff_schedule.json")

    plot_fixed_baseline_dashboard(dashboard_spec, dashboard_png, dashboard_pdf)
    plot_demand_dashboard(demand_spec, demand_png, demand_pdf)
    plot_action_trace(action_spec, action_png, action_pdf)
    plot_capacity_trace(capacity_spec, capacity_png, capacity_pdf)
    _write_summary_csv(summary_csv, stats)
    _write_period_trace_csv(trace_csv, periods, tariff_schedule, traces, stats)
    _write_tariff_json(tariff_json, tariff_schedule)

    print(f"saved dashboard     -> {dashboard_png}")
    print(f"saved dashboard     -> {dashboard_pdf}")
    print(f"saved demand plot   -> {demand_png}")
    print(f"saved demand plot   -> {demand_pdf}")
    print(f"saved action plot   -> {action_png}")
    print(f"saved action plot   -> {action_pdf}")
    print(f"saved capacity plot -> {capacity_png}")
    print(f"saved capacity plot -> {capacity_pdf}")
    print(f"saved summary csv   -> {summary_csv}")
    print(f"saved trace csv     -> {trace_csv}")
    print(f"saved tariff json   -> {tariff_json}")


def main() -> None:
    seeds = [0]
    cfg = TrainEvalConfig(
        hidden=32,
        device="cpu",
        iterations=10,
        episodes_per_iter=10,
        eval_episode_seed=20260325,
    )
    run_fixed_baseline_experiment(seeds=seeds, cfg=cfg)


if __name__ == "__main__":
    main()