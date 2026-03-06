from __future__ import annotations

import csv
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Dict, List

from experiments.sensitivity_plots import (
    LinePlotSpec,
    plot_all_tariff_path_lineplots,
    plot_tariff_path_contact_sheet,
)
from experiments.sensitivity_runner import (
    TrainEvalConfig,
    TariffLevelConfig,
    build_tariff_path_specs_from_phase_structure,
    detect_phase_structure_via_strategy_b,
    evaluate_three_strategies,
    extract_tariff_level_values,
    joint_state_full_label,
    make_scenario_base,
    make_tariff_path_scenario,
)


PRINT_TIMELINE_TRACE = False
TRACE_EPISODE_SEED = 20260306


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cb_get(info: dict, key: str, default: float = 0.0) -> float:
    cb = info.get("cost_breakdown", {}) or {}
    return float(cb.get(key, default))


def _print_model_timeline_trace(H: int, traces: Dict[str, List[dict]], episode_seed: int) -> None:
    A = traces.get("A", [])
    B = traces.get("B", [])
    C = traces.get("C", [])
    print(f"    timeline trace (single episode_seed={episode_seed}, t=1..{H}):")
    print("      t | xi |     A_cost |     B_cost |     C_cost")
    print("    ----+----+------------+------------+------------")
    for t in range(H):
        ia = A[t] if t < len(A) else {}
        ib = B[t] if t < len(B) else {}
        ic = C[t] if t < len(C) else {}
        xi = int(ia.get("xi_t", ib.get("xi_t", ic.get("xi_t", 0))))
        ca = _cb_get(ia, "C_total")
        cb = _cb_get(ib, "C_total")
        cc = _cb_get(ic, "C_total")
        print(f"    {t + 1:>4d} | {xi:>2d} | {ca:>10.1f} | {cb:>10.1f} | {cc:>10.1f}")


def _winner(costA: float, costB: float, costC: float) -> str:
    vals = {"A": costA, "B": costB, "C": costC}
    return min(vals, key=vals.get)


def _write_results_csv(path: str, rows: List[dict]) -> None:
    fieldnames = [
        "path_id", "path_label", "phase1_state", "phase2_state", "phase3_state",
        "phase2_label", "phase3_label", "best_strategy",
        "mean_cost_A", "std_cost_A", "mean_cost_B", "std_cost_B", "mean_cost_C", "std_cost_C",
        "gap_B_minus_A", "gap_C_minus_A", "gap_C_minus_B",
        "phase1_start", "phase1_end", "phase2_start", "phase2_end", "phase3_start", "phase3_end",
        "change_period_1", "change_period_2",
        "A_C_in", "A_C_out", "A_C_fix", "A_C_qual", "A_C_loss", "A_Salvage",
        "B_C_in", "B_C_out", "B_C_fix", "B_C_qual", "B_C_loss", "B_Salvage",
        "C_C_in", "C_C_out", "C_C_fix", "C_C_qual", "C_C_loss", "C_Salvage",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_phase_and_tariff_summary(path: str, phase_info: dict, tariff_levels: dict) -> None:
    payload = {
        "phase_structure": phase_info,
        "tariff_levels": tariff_levels,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_tariff_path_sensitivity(seeds: List[int], cfg: TrainEvalConfig) -> None:
    _ensure_dir("outputs")

    level_cfg = TariffLevelConfig(
        original_side_low_regime=1,
        original_side_high_regime=2,
        new_side_low_regime=1,
        new_side_high_regime=3,
    )

    base = make_scenario_base(validate=False)
    tariff_levels = extract_tariff_level_values(base, level_cfg)
    phase_structure = detect_phase_structure_via_strategy_b(seeds=seeds, cfg=cfg, level_cfg=level_cfg)
    specs = build_tariff_path_specs_from_phase_structure(phase_structure=phase_structure, horizon=int(base.H))

    print("\n=== Running updated tariff uncertainty sensitivity ===")
    print("Step 1. Use Strategy B to detect the structural phase split")
    print(f"  activation period for Phase II start = t{phase_structure.phase2_start}")
    print(f"  ramp-full period for Phase III start = t{phase_structure.phase3_start}")
    print(f"  tariff update 1 at midpoint of Phase II = t{phase_structure.change_period_1}")
    print(f"  tariff update 2 at midpoint of Phase III = t{phase_structure.change_period_2}")
    print(f"  Phase I periods   = {phase_structure.phase1_periods}")
    print(f"  Phase II periods  = {phase_structure.phase2_periods}")
    print(f"  Phase III periods = {phase_structure.phase3_periods}")
    print("Step 2. For each of the 16 tariff combinations, compare A, B, and C under the same tariff path")
    print(f"Seeds = {seeds}")

    print("\nTariff level definition used in this sensitivity analysis:")
    for side, side_levels in tariff_levels.items():
        print(f"  {side}:")
        for level_name, vals in side_levels.items():
            print(f"    {level_name}: {vals}")

    csv_rows: List[dict] = []
    plot_specs: List[LinePlotSpec] = []

    for spec in specs:
        scenario = make_tariff_path_scenario(spec, validate=False, level_cfg=level_cfg)
        stats, traces = evaluate_three_strategies(
            scenario,
            seeds=seeds,
            cfg=cfg,
            return_traces=True,
            trace_episode_seed=TRACE_EPISODE_SEED,
        )
        A, B, C = stats["A"], stats["B"], stats["C"]

        best = _winner(A.mean_total_cost, B.mean_total_cost, C.mean_total_cost)
        print(
            f"  {spec.path_id} | {spec.path_label} | "
            f"A={A.mean_total_cost:.1f}, B={B.mean_total_cost:.1f}, C={C.mean_total_cost:.1f} | best={best}"
        )

        if PRINT_TIMELINE_TRACE:
            _print_model_timeline_trace(H=int(scenario.H), traces=traces, episode_seed=TRACE_EPISODE_SEED)

        csv_rows.append({
            "path_id": spec.path_id,
            "path_label": spec.path_label,
            "phase1_state": spec.phase1_state,
            "phase2_state": spec.phase2_state,
            "phase3_state": spec.phase3_state,
            "phase2_label": joint_state_full_label(spec.phase2_state),
            "phase3_label": joint_state_full_label(spec.phase3_state),
            "best_strategy": best,
            "mean_cost_A": float(A.mean_total_cost),
            "std_cost_A": float(A.std_total_cost),
            "mean_cost_B": float(B.mean_total_cost),
            "std_cost_B": float(B.std_total_cost),
            "mean_cost_C": float(C.mean_total_cost),
            "std_cost_C": float(C.std_total_cost),
            "gap_B_minus_A": float(B.mean_total_cost - A.mean_total_cost),
            "gap_C_minus_A": float(C.mean_total_cost - A.mean_total_cost),
            "gap_C_minus_B": float(C.mean_total_cost - B.mean_total_cost),
            "phase1_start": int(spec.phase1_periods[0]),
            "phase1_end": int(spec.phase1_periods[1]),
            "phase2_start": int(spec.phase2_periods[0]),
            "phase2_end": int(spec.phase2_periods[1]),
            "phase3_start": int(spec.phase3_periods[0]),
            "phase3_end": int(spec.phase3_periods[1]),
            "change_period_1": int(spec.change_period_1),
            "change_period_2": int(spec.change_period_2),
            "A_C_in": A.mean_breakdown.get("C_in", 0.0),
            "A_C_out": A.mean_breakdown.get("C_out", 0.0),
            "A_C_fix": A.mean_breakdown.get("C_fix", 0.0),
            "A_C_qual": A.mean_breakdown.get("C_qual", 0.0),
            "A_C_loss": A.mean_breakdown.get("C_loss", 0.0),
            "A_Salvage": A.mean_breakdown.get("Salvage", 0.0),
            "B_C_in": B.mean_breakdown.get("C_in", 0.0),
            "B_C_out": B.mean_breakdown.get("C_out", 0.0),
            "B_C_fix": B.mean_breakdown.get("C_fix", 0.0),
            "B_C_qual": B.mean_breakdown.get("C_qual", 0.0),
            "B_C_loss": B.mean_breakdown.get("C_loss", 0.0),
            "B_Salvage": B.mean_breakdown.get("Salvage", 0.0),
            "C_C_in": C.mean_breakdown.get("C_in", 0.0),
            "C_C_out": C.mean_breakdown.get("C_out", 0.0),
            "C_C_fix": C.mean_breakdown.get("C_fix", 0.0),
            "C_C_qual": C.mean_breakdown.get("C_qual", 0.0),
            "C_C_loss": C.mean_breakdown.get("C_loss", 0.0),
            "C_Salvage": C.mean_breakdown.get("Salvage", 0.0),
        })

        plot_specs.append(
            LinePlotSpec(
                path_id=spec.path_id,
                path_label=spec.path_label,
                meanA=[float(v) for v in A.mean_cost_by_t],
                meanB=[float(v) for v in B.mean_cost_by_t],
                meanC=[float(v) for v in C.mean_cost_by_t],
                stdA=[float(v) for v in A.std_cost_by_t],
                stdB=[float(v) for v in B.std_cost_by_t],
                stdC=[float(v) for v in C.std_cost_by_t],
                totalA=float(A.mean_total_cost),
                totalB=float(B.mean_total_cost),
                totalC=float(C.mean_total_cost),
                phase1_periods=spec.phase1_periods,
                phase2_periods=spec.phase2_periods,
                phase3_periods=spec.phase3_periods,
                change_period_1=int(spec.change_period_1),
                change_period_2=int(spec.change_period_2),
            )
        )

    csv_path = os.path.join("outputs", "tariff_path_sensitivity_16paths_summary.csv")
    json_path = os.path.join("outputs", "tariff_phase_structure_and_levels.json")
    lineplot_dir = os.path.join("outputs", "tariff_path_lineplots")
    contact_png = os.path.join("outputs", "tariff_path_lineplots_contact_sheet.png")
    contact_pdf = os.path.join("outputs", "tariff_path_lineplots_contact_sheet.pdf")

    _write_results_csv(csv_path, csv_rows)
    _write_phase_and_tariff_summary(
        path=json_path,
        phase_info={
            "phase1_periods": list(phase_structure.phase1_periods),
            "phase2_periods": list(phase_structure.phase2_periods),
            "phase3_periods": list(phase_structure.phase3_periods),
            "phase2_start": int(phase_structure.phase2_start),
            "phase3_start": int(phase_structure.phase3_start),
            "change_period_1": int(phase_structure.change_period_1),
            "change_period_2": int(phase_structure.change_period_2),
            "activation_period_B": int(phase_structure.activation_period_B),
            "ramp_full_period_B": int(phase_structure.ramp_full_period_B),
        },
        tariff_levels=tariff_levels,
    )
    plot_all_tariff_path_lineplots(specs=plot_specs, output_dir=lineplot_dir)
    plot_tariff_path_contact_sheet(specs=plot_specs, out_png=contact_png, out_pdf=contact_pdf)

    print(f"saved csv           -> {csv_path}")
    print(f"saved phase config  -> {json_path}")
    print(f"saved 16 lineplots  -> {lineplot_dir}")
    print(f"saved contact sheet -> {contact_png}")
    print(f"saved contact sheet -> {contact_pdf}")


def main() -> None:
    seeds = [0]
    cfg = TrainEvalConfig(
        hidden=32,
        device="cpu",
        iterations=8,
        episodes_per_iter=8,
        eval_episodes=4,
        phase_detection_eval_episodes=8,
    )
    run_tariff_path_sensitivity(seeds=seeds, cfg=cfg)


if __name__ == "__main__":
    main()