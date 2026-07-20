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

from experiments_1.sensitivity_runner import TrainEvalConfig
from experiments_2.sensitivity_plots import (
    plot_activation_heatmaps,
    plot_gap_heatmaps,
    plot_profile_lines,
    plot_total_cost_heatmaps,
)
from experiments_2.sensitivity_runner import (
    LEVEL_SHIFTS,
    SPAN_LEVELS,
    TAIL_START_PERIOD,
    EXP2_LOCAL_M2_D1_COUT,
    EXP2_LOCAL_M2_RAMP_KAPPA,
    EXP2_LOCAL_FIXED_COST_MULT,
    EXP2_LOCAL_QUAL_COST_MULT,
    evaluate_exp2_grid,
    results_to_rows,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_results_csv(path: str, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_grid_spec(path: str) -> None:
    payload = {
        "tail_start_period": TAIL_START_PERIOD,
        "level_shifts": LEVEL_SHIFTS,
        "span_levels": SPAN_LEVELS,
        "interpretation": {
            "level_shift": "Applied to the period-15 anchor of the M1->D1 tariff for periods 16-80.",
            "span": "Deterministic within-tail oscillation range around the shifted baseline.",
            "exp2_local_override": {
                "M2_D1_c_out": EXP2_LOCAL_M2_D1_COUT,
                "M2_ramp_kappa": EXP2_LOCAL_M2_RAMP_KAPPA,
                "fixed_cost_multiplier": EXP2_LOCAL_FIXED_COST_MULT,
                "qualification_cost_multiplier": EXP2_LOCAL_QUAL_COST_MULT,
            },
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_experiment_2() -> None:
    out_dir = os.path.join("outputs", "experiment_2_tail_level_span_sensitivity_H80_tail65_neggrid_localcost_v2")
    _ensure_dir(out_dir)

    cfg = TrainEvalConfig(
        hidden=32,
        device="cpu",
        iterations=10,
        episodes_per_iter=10,
        eval_episode_seed=20260325,
    )

    results = evaluate_exp2_grid(
        level_shifts=LEVEL_SHIFTS,
        spans=SPAN_LEVELS,
        cfg=cfg,
    )
    rows = results_to_rows(results)

    results_csv = os.path.join(out_dir, "exp2_tail_level_span_results.csv")
    grid_json = os.path.join(out_dir, "exp2_grid_spec.json")
    fig_total_png = os.path.join(out_dir, "fig_total_cost_heatmaps.png")
    fig_total_pdf = os.path.join(out_dir, "fig_total_cost_heatmaps.pdf")
    fig_gap_png = os.path.join(out_dir, "fig_strategy_gap_heatmaps.png")
    fig_gap_pdf = os.path.join(out_dir, "fig_strategy_gap_heatmaps.pdf")
    fig_act_png = os.path.join(out_dir, "fig_action_timing_heatmaps.png")
    fig_act_pdf = os.path.join(out_dir, "fig_action_timing_heatmaps.pdf")
    fig_prof_png = os.path.join(out_dir, "fig_cost_profiles_by_span.png")
    fig_prof_pdf = os.path.join(out_dir, "fig_cost_profiles_by_span.pdf")

    _write_results_csv(results_csv, rows)
    _write_grid_spec(grid_json)
    plot_total_cost_heatmaps(results, fig_total_png, fig_total_pdf)
    plot_gap_heatmaps(results, fig_gap_png, fig_gap_pdf)
    plot_activation_heatmaps(results, fig_act_png, fig_act_pdf)
    plot_profile_lines(results, fig_prof_png, fig_prof_pdf)

    best_b = min(results, key=lambda r: r.total_B)
    best_c = min(results, key=lambda r: r.total_C)
    print("\n=== Experiment 2  last-65-period M1->D1 tariff downside sensitivity (local exp2 cost tweak) ===")
    print(f"Profiles evaluated = {len(results)}")
    print(f"Level shifts = {[int(round(v*100)) for v in LEVEL_SHIFTS]}%")
    print(f"Span levels = {[f'±{int(round(v*100))}%' for v in SPAN_LEVELS]}")
    print(
        "Exp2 local override: "
        f"M2->D1 c_out={EXP2_LOCAL_M2_D1_COUT:.2f} | "
        f"ramp_kappa={EXP2_LOCAL_M2_RAMP_KAPPA:.3f} | "
        f"F x{EXP2_LOCAL_FIXED_COST_MULT:.2f} | "
        f"qual_G x{EXP2_LOCAL_QUAL_COST_MULT:.2f}"
    )
    print(
        "Best B profile: "
        f"shift={int(round(best_b.level_shift*100))}% | "
        f"span=±{int(round(best_b.span*100))}% | "
        f"cost={best_b.total_B:.2f} | activation={best_b.activation_B}"
    )
    print(
        "Best C profile: "
        f"shift={int(round(best_c.level_shift*100))}% | "
        f"span=±{int(round(best_c.span*100))}% | "
        f"cost={best_c.total_C:.2f} | withdrawal={best_c.withdrawal_C}"
    )
    print(f"saved results csv      -> {results_csv}")
    print(f"saved grid spec json   -> {grid_json}")
    print(f"saved total heatmaps   -> {fig_total_png}")
    print(f"saved gap heatmaps     -> {fig_gap_png}")
    print(f"saved B timing heatmap -> {fig_act_png}")
    print(f"saved profile lines    -> {fig_prof_png}")


if __name__ == "__main__":
    run_experiment_2()
