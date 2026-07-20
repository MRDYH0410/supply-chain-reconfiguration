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
from experiments_3.sensitivity_plots import (
    plot_action_timing_vs_horizon,
    plot_cost_gap_vs_horizon,
    plot_total_cost_vs_horizon,
    plot_total_demand_vs_horizon,
)
from experiments_3.sensitivity_runner import (
    HORIZON_LEVELS,
    evaluate_exp3_horizon_grid,
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
        "horizon_levels": HORIZON_LEVELS,
        "interpretation": {
            "study_goal": "Hold the Experiment 1 fixed tariff environment constant and vary only the planning horizon H.",
            "extension_rule": "After period 30, the fixed baseline seasonal demand cycle and tariff tail patterns repeat deterministically.",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_experiment_3() -> None:
    out_dir = os.path.join("outputs", "experiment_3_horizon_sensitivity")
    _ensure_dir(out_dir)

    cfg = TrainEvalConfig(
        hidden=32,
        device="cpu",
        iterations=10,
        episodes_per_iter=10,
        eval_episode_seed=20260325,
    )

    results = evaluate_exp3_horizon_grid(
        horizons=HORIZON_LEVELS,
        cfg=cfg,
    )
    rows = results_to_rows(results)

    results_csv = os.path.join(out_dir, "exp3_horizon_sensitivity_results.csv")
    grid_json = os.path.join(out_dir, "exp3_horizon_grid_spec.json")
    fig_total_png = os.path.join(out_dir, "fig_total_discounted_cost_vs_horizon.png")
    fig_total_pdf = os.path.join(out_dir, "fig_total_discounted_cost_vs_horizon.pdf")
    fig_gap_png = os.path.join(out_dir, "fig_strategy_gap_vs_horizon.png")
    fig_gap_pdf = os.path.join(out_dir, "fig_strategy_gap_vs_horizon.pdf")
    fig_demand_png = os.path.join(out_dir, "fig_total_realised_demand_vs_horizon.png")
    fig_demand_pdf = os.path.join(out_dir, "fig_total_realised_demand_vs_horizon.pdf")
    fig_timing_png = os.path.join(out_dir, "fig_action_timing_vs_horizon.png")
    fig_timing_pdf = os.path.join(out_dir, "fig_action_timing_vs_horizon.pdf")

    _write_results_csv(results_csv, rows)
    _write_grid_spec(grid_json)
    plot_total_cost_vs_horizon(results, fig_total_png, fig_total_pdf)
    plot_cost_gap_vs_horizon(results, fig_gap_png, fig_gap_pdf)
    plot_total_demand_vs_horizon(results, fig_demand_png, fig_demand_pdf)
    plot_action_timing_vs_horizon(results, fig_timing_png, fig_timing_pdf)

    best_b = min(results, key=lambda r: r.total_B)
    best_c = min(results, key=lambda r: r.total_C)
    print("\n=== Experiment 3  horizon sensitivity under the fixed baseline environment ===")
    print(f"Horizons evaluated = {HORIZON_LEVELS}")
    print(
        "Best B horizon: "
        f"H={best_b.horizon} | cost={best_b.total_B:.2f} | activation={best_b.activation_B}"
    )
    print(
        "Best C horizon: "
        f"H={best_c.horizon} | cost={best_c.total_C:.2f} | activation={best_c.activation_C} | withdrawal={best_c.withdrawal_C}"
    )
    print(f"saved results csv   -> {results_csv}")
    print(f"saved grid spec     -> {grid_json}")
    print(f"saved cost figure   -> {fig_total_png}")
    print(f"saved gap figure    -> {fig_gap_png}")
    print(f"saved demand figure -> {fig_demand_png}")
    print(f"saved timing figure -> {fig_timing_png}")


def main() -> None:
    run_experiment_3()


if __name__ == "__main__":
    main()
