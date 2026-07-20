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
from experiments_4.sensitivity_plots import (
    plot_activation_scatter,
    plot_action_timing,
    plot_tariff_profiles,
    plot_total_costs,
    plot_total_demand,
)
from experiments_4.sensitivity_runner import (
    BASE_PATH_SEED,
    EXP4_C_OUT_OVERRIDE,
    STABLE_BAND,
    T1_LOW_TARIFF,
    T2_TARIFF_GRID,
    TARIFF_FLOOR,
    evaluate_exp4_grid,
    results_to_rows,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_results_csv(path: str, rows) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_spec_json(path: str) -> None:
    payload = {
        "experiment": "experiment_4_stable_single_lane_tariff_sensitivity",
        "changed_lane_only": "M1->D1",
        "other_three_lanes": "Kept identical to Experiment 1 fixed baseline.",
        "t1_tariff": T1_LOW_TARIFF,
        "t2_tariff_grid": T2_TARIFF_GRID,
        "t_ge_3_rule": {
            "type": "stable_uniform_band",
            "band": STABLE_BAND,
            "floor": TARIFF_FLOOR,
            "interpretation": "tau_t is sampled uniformly from [max(1%, tau_{t-1}-10pp), tau_{t-1}+10pp].",
        },
        "path_seed_rule": f"BASE_PATH_SEED + round(t2_tariff*1000), BASE_PATH_SEED={BASE_PATH_SEED}",
        "experiment_4_only_c_out_override": EXP4_C_OUT_OVERRIDE,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_experiment_4() -> None:
    out_dir = os.path.join("outputs", "experiment_4_stable_single_lane_tariff")
    _ensure_dir(out_dir)

    cfg = TrainEvalConfig(
        hidden=32,
        device="cpu",
        iterations=10,
        episodes_per_iter=10,
        eval_episode_seed=20260325,
    )

    results = evaluate_exp4_grid(T2_TARIFF_GRID, cfg)
    rows = results_to_rows(results)

    _write_results_csv(os.path.join(out_dir, "exp4_results.csv"), rows)
    _write_spec_json(os.path.join(out_dir, "exp4_spec.json"))

    plot_tariff_profiles(results, os.path.join(out_dir, "fig_tariff_profiles.png"), os.path.join(out_dir, "fig_tariff_profiles.pdf"))
    plot_total_costs(results, os.path.join(out_dir, "fig_total_cost_vs_t2.png"), os.path.join(out_dir, "fig_total_cost_vs_t2.pdf"))
    plot_total_demand(results, os.path.join(out_dir, "fig_total_demand_vs_t2.png"), os.path.join(out_dir, "fig_total_demand_vs_t2.pdf"))
    plot_action_timing(results, os.path.join(out_dir, "fig_action_timing_vs_t2.png"), os.path.join(out_dir, "fig_action_timing_vs_t2.pdf"))
    plot_activation_scatter(results, os.path.join(out_dir, "fig_activation_scatter_vs_t2.png"), os.path.join(out_dir, "fig_activation_scatter_vs_t2.pdf"))

    best_b = min(results, key=lambda r: r.total_B)
    best_c = min(results, key=lambda r: r.total_C)
    print("\n=== Experiment 4  stable tariff environment on M1->D1 only ===")
    print(f"Profiles evaluated = {len(results)}")
    print(f"t2 tariff grid (%) = {[int(round(v*100)) for v in T2_TARIFF_GRID]}")
    print(f"Experiment-4-only c_out override = {EXP4_C_OUT_OVERRIDE}")
    print(f"Best B at t2={int(round(best_b.t2_tariff*100))}% | cost={best_b.total_B:.2f} | activation={best_b.activation_B}")
    print(f"Best C at t2={int(round(best_c.t2_tariff*100))}% | cost={best_c.total_C:.2f} | withdrawal={best_c.withdrawal_C}")
    print(f"saved outputs -> {out_dir}")


def main() -> None:
    run_experiment_4()


if __name__ == "__main__":
    main()
