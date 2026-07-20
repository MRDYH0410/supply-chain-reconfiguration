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
from experiments_5.sensitivity_plots import (
    plot_activation_scatter,
    plot_action_and_resets,
    plot_tariff_profiles,
    plot_total_costs,
    plot_total_demand,
)
from experiments_5.sensitivity_runner import (
    BASE_PATH_SEED,
    DEFAULT_N_REPLICATIONS,
    RESET_STREAK,
    RESET_THRESHOLD,
    RISK_BUCKETS,
    T1_LOW_TARIFF,
    T2_TARIFF_GRID,
    TARIFF_FLOOR,
    evaluate_exp5_grid,
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


def _write_spec_json(path: str, n_replications: int) -> None:
    payload = {
        "experiment": "experiment_5_risk_single_lane_tariff_sensitivity",
        "changed_lane_only": "M1->D1",
        "other_three_lanes": "Kept identical to Experiment 1 fixed baseline.",
        "t1_tariff": T1_LOW_TARIFF,
        "t2_tariff_grid": T2_TARIFF_GRID,
        "n_replications_per_t2": int(n_replications),
        "aggregation": "All reported costs, demands, timings, resets, and tariff paths are Monte Carlo means across replications.",
        "t_ge_3_rule": {
            "type": "risk_bucket_random_walk",
            "floor": TARIFF_FLOOR,
            "buckets": [
                {"probability": p, "abs_change_low": lo, "abs_change_high": hi}
                for p, lo, hi in RISK_BUCKETS
            ],
            "interpretation": "Each period draws one change-magnitude bucket, then applies a symmetric plus/minus shock in percentage-point terms.",
        },
        "risk_control": {
            "trigger": f"If the previous {RESET_STREAK} realised tariffs are all above {RESET_THRESHOLD}",
            "action": "Reset the next period tariff to the period-2 anchor, then continue the same transmission rule.",
        },
        "path_seed_rule": f"BASE_PATH_SEED + 100000*round(t2_tariff*1000) + replication_idx, BASE_PATH_SEED={BASE_PATH_SEED}",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_experiment_5(n_replications: int = DEFAULT_N_REPLICATIONS) -> None:
    out_dir = os.path.join("outputs", "experiment_5_risk_single_lane_tariff")
    _ensure_dir(out_dir)

    cfg = TrainEvalConfig(
        hidden=32,
        device="cpu",
        iterations=10,
        episodes_per_iter=10,
        eval_episode_seed=20260325,
    )

    results = evaluate_exp5_grid(T2_TARIFF_GRID, cfg, n_replications=int(n_replications))
    rows = results_to_rows(results)

    _write_results_csv(os.path.join(out_dir, "exp5_results.csv"), rows)
    _write_spec_json(os.path.join(out_dir, "exp5_spec.json"), int(n_replications))

    plot_tariff_profiles(results, os.path.join(out_dir, "fig_tariff_profiles.png"), os.path.join(out_dir, "fig_tariff_profiles.pdf"))
    plot_total_costs(results, os.path.join(out_dir, "fig_total_cost_vs_t2.png"), os.path.join(out_dir, "fig_total_cost_vs_t2.pdf"))
    plot_total_demand(results, os.path.join(out_dir, "fig_total_demand_vs_t2.png"), os.path.join(out_dir, "fig_total_demand_vs_t2.pdf"))
    plot_action_and_resets(results, os.path.join(out_dir, "fig_action_timing_and_resets.png"), os.path.join(out_dir, "fig_action_timing_and_resets.pdf"))
    plot_activation_scatter(results, os.path.join(out_dir, "fig_activation_scatter_vs_t2.png"), os.path.join(out_dir, "fig_activation_scatter_vs_t2.pdf"))

    best_b = min(results, key=lambda r: r.total_B)
    best_c = min(results, key=lambda r: r.total_C)
    print("\n=== Experiment 5  risk tariff environment on M1->D1 only ===")
    print(f"Profiles evaluated = {len(results)}")
    print(f"Monte Carlo replications per t2 = {int(n_replications)}")
    print(f"t2 tariff grid (%) = {[int(round(v*100)) for v in T2_TARIFF_GRID]}")
    print(
        f"Best mean B at t2={int(round(best_b.t2_tariff*100))}% | cost={best_b.total_B:.2f} | "
        f"mean activation={best_b.activation_B:.2f} | mean resets={best_b.mean_reset_count:.2f}"
    )
    print(
        f"Best mean C at t2={int(round(best_c.t2_tariff*100))}% | cost={best_c.total_C:.2f} | "
        f"mean withdrawal={best_c.withdrawal_C:.2f} | mean resets={best_c.mean_reset_count:.2f}"
    )
    print(f"saved outputs -> {out_dir}")


def main() -> None:
    run_experiment_5()


if __name__ == "__main__":
    main()
