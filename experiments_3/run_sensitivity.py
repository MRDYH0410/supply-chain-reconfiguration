from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Dict, List, Sequence

from experiments_2.sensitivity_runner import TariffLevelConfig
from experiments_3.sensitivity_plots import (
    PathContactSheetSpec,
    PathTariffPoint,
    ProfileSummaryPoint,
    plot_path_phase2_cost_vs_profile,
    plot_path_phase2_length_vs_profile,
    plot_path_total_cost_contact_sheet,
    plot_path_total_cost_vs_profile,
    plot_profile_cost_gaps,
    plot_profile_phase2_cost,
    plot_profile_phase_length,
    plot_profile_total_cost,
    plot_profile_win_share,
)
from experiments_3.sensitivity_runner import (
    TrainEvalConfig,
    build_default_tariff_profiles,
    build_tariff_path_specs_from_phase_structure,
    describe_tariff_profile,
    detect_phase_structure_via_strategy_b,
    evaluate_three_strategies,
    joint_state_full_label,
    make_joint_tariff_scenario_with_profile,
    make_profiled_tariff_path_scenario,
)


PRINT_TIMELINE_TRACE = False
TRACE_EPISODE_SEED = 20260310


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_token(value: str) -> str:
    token = str(value).strip().replace(" ", "_").replace("|", "-").replace(">", "-")
    token = token.replace("/", "-").replace("\\", "-").replace(":", "-")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def _profile_tag(profile_id: str) -> str:
    return _safe_token(profile_id)


def _phase_mean(series: Sequence[float], periods: tuple[int, int]) -> float:
    start, end = int(periods[0]), int(periods[1])
    if start > end:
        return 0.0
    vals = [float(series[t - 1]) for t in range(start, end + 1) if 1 <= t <= len(series)]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _winner(costA: float, costB: float, costC: float) -> str:
    vals = {"A": costA, "B": costB, "C": costC}
    return min(vals, key=vals.get)


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flatten_profile_levels(levels: dict) -> dict:
    out: dict = {}
    for side in ["original_side", "new_side"]:
        prefix = "orig" if side == "original_side" else "new"
        for level in ["low", "high"]:
            for market_id, value in levels[side][level].items():
                out[f"{prefix}_{level}_{market_id}"] = float(value)
    return out


def _write_profile_raw_csv(path: str, rows: List[dict]) -> None:
    fieldnames = [
        "profile_id", "profile_name", "low_scale", "high_scale",
        "phase2_length", "phase1_start", "phase1_end", "phase2_start", "phase2_end", "phase3_start", "phase3_end",
        "change_period_1", "change_period_2",
        "path_id", "path_label", "phase1_state", "phase2_state", "phase3_state",
        "phase2_label", "phase3_label", "best_strategy",
        "mean_cost_A", "std_cost_A", "mean_cost_B", "std_cost_B", "mean_cost_C", "std_cost_C",
        "gap_B_minus_A", "gap_C_minus_A", "gap_C_minus_B",
        "phase2_mean_cost_A", "phase2_mean_cost_B", "phase2_mean_cost_C",
        "phase2_gap_B_minus_A", "phase2_gap_C_minus_A", "phase2_gap_C_minus_B",
        "orig_low_D1", "orig_high_D1", "new_low_D1", "new_high_D1",
        "orig_low_D2", "orig_high_D2", "new_low_D2", "new_high_D2",
    ]
    _write_csv(path, rows, fieldnames)


def _write_profile_summary_csv(path: str, rows: List[dict]) -> None:
    fieldnames = [
        "profile_id", "profile_name", "low_scale", "high_scale",
        "phase2_length", "phase2_start", "phase3_start", "change_period_1", "change_period_2",
        "mean_total_cost_A", "mean_total_cost_B", "mean_total_cost_C",
        "phase2_mean_cost_A", "phase2_mean_cost_B", "phase2_mean_cost_C",
        "mean_gap_B_minus_A", "mean_gap_C_minus_A", "mean_gap_C_minus_B",
        "phase2_gap_B_minus_A", "phase2_gap_C_minus_A", "phase2_gap_C_minus_B",
        "win_count_A", "win_count_B", "win_count_C",
        "win_share_A", "win_share_B", "win_share_C",
        "orig_low_D1", "orig_high_D1", "new_low_D1", "new_high_D1",
        "orig_low_D2", "orig_high_D2", "new_low_D2", "new_high_D2",
    ]
    _write_csv(path, rows, fieldnames)


def _build_path_comparison_wide_row(path_rows: List[dict]) -> dict:
    rows = list(path_rows)
    first = rows[0]
    out = {
        "path_id": first["path_id"],
        "path_label": first["path_label"],
        "phase1_state": first["phase1_state"],
        "phase2_state": first["phase2_state"],
        "phase3_state": first["phase3_state"],
        "phase2_label": first["phase2_label"],
        "phase3_label": first["phase3_label"],
        "n_profiles": len(rows),
    }
    for row in rows:
        tag = _profile_tag(str(row["profile_id"]))
        out[f"profile_name_{tag}"] = str(row["profile_name"])
        out[f"low_scale_{tag}"] = float(row["low_scale"])
        out[f"high_scale_{tag}"] = float(row["high_scale"])
        out[f"phase2_length_{tag}"] = int(row["phase2_length"])
        out[f"phase2_start_{tag}"] = int(row["phase2_start"])
        out[f"phase2_end_{tag}"] = int(row["phase2_end"])
        out[f"phase3_start_{tag}"] = int(row["phase3_start"])
        out[f"phase3_end_{tag}"] = int(row["phase3_end"])
        out[f"best_strategy_{tag}"] = row["best_strategy"]
        out[f"mean_cost_A_{tag}"] = float(row["mean_cost_A"])
        out[f"mean_cost_B_{tag}"] = float(row["mean_cost_B"])
        out[f"mean_cost_C_{tag}"] = float(row["mean_cost_C"])
        out[f"gap_B_minus_A_{tag}"] = float(row["gap_B_minus_A"])
        out[f"gap_C_minus_A_{tag}"] = float(row["gap_C_minus_A"])
        out[f"phase2_mean_cost_A_{tag}"] = float(row["phase2_mean_cost_A"])
        out[f"phase2_mean_cost_B_{tag}"] = float(row["phase2_mean_cost_B"])
        out[f"phase2_mean_cost_C_{tag}"] = float(row["phase2_mean_cost_C"])
    return out


def _write_single_row_csv(path: str, row: dict) -> None:
    _write_csv(path, [row], list(row.keys()))


def _write_path_outputs(raw_rows: List[dict], out_dir: str) -> None:
    by_path_dir = os.path.join(out_dir, "by_path")
    _ensure_dir(by_path_dir)

    grouped: dict[str, List[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row["path_id"])].append(row)

    summary_rows: List[dict] = []
    summary_path = os.path.join(out_dir, "tariff_level_summary_by_path.csv")
    fieldnames = [
        "path_id", "path_label", "phase1_state", "phase2_state", "phase3_state",
        "phase2_label", "phase3_label", "n_profiles",
        "mean_phase2_length", "min_phase2_length", "max_phase2_length",
        "mean_cost_A_over_profiles", "mean_cost_B_over_profiles", "mean_cost_C_over_profiles",
        "min_gap_B_minus_A", "max_gap_B_minus_A", "min_gap_C_minus_A", "max_gap_C_minus_A",
        "best_count_A", "best_count_B", "best_count_C",
    ]
    contact_specs: List[PathContactSheetSpec] = []

    for path_id in sorted(grouped.keys()):
        rows = list(grouped[path_id])
        first = rows[0]
        folder_name = f"{first['path_id']}_{_safe_token(first['path_label'])}"
        path_dir = os.path.join(by_path_dir, folder_name)
        _ensure_dir(path_dir)

        long_csv = os.path.join(path_dir, "comparison_long.csv")
        wide_csv = os.path.join(path_dir, "comparison_wide.csv")
        meta_json = os.path.join(path_dir, "meta.json")

        _write_profile_raw_csv(long_csv, rows)
        _write_single_row_csv(wide_csv, _build_path_comparison_wide_row(rows))

        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "path_id": first["path_id"],
                    "path_label": first["path_label"],
                    "phase_state_template": {
                        "phase1_state": first["phase1_state"],
                        "phase2_state": first["phase2_state"],
                        "phase3_state": first["phase3_state"],
                        "phase2_label": first["phase2_label"],
                        "phase3_label": first["phase3_label"],
                    },
                    "note": "Within each tariff-path folder, the joint tariff-state template is fixed while the numeric tariff level behind H and L changes with the profile.",
                    "available_profiles": [
                        {
                            "profile_id": str(r["profile_id"]),
                            "profile_name": str(r["profile_name"]),
                            "low_scale": float(r["low_scale"]),
                            "high_scale": float(r["high_scale"]),
                        }
                        for r in rows
                    ],
                },
                f,
                indent=2,
            )

        points = [
            PathTariffPoint(
                path_id=str(r["path_id"]),
                path_label=str(r["path_label"]),
                profile_id=str(r["profile_id"]),
                profile_name=str(r["profile_name"]),
                low_scale=float(r["low_scale"]),
                high_scale=float(r["high_scale"]),
                phase2_length=int(r["phase2_length"]),
                phase2_start=int(r["phase2_start"]),
                phase3_start=int(r["phase3_start"]),
                mean_cost_A=float(r["mean_cost_A"]),
                mean_cost_B=float(r["mean_cost_B"]),
                mean_cost_C=float(r["mean_cost_C"]),
                std_cost_A=float(r["std_cost_A"]),
                std_cost_B=float(r["std_cost_B"]),
                std_cost_C=float(r["std_cost_C"]),
                phase2_mean_cost_A=float(r["phase2_mean_cost_A"]),
                phase2_mean_cost_B=float(r["phase2_mean_cost_B"]),
                phase2_mean_cost_C=float(r["phase2_mean_cost_C"]),
                gap_B_minus_A=float(r["gap_B_minus_A"]),
                gap_C_minus_A=float(r["gap_C_minus_A"]),
                best_strategy=str(r["best_strategy"]),
            )
            for r in rows
        ]
        plot_path_total_cost_vs_profile(
            points,
            out_png=os.path.join(path_dir, "fig_total_cost_vs_profile.png"),
            out_pdf=os.path.join(path_dir, "fig_total_cost_vs_profile.pdf"),
        )
        plot_path_phase2_cost_vs_profile(
            points,
            out_png=os.path.join(path_dir, "fig_phase2_cost_vs_profile.png"),
            out_pdf=os.path.join(path_dir, "fig_phase2_cost_vs_profile.pdf"),
        )
        plot_path_phase2_length_vs_profile(
            points,
            out_png=os.path.join(path_dir, "fig_phase2_length_vs_profile.png"),
            out_pdf=os.path.join(path_dir, "fig_phase2_length_vs_profile.pdf"),
        )

        best_counter = Counter(str(r["best_strategy"]) for r in rows)
        summary_rows.append({
            "path_id": first["path_id"],
            "path_label": first["path_label"],
            "phase1_state": first["phase1_state"],
            "phase2_state": first["phase2_state"],
            "phase3_state": first["phase3_state"],
            "phase2_label": first["phase2_label"],
            "phase3_label": first["phase3_label"],
            "n_profiles": len(rows),
            "mean_phase2_length": sum(float(r["phase2_length"]) for r in rows) / len(rows),
            "min_phase2_length": min(int(r["phase2_length"]) for r in rows),
            "max_phase2_length": max(int(r["phase2_length"]) for r in rows),
            "mean_cost_A_over_profiles": sum(float(r["mean_cost_A"]) for r in rows) / len(rows),
            "mean_cost_B_over_profiles": sum(float(r["mean_cost_B"]) for r in rows) / len(rows),
            "mean_cost_C_over_profiles": sum(float(r["mean_cost_C"]) for r in rows) / len(rows),
            "min_gap_B_minus_A": min(float(r["gap_B_minus_A"]) for r in rows),
            "max_gap_B_minus_A": max(float(r["gap_B_minus_A"]) for r in rows),
            "min_gap_C_minus_A": min(float(r["gap_C_minus_A"]) for r in rows),
            "max_gap_C_minus_A": max(float(r["gap_C_minus_A"]) for r in rows),
            "best_count_A": int(best_counter.get("A", 0)),
            "best_count_B": int(best_counter.get("B", 0)),
            "best_count_C": int(best_counter.get("C", 0)),
        })
        _write_csv(summary_path, summary_rows, fieldnames)

        contact_specs.append(
            PathContactSheetSpec(
                path_id=str(first["path_id"]),
                path_label=str(first["path_label"]),
                points=points,
            )
        )

    plot_path_total_cost_contact_sheet(
        specs=contact_specs,
        out_png=os.path.join(out_dir, "fig_16paths_total_cost_vs_profile.png"),
        out_pdf=os.path.join(out_dir, "fig_16paths_total_cost_vs_profile.pdf"),
    )


def run_tariff_level_sensitivity(
    seeds: List[int],
    cfg: TrainEvalConfig,
    profiles=None,
) -> None:
    if profiles is None:
        profiles = build_default_tariff_profiles()

    out_dir = os.path.join("outputs", "tariff_level_sensitivity")
    raw_dir = os.path.join(out_dir, "raw")
    _ensure_dir(out_dir)
    _ensure_dir(raw_dir)

    level_cfg = TariffLevelConfig(
        original_side_low_regime=1,
        original_side_high_regime=2,
        new_side_low_regime=1,
        new_side_high_regime=3,
    )

    raw_rows: List[dict] = []
    summary_rows: List[dict] = []
    summary_points: List[ProfileSummaryPoint] = []

    print("\n=== Running sensitivity experiment 3: tariff-level uncertainty ===")
    print("Research focus: keep the 16 tariff paths fixed, but vary the numeric tariff intensity behind H and L.")
    print("Constraint used in code: for every edge and market, low tariff state is forced not to exceed the high tariff state.")
    print(f"Seeds = {seeds}")
    print("Profiles =")
    for p in profiles:
        print(f"  {p.profile_id} | {p.profile_name} | low_scale={p.low_scale:.2f}, high_scale={p.high_scale:.2f} | {p.note}")

    for profile in profiles:
        profile_desc = describe_tariff_profile(profile, level_cfg=level_cfg)
        profile_levels = profile_desc.tariff_levels
        scenario_base = make_joint_tariff_scenario_with_profile(
            profile=profile,
            validate=False,
            level_cfg=level_cfg,
        )

        phase_structure = detect_phase_structure_via_strategy_b(
            seeds=seeds,
            cfg=cfg,
            level_cfg=level_cfg,
            scenario=scenario_base,
        )
        specs = build_tariff_path_specs_from_phase_structure(
            phase_structure=phase_structure,
            horizon=int(scenario_base.H),
        )
        phase2_length = int(phase_structure.phase2_periods[1] - phase_structure.phase2_periods[0] + 1)

        print(
            f"\n{profile.profile_id} | {profile.profile_name} | "
            f"Phase II={phase_structure.phase2_periods} ({phase2_length} periods) | "
            f"Phase III starts at t={phase_structure.phase3_start}"
        )
        print(f"  tariff levels = {profile_levels}")

        per_profile_rows: List[dict] = []
        best_counter: Counter = Counter()
        level_flat = _flatten_profile_levels(profile_levels)

        for spec in specs:
            scenario = make_profiled_tariff_path_scenario(
                spec,
                profile=profile,
                validate=False,
                level_cfg=level_cfg,
            )
            stats = evaluate_three_strategies(
                scenario,
                seeds=seeds,
                cfg=cfg,
                return_traces=False,
            )
            A, B, C = stats["A"], stats["B"], stats["C"]
            best = _winner(A.mean_total_cost, B.mean_total_cost, C.mean_total_cost)
            best_counter[best] += 1

            phase2_mean_A = _phase_mean(A.mean_cost_by_t, spec.phase2_periods)
            phase2_mean_B = _phase_mean(B.mean_cost_by_t, spec.phase2_periods)
            phase2_mean_C = _phase_mean(C.mean_cost_by_t, spec.phase2_periods)

            row = {
                "profile_id": str(profile.profile_id),
                "profile_name": str(profile.profile_name),
                "low_scale": float(profile.low_scale),
                "high_scale": float(profile.high_scale),
                "phase2_length": int(phase2_length),
                "phase1_start": int(spec.phase1_periods[0]),
                "phase1_end": int(spec.phase1_periods[1]),
                "phase2_start": int(spec.phase2_periods[0]),
                "phase2_end": int(spec.phase2_periods[1]),
                "phase3_start": int(spec.phase3_periods[0]),
                "phase3_end": int(spec.phase3_periods[1]),
                "change_period_1": int(phase_structure.change_period_1),
                "change_period_2": int(phase_structure.change_period_2),
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
                "phase2_mean_cost_A": float(phase2_mean_A),
                "phase2_mean_cost_B": float(phase2_mean_B),
                "phase2_mean_cost_C": float(phase2_mean_C),
                "phase2_gap_B_minus_A": float(phase2_mean_B - phase2_mean_A),
                "phase2_gap_C_minus_A": float(phase2_mean_C - phase2_mean_A),
                "phase2_gap_C_minus_B": float(phase2_mean_C - phase2_mean_B),
            }
            row.update(level_flat)
            raw_rows.append(row)
            per_profile_rows.append(row)

            print(
                f"  {spec.path_id} | {spec.path_label} | "
                f"A={A.mean_total_cost:.1f}, B={B.mean_total_cost:.1f}, C={C.mean_total_cost:.1f} | best={best}"
            )

        n_paths = max(1, len(per_profile_rows))
        mean_total_cost_A = sum(r["mean_cost_A"] for r in per_profile_rows) / n_paths
        mean_total_cost_B = sum(r["mean_cost_B"] for r in per_profile_rows) / n_paths
        mean_total_cost_C = sum(r["mean_cost_C"] for r in per_profile_rows) / n_paths
        phase2_mean_cost_A = sum(r["phase2_mean_cost_A"] for r in per_profile_rows) / n_paths
        phase2_mean_cost_B = sum(r["phase2_mean_cost_B"] for r in per_profile_rows) / n_paths
        phase2_mean_cost_C = sum(r["phase2_mean_cost_C"] for r in per_profile_rows) / n_paths
        mean_gap_B_minus_A = sum(r["gap_B_minus_A"] for r in per_profile_rows) / n_paths
        mean_gap_C_minus_A = sum(r["gap_C_minus_A"] for r in per_profile_rows) / n_paths
        mean_gap_C_minus_B = sum(r["gap_C_minus_B"] for r in per_profile_rows) / n_paths
        phase2_gap_B_minus_A = sum(r["phase2_gap_B_minus_A"] for r in per_profile_rows) / n_paths
        phase2_gap_C_minus_A = sum(r["phase2_gap_C_minus_A"] for r in per_profile_rows) / n_paths
        phase2_gap_C_minus_B = sum(r["phase2_gap_C_minus_B"] for r in per_profile_rows) / n_paths

        summary_row = {
            "profile_id": str(profile.profile_id),
            "profile_name": str(profile.profile_name),
            "low_scale": float(profile.low_scale),
            "high_scale": float(profile.high_scale),
            "phase2_length": int(phase2_length),
            "phase2_start": int(phase_structure.phase2_start),
            "phase3_start": int(phase_structure.phase3_start),
            "change_period_1": int(phase_structure.change_period_1),
            "change_period_2": int(phase_structure.change_period_2),
            "mean_total_cost_A": float(mean_total_cost_A),
            "mean_total_cost_B": float(mean_total_cost_B),
            "mean_total_cost_C": float(mean_total_cost_C),
            "phase2_mean_cost_A": float(phase2_mean_cost_A),
            "phase2_mean_cost_B": float(phase2_mean_cost_B),
            "phase2_mean_cost_C": float(phase2_mean_cost_C),
            "mean_gap_B_minus_A": float(mean_gap_B_minus_A),
            "mean_gap_C_minus_A": float(mean_gap_C_minus_A),
            "mean_gap_C_minus_B": float(mean_gap_C_minus_B),
            "phase2_gap_B_minus_A": float(phase2_gap_B_minus_A),
            "phase2_gap_C_minus_A": float(phase2_gap_C_minus_A),
            "phase2_gap_C_minus_B": float(phase2_gap_C_minus_B),
            "win_count_A": int(best_counter.get("A", 0)),
            "win_count_B": int(best_counter.get("B", 0)),
            "win_count_C": int(best_counter.get("C", 0)),
            "win_share_A": float(best_counter.get("A", 0) / n_paths),
            "win_share_B": float(best_counter.get("B", 0) / n_paths),
            "win_share_C": float(best_counter.get("C", 0) / n_paths),
        }
        summary_row.update(level_flat)
        summary_rows.append(summary_row)
        summary_points.append(
            ProfileSummaryPoint(
                profile_id=str(profile.profile_id),
                profile_name=str(profile.profile_name),
                low_scale=float(profile.low_scale),
                high_scale=float(profile.high_scale),
                phase2_length=int(phase2_length),
                mean_total_cost_A=float(mean_total_cost_A),
                mean_total_cost_B=float(mean_total_cost_B),
                mean_total_cost_C=float(mean_total_cost_C),
                phase2_mean_cost_A=float(phase2_mean_cost_A),
                phase2_mean_cost_B=float(phase2_mean_cost_B),
                phase2_mean_cost_C=float(phase2_mean_cost_C),
                mean_gap_B_minus_A=float(mean_gap_B_minus_A),
                mean_gap_C_minus_A=float(mean_gap_C_minus_A),
                mean_gap_C_minus_B=float(mean_gap_C_minus_B),
                phase2_gap_B_minus_A=float(phase2_gap_B_minus_A),
                phase2_gap_C_minus_A=float(phase2_gap_C_minus_A),
                phase2_gap_C_minus_B=float(phase2_gap_C_minus_B),
                win_share_A=float(best_counter.get("A", 0) / n_paths),
                win_share_B=float(best_counter.get("B", 0) / n_paths),
                win_share_C=float(best_counter.get("C", 0) / n_paths),
            )
        )

        print(
            "  aggregate across 16 tariff paths | "
            f"Phase II length={phase2_length}, "
            f"mean total cost A/B/C=({mean_total_cost_A:.1f}, {mean_total_cost_B:.1f}, {mean_total_cost_C:.1f}), "
            f"win counts A/B/C=({best_counter.get('A', 0)}, {best_counter.get('B', 0)}, {best_counter.get('C', 0)})"
        )

    raw_csv = os.path.join(raw_dir, "tariff_level_raw_long.csv")
    summary_csv = os.path.join(raw_dir, "tariff_level_summary_by_profile.csv")
    meta_json = os.path.join(out_dir, "tariff_level_meta.json")

    _write_profile_raw_csv(raw_csv, raw_rows)
    _write_profile_summary_csv(summary_csv, summary_rows)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "profiles": [
                    {
                        "profile_id": p.profile_id,
                        "profile_name": p.profile_name,
                        "low_scale": p.low_scale,
                        "high_scale": p.high_scale,
                        "note": p.note,
                    }
                    for p in profiles
                ],
                "tariff_level_config": {
                    "original_side_low_regime": level_cfg.original_side_low_regime,
                    "original_side_high_regime": level_cfg.original_side_high_regime,
                    "new_side_low_regime": level_cfg.new_side_low_regime,
                    "new_side_high_regime": level_cfg.new_side_high_regime,
                },
                "profile_tariff_levels": {
                    p.profile_id: describe_tariff_profile(p, level_cfg=level_cfg).tariff_levels for p in profiles
                },
                "note": "For each tariff-level profile, the phase split is re-detected via Strategy B and then the same 16 tariff paths are evaluated.",
            },
            f,
            indent=2,
        )

    plot_profile_phase_length(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_profile_vs_phase2_length.png"),
        out_pdf=os.path.join(out_dir, "fig_profile_vs_phase2_length.pdf"),
    )
    plot_profile_total_cost(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_profile_vs_total_cost.png"),
        out_pdf=os.path.join(out_dir, "fig_profile_vs_total_cost.pdf"),
    )
    plot_profile_phase2_cost(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_profile_vs_phase2_cost.png"),
        out_pdf=os.path.join(out_dir, "fig_profile_vs_phase2_cost.pdf"),
    )
    plot_profile_cost_gaps(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_profile_vs_cost_gaps.png"),
        out_pdf=os.path.join(out_dir, "fig_profile_vs_cost_gaps.pdf"),
    )
    plot_profile_win_share(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_profile_vs_win_share.png"),
        out_pdf=os.path.join(out_dir, "fig_profile_vs_win_share.pdf"),
    )
    _write_path_outputs(raw_rows=raw_rows, out_dir=out_dir)

    print(f"\nsaved raw csv              -> {raw_csv}")
    print(f"saved summary-by-profile  -> {summary_csv}")
    print(f"saved summary-by-path     -> {os.path.join(out_dir, 'tariff_level_summary_by_path.csv')}")
    print(f"saved by-path folders     -> {os.path.join(out_dir, 'by_path')}")
    print(f"saved 16-path sheet       -> {os.path.join(out_dir, 'fig_16paths_total_cost_vs_profile.png')}")
    print(f"saved meta json           -> {meta_json}")
    print(f"saved overall figures     -> {out_dir}")


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

    run_tariff_level_sensitivity(
        seeds=seeds,
        cfg=cfg,
        profiles=build_default_tariff_profiles(),
    )


if __name__ == "__main__":
    main()