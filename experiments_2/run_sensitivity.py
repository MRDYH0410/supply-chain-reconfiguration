from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Dict, List, Sequence

from experiments_2.sensitivity_plots import (
    LinePlotSpec,
    PathContactSheetSpec,
    PathRampPoint,
    RampSummaryPoint,
    plot_all_tariff_path_lineplots,
    plot_path_phase2_cost_vs_kappa,
    plot_path_phase2_length_vs_kappa,
    plot_path_total_cost_contact_sheet,
    plot_path_total_cost_vs_kappa,
    plot_ramp_cost_gaps,
    plot_ramp_phase2_cost,
    plot_ramp_phase_length,
    plot_ramp_total_cost,
    plot_ramp_win_share,
    plot_tariff_path_contact_sheet,
)
from experiments_2.sensitivity_runner import (
    TrainEvalConfig,
    TariffLevelConfig,
    build_tariff_path_specs_from_phase_structure,
    detect_phase_structure_via_strategy_b,
    evaluate_three_strategies,
    extract_tariff_level_values,
    joint_state_full_label,
    make_scenario_base,
    make_scenario_with_candidate_ramp,
    make_tariff_path_scenario,
)


PRINT_TIMELINE_TRACE = False
TRACE_EPISODE_SEED = 20260306


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_token(value: str) -> str:
    token = str(value).strip().replace(" ", "_").replace("|", "-").replace(">", "-")
    token = token.replace("/", "-").replace("\\", "-").replace(":", "-")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def _kappa_tag(kappa: float) -> str:
    return f"k{float(kappa):.2f}".replace(".", "p")


def _cb_get(info: dict, key: str, default: float = 0.0) -> float:
    cb = info.get("cost_breakdown", {}) or {}
    return float(cb.get(key, default))


def _phase_mean(series: Sequence[float], periods: tuple[int, int]) -> float:
    start, end = int(periods[0]), int(periods[1])
    if start > end:
        return 0.0
    vals = [float(series[t - 1]) for t in range(start, end + 1) if 1 <= t <= len(series)]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


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


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_ramp_raw_csv(path: str, rows: List[dict]) -> None:
    fieldnames = [
        "ramp_kappa", "ramp_full_age", "phase2_length",
        "phase1_start", "phase1_end", "phase2_start", "phase2_end", "phase3_start", "phase3_end",
        "change_period_1", "change_period_2",
        "path_id", "path_label", "phase1_state", "phase2_state", "phase3_state",
        "phase2_label", "phase3_label", "best_strategy",
        "mean_cost_A", "std_cost_A", "mean_cost_B", "std_cost_B", "mean_cost_C", "std_cost_C",
        "gap_B_minus_A", "gap_C_minus_A", "gap_C_minus_B",
        "phase2_mean_cost_A", "phase2_mean_cost_B", "phase2_mean_cost_C",
        "phase2_gap_B_minus_A", "phase2_gap_C_minus_A", "phase2_gap_C_minus_B",
    ]
    _write_csv(path, rows, fieldnames)


def _write_ramp_summary_csv(path: str, rows: List[dict]) -> None:
    fieldnames = [
        "ramp_kappa", "ramp_full_age", "phase2_length", "phase2_start", "phase3_start",
        "change_period_1", "change_period_2",
        "mean_total_cost_A", "mean_total_cost_B", "mean_total_cost_C",
        "phase2_mean_cost_A", "phase2_mean_cost_B", "phase2_mean_cost_C",
        "mean_gap_B_minus_A", "mean_gap_C_minus_A", "mean_gap_C_minus_B",
        "phase2_gap_B_minus_A", "phase2_gap_C_minus_A", "phase2_gap_C_minus_B",
        "win_count_A", "win_count_B", "win_count_C",
        "win_share_A", "win_share_B", "win_share_C",
    ]
    _write_csv(path, rows, fieldnames)


def _build_path_comparison_wide_row(path_rows: List[dict]) -> dict:
    rows = sorted(path_rows, key=lambda r: float(r["ramp_kappa"]))
    first = rows[0]
    out = {
        "path_id": first["path_id"],
        "path_label": first["path_label"],
        "phase1_state": first["phase1_state"],
        "phase2_state": first["phase2_state"],
        "phase3_state": first["phase3_state"],
        "phase2_label": first["phase2_label"],
        "phase3_label": first["phase3_label"],
        "n_kappas": len(rows),
    }
    for row in rows:
        tag = _kappa_tag(float(row["ramp_kappa"]))
        out[f"ramp_kappa_{tag}"] = float(row["ramp_kappa"])
        out[f"ramp_full_age_{tag}"] = int(row["ramp_full_age"])
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
    contact_specs: List[PathContactSheetSpec] = []

    for path_id in sorted(grouped.keys()):
        rows = sorted(grouped[path_id], key=lambda r: float(r["ramp_kappa"]))
        first = rows[0]
        folder_name = f"{first['path_id']}_{_safe_token(first['path_label'])}"
        path_dir = os.path.join(by_path_dir, folder_name)
        _ensure_dir(path_dir)

        long_csv = os.path.join(path_dir, "comparison_long.csv")
        wide_csv = os.path.join(path_dir, "comparison_wide.csv")
        meta_json = os.path.join(path_dir, "meta.json")

        _write_ramp_raw_csv(long_csv, rows)
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
                    "note": "Within each tariff-path folder, the tariff-state template is fixed while phase timing is re-detected for each ramp_kappa.",
                    "available_kappas": [float(r["ramp_kappa"]) for r in rows],
                },
                f,
                indent=2,
            )

        points = [
            PathRampPoint(
                path_id=str(r["path_id"]),
                path_label=str(r["path_label"]),
                ramp_kappa=float(r["ramp_kappa"]),
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

        contact_specs.append(
            PathContactSheetSpec(
                path_id=str(first["path_id"]),
                path_label=str(first["path_label"]),
                points=points,
            )
        )

        plot_path_total_cost_vs_kappa(
            points,
            out_png=os.path.join(path_dir, "fig_total_cost_vs_kappa.png"),
            out_pdf=os.path.join(path_dir, "fig_total_cost_vs_kappa.pdf"),
        )
        plot_path_phase2_cost_vs_kappa(
            points,
            out_png=os.path.join(path_dir, "fig_phase2_cost_vs_kappa.png"),
            out_pdf=os.path.join(path_dir, "fig_phase2_cost_vs_kappa.pdf"),
        )
        plot_path_phase2_length_vs_kappa(
            points,
            out_png=os.path.join(path_dir, "fig_phase2_length_vs_kappa.png"),
            out_pdf=os.path.join(path_dir, "fig_phase2_length_vs_kappa.pdf"),
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
            "n_kappas": len(rows),
            "mean_phase2_length": sum(float(r["phase2_length"]) for r in rows) / len(rows),
            "min_phase2_length": min(int(r["phase2_length"]) for r in rows),
            "max_phase2_length": max(int(r["phase2_length"]) for r in rows),
            "mean_cost_A_over_kappa": sum(float(r["mean_cost_A"]) for r in rows) / len(rows),
            "mean_cost_B_over_kappa": sum(float(r["mean_cost_B"]) for r in rows) / len(rows),
            "mean_cost_C_over_kappa": sum(float(r["mean_cost_C"]) for r in rows) / len(rows),
            "min_gap_B_minus_A": min(float(r["gap_B_minus_A"]) for r in rows),
            "max_gap_B_minus_A": max(float(r["gap_B_minus_A"]) for r in rows),
            "min_gap_C_minus_A": min(float(r["gap_C_minus_A"]) for r in rows),
            "max_gap_C_minus_A": max(float(r["gap_C_minus_A"]) for r in rows),
            "best_count_A": int(best_counter.get("A", 0)),
            "best_count_B": int(best_counter.get("B", 0)),
            "best_count_C": int(best_counter.get("C", 0)),
        })

    summary_path = os.path.join(out_dir, "ramp_capability_summary_by_path.csv")
    fieldnames = [
        "path_id", "path_label", "phase1_state", "phase2_state", "phase3_state",
        "phase2_label", "phase3_label", "n_kappas",
        "mean_phase2_length", "min_phase2_length", "max_phase2_length",
        "mean_cost_A_over_kappa", "mean_cost_B_over_kappa", "mean_cost_C_over_kappa",
        "min_gap_B_minus_A", "max_gap_B_minus_A", "min_gap_C_minus_A", "max_gap_C_minus_A",
        "best_count_A", "best_count_B", "best_count_C",
    ]
    _write_csv(summary_path, summary_rows, fieldnames)

    plot_path_total_cost_contact_sheet(
        contact_specs,
        out_png=os.path.join(out_dir, "fig_total_cost_vs_kappa_contact_sheet.png"),
        out_pdf=os.path.join(out_dir, "fig_total_cost_vs_kappa_contact_sheet.pdf"),
    )


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


def run_ramp_capability_sensitivity(
    seeds: List[int],
    cfg: TrainEvalConfig,
    ramp_kappas: List[float] | None = None,
) -> None:
    if ramp_kappas is None:
        # ramp_kappas = [0.35, 0.45, 0.55, 0.70, 0.90, 1.20]
        ramp_kappas = [0.384, 0.461, 0.576, 0.768, 1.151]
    ramp_kappas = [float(x) for x in ramp_kappas]

    out_dir = os.path.join("outputs", "ramp_capability_sensitivity")
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
    summary_points: List[RampSummaryPoint] = []

    print("\n=== Running sensitivity experiment 2: ramp-up capability ===")
    print("Research focus: slower candidate ramp-up extends Phase II and may enlarge the value of strategies B and C.")
    print(f"Seeds = {seeds}")
    print(f"Ramp grid = {ramp_kappas}")
    print("Output view: keep raw long table, and additionally reorganize results by tariff path for paper-ready selection.")

    for ramp_kappa in ramp_kappas:
        scenario_base = make_scenario_with_candidate_ramp(ramp_kappa=ramp_kappa, validate=False)
        tariff_levels = extract_tariff_level_values(scenario_base, level_cfg)
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
            f"\nκ={ramp_kappa:.2f} | theoretical full-ramp age={scenario_base.ramp_full_age} | "
            f"Phase II={phase_structure.phase2_periods} ({phase2_length} periods) | "
            f"Phase III starts at t={phase_structure.phase3_start}"
        )
        print(f"  tariff levels unchanged: {tariff_levels}")

        per_kappa_rows: List[dict] = []
        best_counter: Counter = Counter()

        for spec in specs:
            scenario = make_tariff_path_scenario(
                spec,
                validate=False,
                level_cfg=level_cfg,
                base_scenario=scenario_base,
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
                "ramp_kappa": float(ramp_kappa),
                "ramp_full_age": int(scenario_base.ramp_full_age),
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
            raw_rows.append(row)
            per_kappa_rows.append(row)

            print(
                f"  {spec.path_id} | {spec.path_label} | "
                f"A={A.mean_total_cost:.1f}, B={B.mean_total_cost:.1f}, C={C.mean_total_cost:.1f} | best={best}"
            )

        n_paths = max(1, len(per_kappa_rows))
        mean_total_cost_A = sum(r["mean_cost_A"] for r in per_kappa_rows) / n_paths
        mean_total_cost_B = sum(r["mean_cost_B"] for r in per_kappa_rows) / n_paths
        mean_total_cost_C = sum(r["mean_cost_C"] for r in per_kappa_rows) / n_paths
        phase2_mean_cost_A = sum(r["phase2_mean_cost_A"] for r in per_kappa_rows) / n_paths
        phase2_mean_cost_B = sum(r["phase2_mean_cost_B"] for r in per_kappa_rows) / n_paths
        phase2_mean_cost_C = sum(r["phase2_mean_cost_C"] for r in per_kappa_rows) / n_paths
        mean_gap_B_minus_A = sum(r["gap_B_minus_A"] for r in per_kappa_rows) / n_paths
        mean_gap_C_minus_A = sum(r["gap_C_minus_A"] for r in per_kappa_rows) / n_paths
        mean_gap_C_minus_B = sum(r["gap_C_minus_B"] for r in per_kappa_rows) / n_paths
        phase2_gap_B_minus_A = sum(r["phase2_gap_B_minus_A"] for r in per_kappa_rows) / n_paths
        phase2_gap_C_minus_A = sum(r["phase2_gap_C_minus_A"] for r in per_kappa_rows) / n_paths
        phase2_gap_C_minus_B = sum(r["phase2_gap_C_minus_B"] for r in per_kappa_rows) / n_paths

        summary_row = {
            "ramp_kappa": float(ramp_kappa),
            "ramp_full_age": int(scenario_base.ramp_full_age),
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
        summary_rows.append(summary_row)
        summary_points.append(
            RampSummaryPoint(
                ramp_kappa=float(ramp_kappa),
                ramp_full_age=int(scenario_base.ramp_full_age),
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

    raw_csv = os.path.join(raw_dir, "ramp_capability_raw_long.csv")
    summary_csv = os.path.join(raw_dir, "ramp_capability_summary_by_kappa.csv")
    meta_json = os.path.join(out_dir, "ramp_capability_meta.json")

    _write_ramp_raw_csv(raw_csv, raw_rows)
    _write_ramp_summary_csv(summary_csv, summary_rows)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ramp_kappas": ramp_kappas,
                "tariff_level_config": {
                    "original_side_low_regime": level_cfg.original_side_low_regime,
                    "original_side_high_regime": level_cfg.original_side_high_regime,
                    "new_side_low_regime": level_cfg.new_side_low_regime,
                    "new_side_high_regime": level_cfg.new_side_high_regime,
                },
                "note": "For each kappa, the phase split is re-detected via Strategy B before evaluating the 16 tariff paths. Raw outputs are kept in long form and also reorganized by tariff path.",
            },
            f,
            indent=2,
        )

    plot_ramp_phase_length(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_ramp_vs_phase2_length.png"),
        out_pdf=os.path.join(out_dir, "fig_ramp_vs_phase2_length.pdf"),
    )
    plot_ramp_total_cost(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_ramp_vs_total_cost.png"),
        out_pdf=os.path.join(out_dir, "fig_ramp_vs_total_cost.pdf"),
    )
    plot_ramp_phase2_cost(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_ramp_vs_phase2_cost.png"),
        out_pdf=os.path.join(out_dir, "fig_ramp_vs_phase2_cost.pdf"),
    )
    plot_ramp_cost_gaps(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_ramp_vs_cost_gaps.png"),
        out_pdf=os.path.join(out_dir, "fig_ramp_vs_cost_gaps.pdf"),
    )
    plot_ramp_win_share(
        points=summary_points,
        out_png=os.path.join(out_dir, "fig_ramp_vs_win_share.png"),
        out_pdf=os.path.join(out_dir, "fig_ramp_vs_win_share.pdf"),
    )
    _write_path_outputs(raw_rows=raw_rows, out_dir=out_dir)

    print(f"\nsaved raw csv             -> {raw_csv}")
    print(f"saved summary-by-kappa   -> {summary_csv}")
    print(f"saved summary-by-path    -> {os.path.join(out_dir, 'ramp_capability_summary_by_path.csv')}")
    print(f"saved by-path folders    -> {os.path.join(out_dir, 'by_path')}")
    print(f"saved path contact sheet -> {os.path.join(out_dir, 'fig_total_cost_vs_kappa_contact_sheet.png')}")
    print(f"saved meta json          -> {meta_json}")
    print(f"saved overall figures    -> {out_dir}")


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

    # Experiment 1
    # run_tariff_path_sensitivity(seeds=seeds, cfg=cfg)

    # Experiment 2
    run_ramp_capability_sensitivity(
        seeds=seeds,
        cfg=cfg,
        # ramp_kappas=[0.35, 0.45, 0.55, 0.70, 0.90, 1.20],
        ramp_kappas=[0.384, 0.461, 0.576, 0.768, 1.151],

    )


if __name__ == "__main__":
    main()