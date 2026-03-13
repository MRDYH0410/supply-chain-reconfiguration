from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


QUARTER_PATTERN = re.compile(r"^(\d{4})\s*FQ([1-4])$")


@dataclass
class CalibrationArtifacts:
    full_series_musd: pd.Series
    train_series_musd: pd.Series
    actual_2025_musd: pd.Series
    predicted_2025_musd: pd.Series
    results_table: pd.DataFrame
    best_model_order: Tuple[int, int, int]
    best_seasonal_order: Tuple[int, int, int, int]
    best_aic: float


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Extract the Automotive row from the Excel file, normalize units to million USD, "
            "fit a quarterly SARIMA model on 2015 FQ4 to 2024 FQ4, forecast 2025 FQ1 to 2025 FQ4, "
            "and compute tariff-shock gaps as Predicted - Actual."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(script_dir / "more detailed.xlsx"),
        help="Path to the Excel file. Default is 'more detailed.xlsx' in the same folder as this script.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Sheet name. Default uses the first sheet.",
    )
    parser.add_argument(
        "--target-row-label",
        type=str,
        default="Automotive",
        help="Row label to extract. Default is 'Automotive'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir),
        help="Directory for CSV, JSON, TXT, and optional plot outputs. Default is the same folder as this script.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, save a line chart of actual vs predicted values.",
    )
    return parser.parse_args()


def find_row_by_first_col(raw: pd.DataFrame, target_text: str) -> int:
    target = str(target_text).strip().casefold()
    for idx in raw.index:
        value = raw.iat[idx, 0]
        if isinstance(value, str) and value.strip().casefold() == target:
            return int(idx)
    raise ValueError(f"Could not find row with first-column label '{target_text}'.")


def find_quarter_header_row(raw: pd.DataFrame, scan_rows: int = 20) -> int:
    best_row = None
    best_count = -1
    rows_to_scan = min(scan_rows, len(raw))
    for r in range(rows_to_scan):
        count = 0
        for c in range(raw.shape[1]):
            val = raw.iat[r, c]
            if isinstance(val, str) and QUARTER_PATTERN.match(val.strip()):
                count += 1
        if count > best_count:
            best_count = count
            best_row = r
    if best_row is None or best_count <= 0:
        raise ValueError("Could not detect a quarter header row.")
    return int(best_row)


def extract_quarter_columns(raw: pd.DataFrame, quarter_row: int) -> List[int]:
    cols: List[int] = []
    for c in range(raw.shape[1]):
        val = raw.iat[quarter_row, c]
        if isinstance(val, str) and QUARTER_PATTERN.match(val.strip()):
            cols.append(c)
    if not cols:
        raise ValueError("No quarter columns found.")
    return cols


def unit_to_million_factor(unit_label: object) -> float:
    if pd.isna(unit_label):
        return 1.0
    text = str(unit_label).strip().casefold()
    if text in {"millions", "million"}:
        return 1.0
    if text in {"thousands", "thousand"}:
        return 0.001
    if text in {"billions", "billion"}:
        return 1000.0
    raise ValueError(f"Unsupported unit label: {unit_label}")


def quarter_sort_key(label: str) -> Tuple[int, int]:
    m = QUARTER_PATTERN.match(label.strip())
    if not m:
        raise ValueError(f"Invalid quarter label: {label}")
    year = int(m.group(1))
    quarter = int(m.group(2))
    return year, quarter


def quarter_label_to_period(label: str) -> pd.Period:
    year, quarter = quarter_sort_key(label)
    return pd.Period(f"{year}Q{quarter}", freq="Q-DEC")


def extract_normalized_series(
    excel_path: Path,
    sheet_name: Optional[str],
    target_row_label: str,
) -> pd.Series:
    effective_sheet = 0 if sheet_name is None else sheet_name
    raw = pd.read_excel(excel_path, sheet_name=effective_sheet, header=None)

    quarter_row = find_quarter_header_row(raw)
    units_row = find_row_by_first_col(raw, "Units")
    target_row = find_row_by_first_col(raw, target_row_label)
    quarter_cols = extract_quarter_columns(raw, quarter_row)

    records = []
    for c in quarter_cols:
        q_label = str(raw.iat[quarter_row, c]).strip()
        unit_label = raw.iat[units_row, c]
        raw_value = raw.iat[target_row, c]
        if pd.isna(raw_value):
            continue
        factor = unit_to_million_factor(unit_label)
        value_musd = float(raw_value) * factor
        records.append((q_label, value_musd))

    if not records:
        raise ValueError(f"No data found for row '{target_row_label}'.")

    records.sort(key=lambda x: quarter_sort_key(x[0]))
    labels = [r[0] for r in records]
    values = [r[1] for r in records]

    series = pd.Series(values, index=pd.Index(labels, name="quarter"), name=target_row_label)
    series = series.astype(float)
    return series


def split_train_and_2025(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    train_labels = []
    test_labels = []

    for label in series.index:
        year, quarter = quarter_sort_key(label)
        if (year > 2015 or (year == 2015 and quarter >= 4)) and year <= 2024:
            train_labels.append(label)
        elif year == 2025:
            test_labels.append(label)

    expected_test = ["2025 FQ1", "2025 FQ2", "2025 FQ3", "2025 FQ4"]
    if test_labels != expected_test:
        raise ValueError(
            f"Expected 2025 test labels {expected_test}, but found {test_labels}."
        )

    train = series.loc[train_labels].copy()
    test = series.loc[test_labels].copy()

    if train.empty:
        raise ValueError("Training set is empty. Check the available quarters in the input file.")

    return train, test


def fit_best_sarima(
    train_series_musd: pd.Series,
) -> Tuple[object, Tuple[int, int, int], Tuple[int, int, int, int], float]:
    y = train_series_musd.copy()
    y.index = pd.period_range(
        start=quarter_label_to_period(y.index[0]),
        periods=len(y),
        freq="Q-DEC",
    )

    candidate_orders = [(p, 1, q) for p in (0, 1, 2) for q in (0, 1, 2)]
    candidate_seasonals = [(P, 1, Q, 4) for P in (0, 1) for Q in (0, 1)]

    best_fit = None
    best_order = None
    best_seasonal = None
    best_aic = np.inf

    for order in candidate_orders:
        for seasonal in candidate_seasonals:
            try:
                model = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seasonal,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False)
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_fit = fit
                    best_order = order
                    best_seasonal = seasonal
                    best_aic = float(fit.aic)
            except Exception:
                continue

    if best_fit is None:
        raise RuntimeError("All SARIMA candidates failed. Please inspect the input series.")

    return best_fit, best_order, best_seasonal, best_aic


def calibrate_automotive_tariff_shock(
    excel_path: Path,
    sheet_name: Optional[str] = None,
    target_row_label: str = "Automotive",
) -> CalibrationArtifacts:
    full_series_musd = extract_normalized_series(excel_path, sheet_name, target_row_label)
    train_musd, actual_2025_musd = split_train_and_2025(full_series_musd)

    fit, best_order, best_seasonal, best_aic = fit_best_sarima(train_musd)
    forecast_values = fit.get_forecast(steps=4).predicted_mean

    predicted_2025_musd = pd.Series(
        np.asarray(forecast_values, dtype=float),
        index=actual_2025_musd.index,
        name="predicted_musd",
    )

    results = pd.DataFrame(
        {
            "actual_musd": actual_2025_musd.astype(float),
            "predicted_musd": predicted_2025_musd.astype(float),
        }
    )
    results["gap_musd"] = results["predicted_musd"] - results["actual_musd"]
    results["impact_ratio"] = results["gap_musd"] / results["predicted_musd"]
    results["demand_multiplier_actual_over_predicted"] = (
        results["actual_musd"] / results["predicted_musd"]
    )
    results.index.name = "quarter"

    return CalibrationArtifacts(
        full_series_musd=full_series_musd,
        train_series_musd=train_musd,
        actual_2025_musd=actual_2025_musd,
        predicted_2025_musd=predicted_2025_musd,
        results_table=results,
        best_model_order=best_order,
        best_seasonal_order=best_seasonal,
        best_aic=best_aic,
    )


def build_story_case_patch_example(multiplier_by_quarter: Dict[str, float]) -> str:
    fq1 = multiplier_by_quarter.get("2025 FQ1", 1.0)
    fq2 = multiplier_by_quarter.get("2025 FQ2", 1.0)
    fq3 = multiplier_by_quarter.get("2025 FQ3", 1.0)
    fq4 = multiplier_by_quarter.get("2025 FQ4", 1.0)

    return f"""# --------------------------
# Option A. Minimal manual use in story_case.py
# --------------------------
# If one model period = one quarter, you can turn scalar d0 into a 20-period series.
# Example for D1, using the 2025 quarterly shock multipliers as a repeating cycle.
# This changes baseline demand directly and does NOT require editing scenario.py.

base_d0_D1 = 140.0
quarterly_multiplier = [{fq1:.6f}, {fq2:.6f}, {fq3:.6f}, {fq4:.6f}]
d0_D1 = [base_d0_D1 * quarterly_multiplier[t % 4] for t in range(20)]

base_d0_D2 = 80.0
d0_D2 = [base_d0_D2 * quarterly_multiplier[t % 4] for t in range(20)]

# Then replace in story_case.py
# "D1": MarketParams("D1", d0=d0_D1, eta=0.2, cu=15.0, price=30.0, demand_floor=0.90)
# "D2": MarketParams("D2", d0=d0_D2, eta=0.1, cu=10.0, price=28.0, demand_floor=0.92)

# --------------------------
# Option B. More rigorous use later
# --------------------------
# Keep d0 unchanged, and add an external tariff shock multiplier in scenario.py
# potential_demand = original_potential_demand * quarterly_multiplier[current_quarter]
# I recommend this option for the next step because it preserves your current eta mechanism.
"""


def write_outputs(
    artifacts: CalibrationArtifacts,
    output_dir: Path,
    make_plot: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    full_series_df = artifacts.full_series_musd.reset_index()
    full_series_df.columns = ["quarter", "automotive_musd"]
    full_series_df.to_csv(output_dir / "automotive_full_series_musd.csv", index=False)

    results_rounded = artifacts.results_table.copy()
    for col in results_rounded.columns:
        results_rounded[col] = results_rounded[col].astype(float).round(6)
    results_rounded.reset_index().to_csv(
        output_dir / "automotive_2025_tariff_shock_results.csv",
        index=False,
    )

    summary = {
        "model": {
            "type": "SARIMA",
            "order": list(artifacts.best_model_order),
            "seasonal_order": list(artifacts.best_seasonal_order),
            "aic": round(float(artifacts.best_aic), 6),
        },
        "train_window": [
            artifacts.train_series_musd.index[0],
            artifacts.train_series_musd.index[-1],
        ],
        "forecast_window": list(artifacts.actual_2025_musd.index),
        "quarterly_results": {
            quarter: {
                "actual_musd": round(float(row.actual_musd), 6),
                "predicted_musd": round(float(row.predicted_musd), 6),
                "gap_musd": round(float(row.gap_musd), 6),
                "impact_ratio": round(float(row.impact_ratio), 6),
                "demand_multiplier_actual_over_predicted": round(
                    float(row.demand_multiplier_actual_over_predicted), 6
                ),
            }
            for quarter, row in artifacts.results_table.iterrows()
        },
        "annual_summary": {
            "actual_2025_musd": round(float(artifacts.results_table["actual_musd"].sum()), 6),
            "predicted_2025_musd": round(float(artifacts.results_table["predicted_musd"].sum()), 6),
            "gap_2025_musd": round(float(artifacts.results_table["gap_musd"].sum()), 6),
            "impact_ratio_2025": round(
                float(
                    artifacts.results_table["gap_musd"].sum()
                    / artifacts.results_table["predicted_musd"].sum()
                ),
                6,
            ),
            "average_multiplier_actual_over_predicted": round(
                float(
                    artifacts.results_table["actual_musd"].sum()
                    / artifacts.results_table["predicted_musd"].sum()
                ),
                6,
            ),
        },
        "paste_ready_for_model": {
            "quarterly_multiplier": {
                quarter: round(float(val), 6)
                for quarter, val in artifacts.results_table[
                    "demand_multiplier_actual_over_predicted"
                ].items()
            },
            "quarterly_gap_musd": {
                quarter: round(float(val), 6)
                for quarter, val in artifacts.results_table["gap_musd"].items()
            },
        },
    }

    with open(output_dir / "automotive_2025_tariff_shock_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output_dir / "story_case_patch_example.txt", "w", encoding="utf-8") as f:
        f.write(build_story_case_patch_example(summary["paste_ready_for_model"]["quarterly_multiplier"]))

    if make_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        hist = artifacts.train_series_musd.copy()
        hist.index = [str(q) for q in hist.index]
        actual = artifacts.actual_2025_musd.copy()
        pred = artifacts.predicted_2025_musd.copy()

        ax.plot(hist.index, hist.values, marker="o", label="Train actual (2015 FQ4 - 2024 FQ4)")
        ax.plot(actual.index, actual.values, marker="o", label="2025 actual")
        ax.plot(pred.index, pred.values, marker="o", linestyle="--", label="2025 forecast")

        ax.set_title("Automotive revenue series and 2025 counterfactual forecast")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Million USD")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "automotive_2025_tariff_shock_plot.png", dpi=200)
        plt.close(fig)


def print_console_summary(artifacts: CalibrationArtifacts) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    print("=" * 80)
    print("Automotive tariff-shock calibration complete")
    print("=" * 80)
    print(f"Training window: {artifacts.train_series_musd.index[0]} to {artifacts.train_series_musd.index[-1]}")
    print(f"Forecast window: {artifacts.actual_2025_musd.index[0]} to {artifacts.actual_2025_musd.index[-1]}")
    print(
        f"Selected SARIMA order={artifacts.best_model_order}, "
        f"seasonal_order={artifacts.best_seasonal_order}, AIC={artifacts.best_aic:.3f}"
    )
    print()
    print("2025 results in million USD per quarter")
    print(artifacts.results_table.round(6))
    print()

    annual_gap = artifacts.results_table["gap_musd"].sum()
    annual_pred = artifacts.results_table["predicted_musd"].sum()
    annual_act = artifacts.results_table["actual_musd"].sum()
    annual_ratio = annual_gap / annual_pred
    annual_multiplier = annual_act / annual_pred

    print("Annual summary")
    print(f"  Predicted 2025 total = {annual_pred:.6f} musd")
    print(f"  Actual 2025 total    = {annual_act:.6f} musd")
    print(f"  Gap = Pred - Actual  = {annual_gap:.6f} musd")
    print(f"  Impact ratio         = {annual_ratio:.6%}")
    print(f"  Avg multiplier       = {annual_multiplier:.6f}")
    print("=" * 80)


def main() -> None:
    args = parse_args()

    excel_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {excel_path}\n"
            f"Please make sure 'more detailed.xlsx' is in the same folder as this script, "
            f"or pass --input manually."
        )

    artifacts = calibrate_automotive_tariff_shock(
        excel_path=excel_path,
        sheet_name=args.sheet,
        target_row_label=args.target_row_label,
    )

    write_outputs(artifacts, output_dir=output_dir, make_plot=args.plot)
    print_console_summary(artifacts)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()