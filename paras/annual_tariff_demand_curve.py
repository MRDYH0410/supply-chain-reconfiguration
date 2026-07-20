
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annual_tariff_demand_curve.py

Purpose
-------
Build a single annual empirical tariff -> demand-retention curve for Tesla
using:
1) quarterly Automotive revenue from "more detailed.xlsx"
2) monthly tariff schedule supplied by the user
3) a counterfactual forecasting step trained on 2015-2024 data
4) monthly 2025 raw retention samples: actual / counterfactual
5) curve fitting on the 2025 monthly samples

Important design choice
-----------------------
- Monthly data are INPUTS used to enlarge the effective sample for forecasting.
- The FINAL OUTPUT is NOT a month-by-month factor series for the main model.
- The final output is a SINGLE empirical relationship:
      demand_retention = f(tariff)
  that can be evaluated at any tariff level in the main model.

Data assumptions
----------------
- The Excel file is located in the same folder as this script and is named:
      more detailed.xlsx
- The row "Automotive" under the "Revenues" section is used.
- Quarterly demand is evenly split across the 3 months of each quarter.
- Tariff schedule is implemented according to the user-provided rules.
- For 2025-04 (T4), the user gave a range 154%-254%.
  The script uses 154% by default and exposes a parameter to change it.

Outputs
-------
Saved under ./tariff_demand_curve_outputs/
- monthly_panel.csv
- model_screening.csv
- monthly_raw_samples_2025.csv
- tariff_curve_screening.csv
- tariff_curve_grid.csv
- tariff_demand_function.json
- run_summary.json

Usage
-----
python annual_tariff_demand_curve.py
python annual_tariff_demand_curve.py --excel "more detailed.xlsx" --t4-rate 154
python annual_tariff_demand_curve.py --excel "more detailed.xlsx" --t4-rate 254

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# ---------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.where(denom == 0.0, 0.0, np.abs(y_true - y_pred) / denom)
    return float(np.mean(out) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------
# Excel ingestion
# ---------------------------------------------------------------------

def load_quarterly_automotive_series(excel_path: str) -> pd.DataFrame:
    """
    Reads the uploaded 'more detailed.xlsx' file and extracts the quarterly
    Automotive revenue series.

    Returns a DataFrame with columns:
    - quarter_label
    - quarter_end
    - value_musd   (million USD)
    """
    raw = pd.read_excel(excel_path, sheet_name=0, header=None)
    first_col = raw.iloc[:, 0].astype(str)

    # Row containing the quarter labels across columns
    period_row_candidates = first_col[first_col.str.contains("Recommended:", na=False)].index.tolist()
    if not period_row_candidates:
        raise ValueError("Could not find the row containing quarterly period labels.")
    period_row = period_row_candidates[0]

    # Rows for Units and the Automotive series
    unit_row_candidates = first_col[first_col.eq("Units")].index.tolist()
    if not unit_row_candidates:
        raise ValueError("Could not find the 'Units' row.")
    unit_row = unit_row_candidates[0]

    revenue_row_candidates = first_col[first_col.eq("Revenues")].index.tolist()
    if not revenue_row_candidates:
        raise ValueError("Could not find the 'Revenues' section row.")

    # Take the first Automotive row after the first Revenues row
    auto_candidates = []
    for idx in first_col[first_col.eq("Automotive")].index.tolist():
        if idx > revenue_row_candidates[0]:
            auto_candidates.append(idx)
    if not auto_candidates:
        raise ValueError("Could not find the Automotive revenue row.")
    auto_row = auto_candidates[0]

    # Quarter-end date row normally sits right below the period labels
    quarter_end_row = period_row + 1

    quarter_labels = raw.iloc[period_row, 1:].tolist()
    quarter_ends = raw.iloc[quarter_end_row, 1:].tolist()
    units = raw.iloc[unit_row, 1:].tolist()
    values = raw.iloc[auto_row, 1:].tolist()

    out_rows = []
    for qlab, qend, unit, val in zip(quarter_labels, quarter_ends, units, values):
        if pd.isna(qlab) or pd.isna(val):
            continue
        qlab = str(qlab).strip()
        if "FQ" not in qlab:
            continue

        value = float(val)
        unit_str = "" if pd.isna(unit) else str(unit).strip().lower()
        if unit_str.startswith("thousand"):
            value_musd = value / 1000.0
        elif unit_str.startswith("million"):
            value_musd = value
        else:
            # Default fallback: assume already in millions if later periods,
            # but do not silently guess without warning.
            raise ValueError(f"Unrecognized unit '{unit}' for quarter {qlab}.")

        out_rows.append(
            {
                "quarter_label": qlab,
                "quarter_end": pd.to_datetime(qend),
                "value_musd": value_musd,
            }
        )

    out = pd.DataFrame(out_rows).sort_values("quarter_end").reset_index(drop=True)
    if out.empty:
        raise ValueError("Automotive quarterly series could not be constructed.")
    return out


def quarter_to_monthly_equal_split(qdf: pd.DataFrame) -> pd.DataFrame:
    """
    Split each quarterly value equally into the 3 months of that quarter.
    The monthly timestamp is the month start.
    """
    rows = []
    for _, r in qdf.iterrows():
        q_end = pd.Timestamp(r["quarter_end"])
        q_label = r["quarter_label"]
        total = float(r["value_musd"])
        monthly_value = total / 3.0
        months = [q_end - pd.offsets.MonthEnd(2),
                  q_end - pd.offsets.MonthEnd(1),
                  q_end]
        for m_end in months:
            m_start = pd.Timestamp(m_end.year, m_end.month, 1)
            rows.append(
                {
                    "month": m_start,
                    "quarter_label": q_label,
                    "quarter_end": q_end,
                    "actual_monthly_musd": monthly_value,
                }
            )
    mdf = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    return mdf


# ---------------------------------------------------------------------
# Tariff schedule
# ---------------------------------------------------------------------

def build_monthly_tariff_schedule(months: pd.Series, t4_rate_percent: float = 154.0) -> pd.DataFrame:
    """
    Build the monthly tariff series using the user's schedule.
    Rates are stored both in percent and decimal.

    Interpreted schedule:
    - 2015-01 through 2018-04: 2.5%
    - 2018-05 through 2024-04: 27.5%
    - 2024-05 through 2024-12: 102.5%
    - 2025-01: 100%
    - 2025-02: 110%
    - 2025-03: 120%
    - 2025-04: T4 = 154% or 254% (parameterized)
    - 2025-05 through 2025-12: 110%

    Note:
    The user provided T13-T14 = 115% for 2026. This script's calibration
    window ends at 2025-12, so that portion is not used here.
    """
    rows = []
    for m in months:
        m = pd.Timestamp(m)
        if m < pd.Timestamp("2018-05-01"):
            rate_pct = 2.5
        elif m < pd.Timestamp("2024-05-01"):
            rate_pct = 27.5
        elif m < pd.Timestamp("2025-01-01"):
            rate_pct = 102.5
        else:
            if m.year != 2025:
                raise ValueError(f"Unexpected calibration month outside 2025 in tariff tail: {m}")
            mapping_2025 = {
                1: 100.0,
                2: 110.0,
                3: 120.0,
                4: float(t4_rate_percent),
                5: 110.0,
                6: 110.0,
                7: 110.0,
                8: 110.0,
                9: 110.0,
                10: 110.0,
                11: 110.0,
                12: 110.0,
            }
            rate_pct = mapping_2025[m.month]

        rows.append({"month": m, "tariff_pct": rate_pct, "tariff_decimal": rate_pct / 100.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Forecasting features and recursive forecasting
# ---------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month_num"] = out["month"].dt.month
    out["year_num"] = out["month"].dt.year
    out["time_idx"] = np.arange(len(out))
    # cyclic month encoding
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12.0)
    return out


def make_supervised_frame(series_df: pd.DataFrame, target_col: str, max_lag: int = 12) -> pd.DataFrame:
    out = series_df[["month", target_col, "month_num", "year_num", "time_idx", "month_sin", "month_cos"]].copy()
    for lag in range(1, max_lag + 1):
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    out["rolling_mean_3"] = out[target_col].shift(1).rolling(3).mean()
    out["rolling_mean_6"] = out[target_col].shift(1).rolling(6).mean()
    out["rolling_mean_12"] = out[target_col].shift(1).rolling(12).mean()
    out = out.dropna().reset_index(drop=True)
    return out


FEATURE_COLS = [
    "month_num",
    "year_num",
    "time_idx",
    "month_sin",
    "month_cos",
] + [f"lag_{i}" for i in range(1, 13)] + ["rolling_mean_3", "rolling_mean_6", "rolling_mean_12"]


def fit_linear_like_model(model_name: str):
    if model_name == "ridge_lag_calendar":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        )
    if model_name == "elasticnet_lag_calendar":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)),
            ]
        )
    if model_name == "random_forest_lag_calendar":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
        )
    if model_name == "gradient_boosting_lag_calendar":
        return GradientBoostingRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=3,
            loss="squared_error",
        )
    raise ValueError(f"Unknown model: {model_name}")


def recursive_forecast_with_lags(
    history_df: pd.DataFrame,
    forecast_months: pd.Series,
    model_name: str,
    target_col: str = "actual_monthly_musd",
) -> pd.DataFrame:
    """
    Fit once on available history and recursively forecast the next months.
    """
    hist = add_calendar_features(history_df.copy())
    supervised = make_supervised_frame(hist, target_col=target_col, max_lag=12)

    X_train = supervised[FEATURE_COLS].values
    y_train = supervised[target_col].values

    model = fit_linear_like_model(model_name)
    model.fit(X_train, y_train)

    # Work on a copy of the history target sequence
    values = hist[["month", target_col]].copy()
    values = values.set_index("month")[target_col].astype(float).to_dict()

    preds = []
    # We also need time_idx continuation
    last_time_idx = int(hist["time_idx"].iloc[-1])

    for step, m in enumerate(pd.to_datetime(forecast_months)):
        month_num = int(m.month)
        year_num = int(m.year)
        time_idx = last_time_idx + step + 1
        month_sin = math.sin(2 * math.pi * month_num / 12.0)
        month_cos = math.cos(2 * math.pi * month_num / 12.0)

        lag_vals = []
        for lag in range(1, 13):
            lag_month = (m - pd.DateOffset(months=lag)).replace(day=1)
            if lag_month not in values:
                raise ValueError(f"Missing lag value for {lag_month} while forecasting {m}.")
            lag_vals.append(float(values[lag_month]))

        feat = {
            "month_num": month_num,
            "year_num": year_num,
            "time_idx": time_idx,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
        for i, v in enumerate(lag_vals, start=1):
            feat[f"lag_{i}"] = v
        feat["rolling_mean_3"] = float(np.mean(lag_vals[:3]))
        feat["rolling_mean_6"] = float(np.mean(lag_vals[:6]))
        feat["rolling_mean_12"] = float(np.mean(lag_vals[:12]))

        x = np.array([[feat[c] for c in FEATURE_COLS]])
        pred = float(model.predict(x)[0])
        pred = max(pred, 1e-6)

        preds.append({"month": m, "forecast": pred})
        values[m] = pred

    return pd.DataFrame(preds)


def seasonal_naive_forecast(history_df: pd.DataFrame, forecast_months: pd.Series, target_col: str) -> pd.DataFrame:
    values = history_df.set_index("month")[target_col].astype(float)
    preds = []
    for m in pd.to_datetime(forecast_months):
        ref = (m - pd.DateOffset(years=1)).replace(day=1)
        if ref not in values.index:
            raise ValueError(f"Seasonal naive missing reference month {ref} for forecast month {m}.")
        preds.append({"month": m, "forecast": float(values.loc[ref])})
    return pd.DataFrame(preds)


def ets_forecast(history_df: pd.DataFrame, forecast_months: pd.Series, target_col: str) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels is not available, ETS forecast cannot run.")
    y = history_df.set_index("month")[target_col].astype(float)
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    fc = fit.forecast(len(forecast_months))
    return pd.DataFrame({"month": pd.to_datetime(forecast_months), "forecast": fc.values})


def screen_forecast_models(monthly_df: pd.DataFrame, target_col: str = "actual_monthly_musd") -> Tuple[pd.DataFrame, str]:
    """
    Validation design:
    - Train window: start through 2023-12
    - Validation window: 2024-01 through 2024-12
    This keeps 2025 untouched for final counterfactual forecasting.
    """
    train_cutoff = pd.Timestamp("2023-12-01")
    val_start = pd.Timestamp("2024-01-01")
    val_end = pd.Timestamp("2024-12-01")

    train_hist = monthly_df[monthly_df["month"] <= train_cutoff].copy()
    val_df = monthly_df[(monthly_df["month"] >= val_start) & (monthly_df["month"] <= val_end)].copy()
    val_months = val_df["month"]

    candidates = ["seasonal_naive", "ridge_lag_calendar", "elasticnet_lag_calendar",
                  "random_forest_lag_calendar", "gradient_boosting_lag_calendar"]
    if HAS_STATSMODELS:
        candidates.append("ets_additive")

    rows = []

    for name in candidates:
        try:
            if name == "seasonal_naive":
                pred_df = seasonal_naive_forecast(train_hist, val_months, target_col=target_col)
            elif name == "ets_additive":
                pred_df = ets_forecast(train_hist, val_months, target_col=target_col)
            else:
                pred_df = recursive_forecast_with_lags(train_hist, val_months, model_name=name, target_col=target_col)

            merged = val_df[["month", target_col]].merge(pred_df, on="month", how="left")
            y_true = merged[target_col].values
            y_pred = merged["forecast"].values
            rows.append(
                {
                    "model": name,
                    "validation_rmse": rmse(y_true, y_pred),
                    "validation_smape_pct": smape(y_true, y_pred),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "model": name,
                    "validation_rmse": np.nan,
                    "validation_smape_pct": np.nan,
                    "error": str(e),
                }
            )

    score_df = pd.DataFrame(rows)
    score_df = score_df.sort_values(["validation_smape_pct", "validation_rmse"], na_position="last").reset_index(drop=True)
    best_row = score_df.dropna(subset=["validation_smape_pct"]).iloc[0]
    best_model = str(best_row["model"])
    return score_df, best_model


def forecast_counterfactual_2025(monthly_df: pd.DataFrame, best_model: str, target_col: str = "actual_monthly_musd") -> pd.DataFrame:
    """
    Fit the selected model on 2015-10 to 2024-12 and forecast 2025-01 to 2025-12.
    """
    hist = monthly_df[monthly_df["month"] <= pd.Timestamp("2024-12-01")].copy()
    future_months = pd.date_range("2025-01-01", "2025-12-01", freq="MS")

    if best_model == "seasonal_naive":
        fc = seasonal_naive_forecast(hist, future_months, target_col=target_col)
    elif best_model == "ets_additive":
        fc = ets_forecast(hist, future_months, target_col=target_col)
    else:
        fc = recursive_forecast_with_lags(hist, future_months, model_name=best_model, target_col=target_col)

    fc = fc.rename(columns={"forecast": "counterfactual_monthly_musd"})
    return fc


# ---------------------------------------------------------------------
# Curve fitting: demand retention = f(tariff)
# ---------------------------------------------------------------------

def fit_isotonic_curve(x: np.ndarray, y: np.ndarray) -> Dict:
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(x, y)
    yhat = model.predict(x)
    return {
        "name": "isotonic_monotone",
        "predictor": lambda z: np.asarray(model.predict(np.asarray(z, dtype=float))),
        "in_sample_rmse": rmse(y, yhat),
        "params": {},
    }


def exp_curve(x: np.ndarray, eta: float) -> np.ndarray:
    return np.exp(-eta * x)


def fit_exponential_curve(x: np.ndarray, y: np.ndarray) -> Dict:
    popt, _ = curve_fit(
        exp_curve,
        x,
        y,
        p0=np.array([0.5]),
        bounds=([0.0], [20.0]),
        maxfev=20000,
    )
    eta = float(popt[0])
    yhat = exp_curve(x, eta)
    return {
        "name": "exponential",
        "predictor": lambda z: exp_curve(np.asarray(z, dtype=float), eta),
        "in_sample_rmse": rmse(y, yhat),
        "params": {"eta": eta},
    }


def floor_exp_curve(x: np.ndarray, floor_: float, eta: float) -> np.ndarray:
    return floor_ + (1.0 - floor_) * np.exp(-eta * x)


def fit_floor_exponential_curve(x: np.ndarray, y: np.ndarray) -> Dict:
    popt, _ = curve_fit(
        floor_exp_curve,
        x,
        y,
        p0=np.array([0.2, 0.5]),
        bounds=([0.0, 0.0], [1.0, 20.0]),
        maxfev=50000,
    )
    floor_ = float(popt[0])
    eta = float(popt[1])
    yhat = floor_exp_curve(x, floor_, eta)
    return {
        "name": "floor_exponential",
        "predictor": lambda z: floor_exp_curve(np.asarray(z, dtype=float), floor_, eta),
        "in_sample_rmse": rmse(y, yhat),
        "params": {"floor": floor_, "eta": eta},
    }


def linear_clip_curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(a + b * x, 0.0, 1.0)


def fit_linear_clipped_curve(x: np.ndarray, y: np.ndarray) -> Dict:
    # Monotone decreasing enforced via b <= 0
    popt, _ = curve_fit(
        lambda z, a, b: linear_clip_curve(z, a, b),
        x,
        y,
        p0=np.array([1.0, -0.1]),
        bounds=([0.0, -10.0], [1.5, 0.0]),
        maxfev=20000,
    )
    a = float(popt[0])
    b = float(popt[1])
    yhat = linear_clip_curve(x, a, b)
    return {
        "name": "linear_clipped",
        "predictor": lambda z: linear_clip_curve(np.asarray(z, dtype=float), a, b),
        "in_sample_rmse": rmse(y, yhat),
        "params": {"a": a, "b": b},
    }


def loocv_rmse(x: np.ndarray, y: np.ndarray, fit_func: Callable[[np.ndarray, np.ndarray], Dict]) -> float:
    preds = []
    truths = []
    n = len(x)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xtr, ytr = x[mask], y[mask]
        xte, yte = x[~mask], y[~mask]
        try:
            fitted = fit_func(xtr, ytr)
            pred = float(np.asarray(fitted["predictor"](xte))[0])
            pred = float(np.clip(pred, 0.0, 1.0))
        except Exception:
            return np.nan
        preds.append(pred)
        truths.append(float(yte[0]))
    return rmse(np.asarray(truths), np.asarray(preds))


def screen_tariff_demand_curves(raw_samples_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    fitters = [
        fit_isotonic_curve,
        fit_exponential_curve,
        fit_floor_exponential_curve,
        fit_linear_clipped_curve,
    ]

    x = raw_samples_df["tariff_decimal"].values.astype(float)
    y = raw_samples_df["raw_retention_factor"].values.astype(float)

    rows = []
    fitted_objects = {}

    for fitter in fitters:
        try:
            fitted = fitter(x, y)
            cv = loocv_rmse(x, y, fitter)
            fitted_objects[fitted["name"]] = fitted
            rows.append(
                {
                    "curve_model": fitted["name"],
                    "loocv_rmse": cv,
                    "in_sample_rmse": fitted["in_sample_rmse"],
                    "params_json": json.dumps(fitted["params"], ensure_ascii=False),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "curve_model": getattr(fitter, "__name__", "unknown"),
                    "loocv_rmse": np.nan,
                    "in_sample_rmse": np.nan,
                    "params_json": json.dumps({"error": str(e)}, ensure_ascii=False),
                }
            )

    score_df = pd.DataFrame(rows).sort_values(["loocv_rmse", "in_sample_rmse"], na_position="last").reset_index(drop=True)
    best_name = str(score_df.dropna(subset=["loocv_rmse"]).iloc[0]["curve_model"])
    best_obj = fitted_objects[best_name]
    return score_df, best_obj


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_pipeline(excel_path: str, output_dir: str, t4_rate_percent: float = 154.0) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 1) Read quarterly Automotive series
    quarterly = load_quarterly_automotive_series(excel_path)

    # 2) Equal split to monthly pseudo-demand
    monthly = quarter_to_monthly_equal_split(quarterly)

    # 3) Attach monthly tariff schedule
    tariff_df = build_monthly_tariff_schedule(monthly["month"], t4_rate_percent=t4_rate_percent)
    monthly = monthly.merge(tariff_df, on="month", how="left")
    monthly = monthly.sort_values("month").reset_index(drop=True)

    # 4) Forecast-model screening using 2024 as validation
    model_screening, best_forecast_model = screen_forecast_models(monthly, target_col="actual_monthly_musd")

    # 5) Counterfactual 2025 monthly forecast using 2015-2024 training
    fc2025 = forecast_counterfactual_2025(monthly, best_forecast_model, target_col="actual_monthly_musd")

    # 6) Build 2025 raw retention samples
    actual_2025 = monthly[(monthly["month"] >= pd.Timestamp("2025-01-01")) &
                          (monthly["month"] <= pd.Timestamp("2025-12-01"))].copy()
    samples = actual_2025.merge(fc2025, on="month", how="left")
    samples["raw_retention_factor"] = samples["actual_monthly_musd"] / samples["counterfactual_monthly_musd"]
    samples["raw_retention_factor"] = samples["raw_retention_factor"].clip(upper=1.0)

    # 7) Fit annual tariff -> demand curve on 2025 monthly samples
    curve_screening, best_curve = screen_tariff_demand_curves(samples)

    # 8) Export curve grid for downstream use
    x_min = min(monthly["tariff_decimal"].min(), samples["tariff_decimal"].min())
    x_max = max(monthly["tariff_decimal"].max(), samples["tariff_decimal"].max())
    grid = np.linspace(x_min, x_max, 300)
    grid_pred = np.asarray(best_curve["predictor"](grid), dtype=float)
    grid_df = pd.DataFrame(
        {
            "tariff_decimal": grid,
            "tariff_pct": grid * 100.0,
            "predicted_retention_factor": np.clip(grid_pred, 0.0, 1.0),
        }
    )

    # 9) Save outputs
    monthly.to_csv(os.path.join(output_dir, "monthly_panel.csv"), index=False)
    model_screening.to_csv(os.path.join(output_dir, "model_screening.csv"), index=False)
    samples.to_csv(os.path.join(output_dir, "monthly_raw_samples_2025.csv"), index=False)
    curve_screening.to_csv(os.path.join(output_dir, "tariff_curve_screening.csv"), index=False)
    grid_df.to_csv(os.path.join(output_dir, "tariff_curve_grid.csv"), index=False)

    function_payload = {
        "selected_forecast_model": best_forecast_model,
        "selected_curve_model": best_curve["name"],
        "curve_params": best_curve["params"],
        "interpretation": "demand_retention = f(tariff_decimal)",
        "tariff_unit_note": "tariff_decimal uses 1.10 for 110%, 0.275 for 27.5%, etc.",
        "counterfactual_logic": "2015-2024 monthly pseudo-demand used to forecast 2025 no-new-intervention path; 2025 actual/predicted capped at 1 defines raw retention samples.",
        "t4_rate_percent_used": float(t4_rate_percent),
    }
    with open(os.path.join(output_dir, "tariff_demand_function.json"), "w", encoding="utf-8") as f:
        json.dump(function_payload, f, ensure_ascii=False, indent=2)

    summary = {
        "excel_path": os.path.abspath(excel_path),
        "output_dir": os.path.abspath(output_dir),
        "quarterly_obs_count": int(len(quarterly)),
        "monthly_obs_count": int(len(monthly)),
        "training_window": "2015-10 to 2024-12",
        "validation_window": "2024-01 to 2024-12",
        "forecast_window": "2025-01 to 2025-12",
        "best_forecast_model": best_forecast_model,
        "best_curve_model": best_curve["name"],
        "curve_params": best_curve["params"],
        "t4_rate_percent_used": float(t4_rate_percent),
        "raw_sample_rows": int(len(samples)),
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--excel",
        type=str,
        default="more detailed.xlsx",
        help="Path to the Excel file. Default: more detailed.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tariff_demand_curve_outputs",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--t4-rate",
        type=float,
        default=154.0,
        help="Tariff rate for 2025-04 (T4) in percent. Use 154 or 254.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        excel_path=args.excel,
        output_dir=args.output_dir,
        t4_rate_percent=args.t4_rate,
    )
