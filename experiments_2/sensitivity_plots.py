from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "A": "#183A5B",   # deep navy
    "B": "#1E8A7A",   # restrained teal
    "C": "#B65E2A",   # muted terracotta
    "gap": "#7F5A9A",
}
GRID_COLOR = "#D4DCE3"
TEXT_DARK = "#24323F"
TEXT_MUTED = "#5E6B77"
PHASE_SHADE_1 = "#F3F6F9"
PHASE_SHADE_2 = "#EEF3F8"
PHASE_SHADE_3 = "#F8FAFC"


@dataclass
class LinePlotSpec:
    path_id: str
    path_label: str
    meanA: List[float]
    meanB: List[float]
    meanC: List[float]
    stdA: List[float]
    stdB: List[float]
    stdC: List[float]
    totalA: float
    totalB: float
    totalC: float
    phase1_periods: Tuple[int, int]
    phase2_periods: Tuple[int, int]
    phase3_periods: Tuple[int, int]
    change_period_1: int
    change_period_2: int


@dataclass
class RampSummaryPoint:
    ramp_kappa: float
    ramp_full_age: int
    phase2_length: int
    mean_total_cost_A: float
    mean_total_cost_B: float
    mean_total_cost_C: float
    phase2_mean_cost_A: float
    phase2_mean_cost_B: float
    phase2_mean_cost_C: float
    mean_gap_B_minus_A: float
    mean_gap_C_minus_A: float
    mean_gap_C_minus_B: float
    phase2_gap_B_minus_A: float
    phase2_gap_C_minus_A: float
    win_share_A: float
    win_share_B: float
    win_share_C: float


@dataclass
class PathRampPoint:
    path_id: str
    path_label: str
    ramp_kappa: float
    phase2_length: int
    phase2_start: int
    phase3_start: int
    mean_cost_A: float
    mean_cost_B: float
    mean_cost_C: float
    std_cost_A: float
    std_cost_B: float
    std_cost_C: float
    phase2_mean_cost_A: float
    phase2_mean_cost_B: float
    phase2_mean_cost_C: float
    gap_B_minus_A: float
    gap_C_minus_A: float
    best_strategy: str


@dataclass
class PathContactSheetSpec:
    path_id: str
    path_label: str
    points: List[PathRampPoint]


def _set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.7,
        "ytick.labelsize": 8.7,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def _apply_axes(ax, ygrid: bool = True) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3, width=0.7, colors=TEXT_DARK)
    if ygrid:
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.6)
    else:
        ax.grid(False)


def _format_path_title(path_id: str, path_label: str) -> str:
    return f"{path_id}   {path_label.replace('->', '→')}"


def _add_phase_shading(ax, spec: LinePlotSpec, ymax: float) -> None:
    phase_specs = [
        (spec.phase1_periods, PHASE_SHADE_1, "Phase I"),
        (spec.phase2_periods, PHASE_SHADE_2, "Phase II"),
        (spec.phase3_periods, PHASE_SHADE_3, "Phase III"),
    ]
    for (s, e), color, label in phase_specs:
        if s <= e:
            ax.axvspan(s - 0.5, e + 0.5, color=color, zorder=0)
            ax.text((s + e) / 2.0, ymax * 0.985, label, ha="center", va="top", fontsize=8.5, color=TEXT_MUTED)

    ax.axvline(spec.phase2_periods[0] - 0.5, linestyle="--", linewidth=0.9, color="#7BA7D7")
    ax.axvline(spec.phase3_periods[0] - 0.5, linestyle="--", linewidth=0.9, color="#7BA7D7")
    ax.axvline(spec.change_period_1 - 0.5, linestyle=":", linewidth=1.0, color="#7BA7D7")
    ax.axvline(spec.change_period_2 - 0.5, linestyle=":", linewidth=1.0, color="#7BA7D7")
    ax.text(spec.change_period_1, ymax * 0.92, f"update 1\nt={spec.change_period_1}", ha="center", va="top", fontsize=7.4, color=TEXT_MUTED)
    ax.text(spec.change_period_2, ymax * 0.92, f"update 2\nt={spec.change_period_2}", ha="center", va="top", fontsize=7.4, color=TEXT_MUTED)


def _plot_strategy_series(ax, x, mean, std, key: str, label: str, show_band: bool = True) -> None:
    ax.plot(
        x, mean,
        color=PALETTE[key],
        linewidth=1.9,
        marker={"A": "o", "B": "s", "C": "^"}[key],
        markersize=4.6,
        markeredgewidth=0.5,
        label=label,
        zorder=3,
    )
    if show_band:
        ax.fill_between(x, mean - std, mean + std, color=PALETTE[key], alpha=0.10, linewidth=0.0, zorder=2)


def _ensure_dir_pair(out_png: str, out_pdf: str) -> None:
    png_dir = os.path.dirname(out_png)
    pdf_dir = os.path.dirname(out_pdf)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
    if pdf_dir and pdf_dir != png_dir:
        os.makedirs(pdf_dir, exist_ok=True)


# ---------------------------------------------------------------------
# Experiment 1 line plots kept as line charts
# ---------------------------------------------------------------------
def plot_single_tariff_path_lineplot(spec: LinePlotSpec, out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    H = len(spec.meanA)
    x = np.arange(1, H + 1)
    A = np.asarray(spec.meanA, dtype=float)
    B = np.asarray(spec.meanB, dtype=float)
    C = np.asarray(spec.meanC, dtype=float)
    sA = np.asarray(spec.stdA, dtype=float)
    sB = np.asarray(spec.stdB, dtype=float)
    sC = np.asarray(spec.stdC, dtype=float)

    ymax = max(A.max() + sA.max(), B.max() + sB.max(), C.max() + sC.max()) * 1.10
    ymax = max(ymax, 1.0)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    _apply_axes(ax)
    _add_phase_shading(ax, spec, ymax)
    _plot_strategy_series(ax, x, A, sA, "A", f"Strategy A  total={spec.totalA:.1f}")
    _plot_strategy_series(ax, x, B, sB, "B", f"Strategy B  total={spec.totalB:.1f}")
    _plot_strategy_series(ax, x, C, sC, "C", f"Strategy C  total={spec.totalC:.1f}")
    ax.set_xlim(1, H)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([1, 10, H])
    ax.set_xlabel("Time period")
    ax.set_ylabel("Mean period cost")
    ax.set_title(_format_path_title(spec.path_id, spec.path_label))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))

    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_all_tariff_path_lineplots(specs: List[LinePlotSpec], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for spec in specs:
        stem = spec.path_label.replace(" ", "_").replace("|", "").replace(">", "")
        plot_single_tariff_path_lineplot(
            spec,
            out_png=os.path.join(output_dir, f"{spec.path_id}_{stem}.png"),
            out_pdf=os.path.join(output_dir, f"{spec.path_id}_{stem}.pdf"),
        )


def plot_tariff_path_contact_sheet(specs: List[LinePlotSpec], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not specs:
        return

    global_ymax = max(max(max(s.meanA), max(s.meanB), max(s.meanC)) for s in specs) * 1.10
    fig, axes = plt.subplots(4, 4, figsize=(17.6, 12.0), constrained_layout=True)
    axes = axes.flatten()

    for idx, (ax, spec) in enumerate(zip(axes, specs)):
        _apply_axes(ax)
        H = len(spec.meanA)
        x = np.arange(1, H + 1)
        _add_phase_shading(ax, spec, global_ymax)
        ax.plot(x, spec.meanA, linewidth=1.35, color=PALETTE["A"], zorder=3)
        ax.plot(x, spec.meanB, linewidth=1.35, color=PALETTE["B"], zorder=3)
        ax.plot(x, spec.meanC, linewidth=1.35, color=PALETTE["C"], zorder=3)
        ax.set_title(_format_path_title(spec.path_id, spec.path_label), fontsize=9.2, pad=4)
        ax.set_xlim(1, H)
        ax.set_ylim(0.0, global_ymax)
        ax.set_xticks([1, 10, H])
        if idx % 4 != 0:
            ax.set_yticklabels([])
        if idx < 12:
            ax.set_xticklabels([])

    for ax in axes[len(specs):]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], linewidth=1.8, marker="o", markersize=4.5, color=PALETTE["A"], label="Strategy A"),
        plt.Line2D([], [], linewidth=1.8, marker="s", markersize=4.5, color=PALETTE["B"], label="Strategy B"),
        plt.Line2D([], [], linewidth=1.8, marker="^", markersize=4.5, color=PALETTE["C"], label="Strategy C"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.supylabel("Mean period cost", x=0.006, fontsize=10)
    fig.supxlabel("Time period", y=0.01, fontsize=10)

    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


# ---------------------------------------------------------------------
# Experiment 2 summary figures
# ---------------------------------------------------------------------
def _extract_xy(points: List[RampSummaryPoint]):
    return np.array([p.ramp_kappa for p in points], dtype=float)


def _age_labels(points: List[RampSummaryPoint]) -> list[str]:
    return [f"κ={p.ramp_kappa:.2f}\nτ={p.ramp_full_age}q" for p in points]


def plot_ramp_phase_length(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    xs = _extract_xy(points)
    phase2 = np.array([p.phase2_length for p in points], dtype=float)
    ramp_age = np.array([p.ramp_full_age for p in points], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, phase2, marker="o", linewidth=1.9, color=PALETTE["A"], label="Detected Phase II length")
    ax.plot(xs, ramp_age, marker="s", linewidth=1.6, linestyle="--", color=PALETTE["B"], label="Theoretical full-ramp age")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Quarters")
    ax.set_title("Ramp-up ability and detected Phase II duration")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_ramp_total_cost(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    xs = _extract_xy(points)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, [p.mean_total_cost_A for p in points], marker="o", linewidth=1.9, color=PALETTE["A"], label="Strategy A")
    ax.plot(xs, [p.mean_total_cost_B for p in points], marker="s", linewidth=1.9, color=PALETTE["B"], label="Strategy B")
    ax.plot(xs, [p.mean_total_cost_C for p in points], marker="^", linewidth=1.9, color=PALETTE["C"], label="Strategy C")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Mean total cost across 16 paths")
    ax.set_title("Total-cost sensitivity to candidate ramp-up capability")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_ramp_phase2_cost(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    xs = _extract_xy(points)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, [p.phase2_mean_cost_A for p in points], marker="o", linewidth=1.9, color=PALETTE["A"], label="Strategy A")
    ax.plot(xs, [p.phase2_mean_cost_B for p in points], marker="s", linewidth=1.9, color=PALETTE["B"], label="Strategy B")
    ax.plot(xs, [p.phase2_mean_cost_C for p in points], marker="^", linewidth=1.9, color=PALETTE["C"], label="Strategy C")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Mean Phase II cost across 16 paths")
    ax.set_title("Phase II exposure under ramp-up sensitivity")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_ramp_cost_gaps(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    xs = _extract_xy(points)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    _apply_axes(ax)
    ax.axhline(0.0, color="#8793A0", linewidth=0.9, linestyle="--")
    ax.plot(xs, [p.mean_gap_B_minus_A for p in points], marker="o", linewidth=1.8, color="#3B82A0", label="B − A")
    ax.plot(xs, [p.mean_gap_C_minus_A for p in points], marker="^", linewidth=1.8, color="#A35D3B", label="C − A")
    ax.plot(xs, [p.mean_gap_C_minus_B for p in points], marker="s", linewidth=1.6, linestyle=":", color=PALETTE["gap"], label="C − B")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Mean cost gap across 16 paths")
    ax.set_title("Strategic value of reconfiguration under ramp-up sensitivity")
    ax.legend(frameon=False, ncol=3)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_ramp_win_share(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    xs = _extract_xy(points)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, [p.win_share_A for p in points], marker="o", linewidth=1.8, color=PALETTE["A"], label="Strategy A")
    ax.plot(xs, [p.win_share_B for p in points], marker="s", linewidth=1.8, color=PALETTE["B"], label="Strategy B")
    ax.plot(xs, [p.win_share_C for p in points], marker="^", linewidth=1.8, color=PALETTE["C"], label="Strategy C")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Share of 16 tariff paths won")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Strategy dominance across tariff paths")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


# ---------------------------------------------------------------------
# Experiment 2 path-level figures
# ---------------------------------------------------------------------
def _kappa_positions(points: List[PathRampPoint]) -> tuple[np.ndarray, list[str]]:
    xs = np.arange(len(points), dtype=float)
    labels = [f"{p.ramp_kappa:.2f}" for p in points]
    return xs, labels


def plot_path_total_cost_vs_kappa(points: List[PathRampPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    xs, labels = _kappa_positions(points)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    _apply_axes(ax)
    yA = np.array([p.mean_cost_A for p in points], dtype=float)
    yB = np.array([p.mean_cost_B for p in points], dtype=float)
    yC = np.array([p.mean_cost_C for p in points], dtype=float)
    sA = np.array([p.std_cost_A for p in points], dtype=float)
    sB = np.array([p.std_cost_B for p in points], dtype=float)
    sC = np.array([p.std_cost_C for p in points], dtype=float)
    _plot_strategy_series(ax, xs, yA, sA, "A", "Strategy A")
    _plot_strategy_series(ax, xs, yB, sB, "B", "Strategy B")
    _plot_strategy_series(ax, xs, yC, sC, "C", "Strategy C")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Mean total cost")
    ax.set_title(f"{points[0].path_id}   {points[0].path_label.replace('->', '→')}\nTotal cost by ramp-up parameter")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_path_phase2_cost_vs_kappa(points: List[PathRampPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    xs, labels = _kappa_positions(points)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, [p.phase2_mean_cost_A for p in points], marker="o", linewidth=1.8, color=PALETTE["A"], label="Strategy A")
    ax.plot(xs, [p.phase2_mean_cost_B for p in points], marker="s", linewidth=1.8, color=PALETTE["B"], label="Strategy B")
    ax.plot(xs, [p.phase2_mean_cost_C for p in points], marker="^", linewidth=1.8, color=PALETTE["C"], label="Strategy C")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Mean Phase II cost")
    ax.set_title(f"{points[0].path_id}   {points[0].path_label.replace('->', '→')}\nPhase II cost by ramp-up parameter")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_path_phase2_length_vs_kappa(points: List[PathRampPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    xs, labels = _kappa_positions(points)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, [p.phase2_length for p in points], marker="o", linewidth=1.8, color="#5E81AC")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Detected Phase II length")
    ax.set_title(f"{points[0].path_id}   {points[0].path_label.replace('->', '→')}\nPhase II duration by ramp-up parameter")
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


# ---------------------------------------------------------------------
# Experiment 2 contact sheet changed to grouped bars
# ---------------------------------------------------------------------
def _draw_grouped_bar_panel(ax, spec: PathContactSheetSpec, y_low: float, y_high: float) -> None:
    pts = list(spec.points)
    xs = np.arange(len(pts), dtype=float)
    width = 0.22

    yA = np.array([p.mean_cost_A for p in pts], dtype=float)
    yB = np.array([p.mean_cost_B for p in pts], dtype=float)
    yC = np.array([p.mean_cost_C for p in pts], dtype=float)

    ax.bar(xs - width, yA, width=width, color=PALETTE["A"], alpha=0.88, edgecolor="white", linewidth=0.35)
    ax.bar(xs,         yB, width=width, color=PALETTE["B"], alpha=0.88, edgecolor="white", linewidth=0.35)
    ax.bar(xs + width, yC, width=width, color=PALETTE["C"], alpha=0.88, edgecolor="white", linewidth=0.35)

    winners = [p.best_strategy for p in pts]
    winner_to_x = {"A": xs - width, "B": xs, "C": xs + width}
    winner_to_color = {"A": PALETTE["A"], "B": PALETTE["B"], "C": PALETTE["C"]}
    marker_y = y_high - 0.05 * (y_high - y_low)
    for i, w in enumerate(winners):
        ax.scatter(float(winner_to_x[w][i]), marker_y, s=16, color=winner_to_color[w], zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p.ramp_kappa:.2f}" for p in pts], rotation=0)
    ax.set_ylim(y_low, y_high)
    ax.set_title(_format_path_title(spec.path_id, spec.path_label), fontsize=9.3, pad=4)


def plot_path_total_cost_contact_sheet(specs: List[PathContactSheetSpec], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not specs:
        return

    all_vals: List[float] = []
    for spec in specs:
        for p in spec.points:
            all_vals.extend([p.mean_cost_A, p.mean_cost_B, p.mean_cost_C])
    y_min = min(all_vals)
    y_max = max(all_vals)
    pad = max(1.0, 0.08 * (y_max - y_min))
    y_low = max(0.0, y_min - pad)
    y_high = y_max + pad

    fig, axes = plt.subplots(4, 4, figsize=(18.2, 12.2), constrained_layout=True)
    axes = axes.flatten()

    for idx, (ax, spec) in enumerate(zip(axes, sorted(specs, key=lambda s: s.path_id))):
        _apply_axes(ax)
        _draw_grouped_bar_panel(ax, spec, y_low, y_high)
        if idx % 4 != 0:
            ax.set_yticklabels([])
        if idx < 12:
            ax.set_xticklabels([])

    for ax in axes[len(specs):]:
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["A"], label="Strategy A"),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["B"], label="Strategy B"),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["C"], label="Strategy C"),
        plt.Line2D([], [], marker="o", linestyle="", color="#4A5563", label="dot = lowest-cost strategy", markersize=5),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.supylabel("Mean total cost", x=0.006, fontsize=10)
    fig.supxlabel("Ramp-up parameter κ", y=0.01, fontsize=10)

    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)