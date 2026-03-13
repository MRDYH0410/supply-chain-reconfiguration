from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


PALETTE = {
    "A": "#183A5B",
    "B": "#1E8A7A",
    "C": "#B65E2A",
}
GRID_COLOR = "#D4DCE3"
TEXT_DARK = "#24323F"
TEXT_MUTED = "#5E6B77"
HEATMAP_CMAP = "Blues_r"


@dataclass
class DurationSummaryPoint:
    profile_id: str
    profile_name: str
    phase2_high_scale: float
    phase2_low_scale: float
    phase3_high_scale: float
    phase3_low_scale: float
    mean_phase2_length: float
    mean_phase3_length: float
    mean_total_cost_A: float
    mean_total_cost_B: float
    mean_total_cost_C: float
    mean_gap_B_minus_A: float
    mean_gap_C_minus_A: float
    mean_gap_C_minus_B: float
    win_share_A: float
    win_share_B: float
    win_share_C: float


@dataclass
class PathDurationPoint:
    path_id: str
    path_label: str
    profile_id: str
    profile_name: str
    phase2_high_scale: float
    phase2_low_scale: float
    phase3_high_scale: float
    phase3_low_scale: float
    phase2_length: int
    phase3_length: int
    mean_cost_A: float
    mean_cost_B: float
    mean_cost_C: float
    std_cost_A: float
    std_cost_B: float
    std_cost_C: float
    gap_B_minus_A: float
    gap_C_minus_A: float
    best_strategy: str


@dataclass
class PathContactSheetSpec:
    path_id: str
    path_label: str
    points: List[PathDurationPoint]


def _set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def _apply_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3, width=0.7, colors=TEXT_DARK)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.6)


def _ensure_dir_pair(out_png: str, out_pdf: str) -> None:
    d1 = os.path.dirname(out_png)
    d2 = os.path.dirname(out_pdf)
    if d1:
        os.makedirs(d1, exist_ok=True)
    if d2 and d2 != d1:
        os.makedirs(d2, exist_ok=True)


def _profile_positions_labels(points) -> tuple[np.ndarray, list[str]]:
    xs = np.arange(len(points), dtype=float)
    labels = [p.profile_id for p in points]
    return xs, labels


def _plot_strategy_lines(ax, xs, yA, yB, yC) -> None:
    ax.plot(xs, yA, marker="o", linewidth=1.85, markersize=4.7, markeredgewidth=0.5, color=PALETTE["A"], label="Strategy A")
    ax.plot(xs, yB, marker="s", linewidth=1.85, markersize=4.7, markeredgewidth=0.5, color=PALETTE["B"], label="Strategy B")
    ax.plot(xs, yC, marker="^", linewidth=1.85, markersize=4.7, markeredgewidth=0.5, color=PALETTE["C"], label="Strategy C")


def plot_duration_phase_lengths(points: List[DurationSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    phase2 = np.array([p.mean_phase2_length for p in points], dtype=float)
    phase3 = np.array([p.mean_phase3_length for p in points], dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, phase2, marker="o", linewidth=1.9, color="#5E81AC", label="Mean Phase II length")
    ax.plot(xs, phase3, marker="s", linewidth=1.9, color="#A3BE8C", label="Mean Phase III length")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Duration profile")
    ax.set_ylabel("Mean length across 16 paths")
    ax.set_title("How state persistence reshapes Phase II and Phase III")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_duration_total_cost(points: List[DurationSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    _apply_axes(ax)
    _plot_strategy_lines(
        ax, xs,
        np.array([p.mean_total_cost_A for p in points], dtype=float),
        np.array([p.mean_total_cost_B for p in points], dtype=float),
        np.array([p.mean_total_cost_C for p in points], dtype=float),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Duration profile")
    ax.set_ylabel("Mean total cost across 16 paths")
    ax.set_title("Total-cost response to tariff-state persistence")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_duration_cost_gaps(points: List[DurationSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _apply_axes(ax)
    ax.axhline(0.0, linewidth=0.9, linestyle="--", color="#8793A0")
    ax.plot(xs, [p.mean_gap_B_minus_A for p in points], marker="o", linewidth=1.8, color="#3B82A0", label="B − A")
    ax.plot(xs, [p.mean_gap_C_minus_A for p in points], marker="^", linewidth=1.8, color="#A35D3B", label="C − A")
    ax.plot(xs, [p.mean_gap_C_minus_B for p in points], marker="s", linewidth=1.6, linestyle=":", color="#7F5A9A", label="C − B")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Duration profile")
    ax.set_ylabel("Mean cost gap across 16 paths")
    ax.set_title("Strategic value of reconfiguration under persistence sensitivity")
    ax.legend(frameon=False, ncol=3)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_duration_win_share(points: List[DurationSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    _apply_axes(ax)
    _plot_strategy_lines(
        ax, xs,
        np.array([p.win_share_A for p in points], dtype=float),
        np.array([p.win_share_B for p in points], dtype=float),
        np.array([p.win_share_C for p in points], dtype=float),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Duration profile")
    ax.set_ylabel("Share of 16 tariff paths won")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Strategy dominance across persistence profiles")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_duration_profile_design(points: List[DurationSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, [p.phase2_high_scale for p in points], marker="o", linewidth=1.8, color="#5E81AC", label="Phase II high")
    ax.plot(xs, [p.phase2_low_scale for p in points], marker="s", linewidth=1.8, color="#88C0D0", label="Phase II low")
    ax.plot(xs, [p.phase3_high_scale for p in points], marker="^", linewidth=1.8, color="#A35D3B", label="Phase III high")
    ax.plot(xs, [p.phase3_low_scale for p in points], marker="D", linewidth=1.8, color="#D08770", label="Phase III low")
    ax.axhline(1.0, linewidth=0.9, linestyle="--", color=TEXT_MUTED, alpha=0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Duration profile")
    ax.set_ylabel("Persistence multiplier")
    ax.set_title("Experiment 4 duration-profile design")
    ax.legend(frameon=False, ncol=2)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_path_total_cost_vs_profile(points: List[PathDurationPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    yA = np.array([p.mean_cost_A for p in points], dtype=float)
    yB = np.array([p.mean_cost_B for p in points], dtype=float)
    yC = np.array([p.mean_cost_C for p in points], dtype=float)
    sA = np.array([p.std_cost_A for p in points], dtype=float)
    sB = np.array([p.std_cost_B for p in points], dtype=float)
    sC = np.array([p.std_cost_C for p in points], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, yA, marker="o", linewidth=1.8, color=PALETTE["A"], label="Strategy A")
    ax.plot(xs, yB, marker="s", linewidth=1.8, color=PALETTE["B"], label="Strategy B")
    ax.plot(xs, yC, marker="^", linewidth=1.8, color=PALETTE["C"], label="Strategy C")
    ax.fill_between(xs, yA - sA, yA + sA, color=PALETTE["A"], alpha=0.10)
    ax.fill_between(xs, yB - sB, yB + sB, color=PALETTE["B"], alpha=0.10)
    ax.fill_between(xs, yC - sC, yC + sC, color=PALETTE["C"], alpha=0.10)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Duration profile")
    ax.set_ylabel("Mean total cost")
    ax.set_title(f"{points[0].path_id}   {points[0].path_label.replace('->', '→')}\nTotal cost by persistence profile")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_path_phase_lengths_vs_profile(points: List[PathDurationPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, [p.phase2_length for p in points], marker="o", linewidth=1.8, color="#5E81AC", label="Phase II length")
    ax.plot(xs, [p.phase3_length for p in points], marker="s", linewidth=1.8, color="#A3BE8C", label="Phase III length")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Duration profile")
    ax.set_ylabel("Periods")
    ax.set_title(f"{points[0].path_id}   {points[0].path_label.replace('->', '→')}\nPhase lengths by persistence profile")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


# ---------------------------------------------------------------------
# Experiment 4 contact sheet changed to heatmap panels
# ---------------------------------------------------------------------
def _heatmap_matrix(spec: PathContactSheetSpec) -> np.ndarray:
    pts = list(spec.points)
    return np.array([
        [p.mean_cost_A for p in pts],
        [p.mean_cost_B for p in pts],
        [p.mean_cost_C for p in pts],
    ], dtype=float)


def plot_path_total_cost_contact_sheet(specs: List[PathContactSheetSpec], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not specs:
        return
    specs = sorted(specs, key=lambda s: s.path_id)

    mats = [_heatmap_matrix(spec) for spec in specs]
    vmin = min(float(mat.min()) for mat in mats)
    vmax = max(float(mat.max()) for mat in mats)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(4, 4, figsize=(18.4, 12.2), constrained_layout=True)
    axes = axes.flatten()

    im = None
    for idx, (ax, spec, mat) in enumerate(zip(axes, specs, mats)):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        im = ax.imshow(mat, cmap=HEATMAP_CMAP, aspect="auto", norm=norm)
        ax.set_title(f"{spec.path_id}   {spec.path_label.replace('->', '→')}", fontsize=9.2, pad=4)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels([p.profile_id for p in spec.points], fontsize=8.1)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["A", "B", "C"], fontsize=8.1)
        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0, colors=TEXT_DARK)
        best_rows = np.argmin(mat, axis=0)
        for col, row in enumerate(best_rows):
            ax.scatter(col, row, s=14, color="#111827", zorder=4)
        if idx % 4 != 0:
            ax.set_yticklabels([])
        if idx < 12:
            ax.set_xticklabels([])

    for ax in axes[len(specs):]:
        ax.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.78, pad=0.01)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Mean total cost", fontsize=9)

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#111827", markersize=4.5, label="dot = lowest-cost strategy in each profile"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=1, frameon=False)
    fig.supylabel("Strategy", x=0.006, fontsize=10)
    fig.supxlabel("Persistence profile", y=0.01, fontsize=10)

    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)