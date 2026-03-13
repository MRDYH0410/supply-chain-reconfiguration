from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "A": "#183A5B",
    "B": "#1E8A7A",
    "C": "#B65E2A",
    "gap": "#7F5A9A",
}
GRID_COLOR = "#D4DCE3"
TEXT_DARK = "#24323F"
TEXT_MUTED = "#5E6B77"


@dataclass
class ProfileSummaryPoint:
    profile_id: str
    profile_name: str
    low_scale: float
    high_scale: float
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
    phase2_gap_C_minus_B: float
    win_share_A: float
    win_share_B: float
    win_share_C: float


@dataclass
class PathTariffPoint:
    path_id: str
    path_label: str
    profile_id: str
    profile_name: str
    low_scale: float
    high_scale: float
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
    points: List[PathTariffPoint]


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
    labels = [f"{p.profile_id}\nL×{p.low_scale:.2f} H×{p.high_scale:.2f}" for p in points]
    return xs, labels


def _profile_short_labels(points) -> tuple[np.ndarray, list[str]]:
    xs = np.arange(len(points), dtype=float)
    labels = [f"{p.profile_id}" for p in points]
    return xs, labels


def _plot_strategy_lines(ax, xs, yA, yB, yC, sA=None, sB=None, sC=None) -> None:
    styles = {
        "A": dict(marker="o", color=PALETTE["A"], label="Strategy A"),
        "B": dict(marker="s", color=PALETTE["B"], label="Strategy B"),
        "C": dict(marker="^", color=PALETTE["C"], label="Strategy C"),
    }
    for key, y, s in [("A", yA, sA), ("B", yB, sB), ("C", yC, sC)]:
        st = styles[key]
        ax.plot(xs, y, linewidth=1.85, markersize=4.8, markeredgewidth=0.5, zorder=3, **st)
        if s is not None:
            ax.fill_between(xs, y - s, y + s, color=st["color"], alpha=0.10, linewidth=0.0, zorder=2)


def plot_profile_phase_length(points: List[ProfileSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    ys = np.array([p.phase2_length for p in points], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, ys, marker="o", linewidth=1.9, color="#5E81AC")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tariff-level profile")
    ax.set_ylabel("Phase II length")
    ax.set_title("Phase II duration under alternative tariff-level definitions")
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_profile_total_cost(points: List[ProfileSummaryPoint], out_png: str, out_pdf: str) -> None:
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
    ax.set_xlabel("Tariff-level profile")
    ax.set_ylabel("Mean total cost across 16 tariff paths")
    ax.set_title("Total-cost response to tariff-level uncertainty")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_profile_phase2_cost(points: List[ProfileSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    _apply_axes(ax)
    _plot_strategy_lines(
        ax, xs,
        np.array([p.phase2_mean_cost_A for p in points], dtype=float),
        np.array([p.phase2_mean_cost_B for p in points], dtype=float),
        np.array([p.phase2_mean_cost_C for p in points], dtype=float),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tariff-level profile")
    ax.set_ylabel("Mean Phase II period cost across 16 tariff paths")
    ax.set_title("Phase II cost exposure under alternative tariff-level profiles")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_profile_cost_gaps(points: List[ProfileSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_positions_labels(points)
    gBA = np.array([p.mean_gap_B_minus_A for p in points], dtype=float)
    gCA = np.array([p.mean_gap_C_minus_A for p in points], dtype=float)
    g2BA = np.array([p.phase2_gap_B_minus_A for p in points], dtype=float)
    g2CA = np.array([p.phase2_gap_C_minus_A for p in points], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    _apply_axes(ax)
    ax.axhline(0.0, linewidth=0.9, linestyle="--", color="#8793A0")
    ax.plot(xs, gBA, marker="o", linewidth=1.8, color="#3B82A0", label="B − A overall")
    ax.plot(xs, gCA, marker="^", linewidth=1.8, color="#A35D3B", label="C − A overall")
    ax.plot(xs, g2BA, marker="s", linewidth=1.45, linestyle=":", color="#5E81AC", label="B − A in Phase II")
    ax.plot(xs, g2CA, marker="D", linewidth=1.45, linestyle=":", color="#D08770", label="C − A in Phase II")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tariff-level profile")
    ax.set_ylabel("Cost gap")
    ax.set_title("Strategic value of reconfiguration under tariff-level sensitivity")
    ax.legend(frameon=False, ncol=2)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_profile_win_share(points: List[ProfileSummaryPoint], out_png: str, out_pdf: str) -> None:
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
    ax.set_xlabel("Tariff-level profile")
    ax.set_ylabel("Share of 16 tariff paths won")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Strategy dominance across tariff paths")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_path_total_cost_vs_profile(points: List[PathTariffPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_short_labels(points)
    yA = np.array([p.mean_cost_A for p in points], dtype=float)
    yB = np.array([p.mean_cost_B for p in points], dtype=float)
    yC = np.array([p.mean_cost_C for p in points], dtype=float)
    sA = np.array([p.std_cost_A for p in points], dtype=float)
    sB = np.array([p.std_cost_B for p in points], dtype=float)
    sC = np.array([p.std_cost_C for p in points], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    _apply_axes(ax)
    _plot_strategy_lines(ax, xs, yA, yB, yC, sA, sB, sC)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tariff-level profile")
    ax.set_ylabel("Mean total cost")
    ax.set_title(f"{points[0].path_id}   {points[0].path_label.replace('->', '→')}\nTotal cost by tariff-level profile")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_path_phase2_cost_vs_profile(points: List[PathTariffPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_short_labels(points)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    _apply_axes(ax)
    _plot_strategy_lines(
        ax, xs,
        np.array([p.phase2_mean_cost_A for p in points], dtype=float),
        np.array([p.phase2_mean_cost_B for p in points], dtype=float),
        np.array([p.phase2_mean_cost_C for p in points], dtype=float),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tariff-level profile")
    ax.set_ylabel("Mean Phase II cost")
    ax.set_title(f"{points[0].path_id}   {points[0].path_label.replace('->', '→')}\nPhase II cost by tariff-level profile")
    ax.legend(frameon=False)
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_path_phase2_length_vs_profile(points: List[PathTariffPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs, labels = _profile_short_labels(points)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    _apply_axes(ax)
    ax.plot(xs, np.array([p.phase2_length for p in points], dtype=float), marker="o", linewidth=1.8, color="#5E81AC")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tariff-level profile")
    ax.set_ylabel("Detected Phase II length")
    ax.set_title(f"{points[0].path_id}   {points[0].path_label.replace('->', '→')}\nPhase II duration by tariff-level profile")
    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def plot_path_total_cost_contact_sheet(specs: List[PathContactSheetSpec], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not specs:
        return
    specs = sorted(specs, key=lambda s: s.path_id)

    y_values: List[float] = []
    for spec in specs:
        for p in spec.points:
            y_values.extend([
                p.mean_cost_A - p.std_cost_A,
                p.mean_cost_A + p.std_cost_A,
                p.mean_cost_B - p.std_cost_B,
                p.mean_cost_B + p.std_cost_B,
                p.mean_cost_C - p.std_cost_C,
                p.mean_cost_C + p.std_cost_C,
            ])
    y_min = min(y_values)
    y_max = max(y_values)
    pad = max(1.0, 0.06 * (y_max - y_min))
    y_low = max(0.0, y_min - pad)
    y_high = y_max + pad

    fig, axes = plt.subplots(4, 4, figsize=(17.8, 11.9), constrained_layout=True)
    axes = axes.flatten()

    for idx, (ax, spec) in enumerate(zip(axes, specs)):
        _apply_axes(ax)
        pts = spec.points
        xs = np.arange(len(pts), dtype=float)
        _plot_strategy_lines(
            ax, xs,
            np.array([p.mean_cost_A for p in pts], dtype=float),
            np.array([p.mean_cost_B for p in pts], dtype=float),
            np.array([p.mean_cost_C for p in pts], dtype=float),
            np.array([p.std_cost_A for p in pts], dtype=float),
            np.array([p.std_cost_B for p in pts], dtype=float),
            np.array([p.std_cost_C for p in pts], dtype=float),
        )
        ax.set_title(f"{spec.path_id}   {spec.path_label.replace('->', '→')}", fontsize=9.3, pad=4)
        ax.set_ylim(y_low, y_high)
        ax.set_xticks(xs)
        ax.set_xticklabels([p.profile_id for p in pts], rotation=0)
        if idx % 4 != 0:
            ax.set_yticklabels([])
        if idx < 12:
            ax.set_xticklabels([])

    for ax in axes[len(specs):]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], linewidth=1.8, marker="o", color=PALETTE["A"], label="Strategy A"),
        plt.Line2D([], [], linewidth=1.8, marker="s", color=PALETTE["B"], label="Strategy B"),
        plt.Line2D([], [], linewidth=1.8, marker="^", color=PALETTE["C"], label="Strategy C"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.supylabel("Mean total cost", x=0.006, fontsize=10)
    fig.supxlabel("Tariff-level profile", y=0.01, fontsize=10)

    _ensure_dir_pair(out_png, out_pdf)
    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)