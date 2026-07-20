from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from experiments_3.sensitivity_runner import Exp3HorizonResult


PALETTE = {
    "A": "#16324F",
    "B": "#2F6C7A",
    "C": "#B06C49",
    "GRID": "#D7DEE7",
    "TEXT": "#243447",
    "EDGE": "#BFC9D4",
    "RESET": "#6C7A89",
}


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": PALETTE["EDGE"],
            "axes.labelcolor": PALETTE["TEXT"],
            "xtick.color": PALETTE["TEXT"],
            "ytick.color": PALETTE["TEXT"],
            "text.color": PALETTE["TEXT"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 15,
            "axes.labelsize": 12.2,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#DCE3EA",
            "legend.facecolor": "white",
        }
    )


def _save(fig, out_png: str, out_pdf: str) -> None:
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax) -> None:
    ax.grid(True, color=PALETTE["GRID"], linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PALETTE["EDGE"])
    ax.spines["bottom"].set_color(PALETTE["EDGE"])


def _sorted(results: List[Exp3HorizonResult]) -> List[Exp3HorizonResult]:
    return sorted(results, key=lambda r: int(r.horizon))


def _plot_series(ax, x, y, color: str, label: str, marker: str) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.35,
        marker=marker,
        markersize=6.8,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.55,
        label=label,
    )


def plot_total_cost_vs_horizon(results: List[Exp3HorizonResult], out_png: str, out_pdf: str) -> None:
    _apply_theme()
    rows = _sorted(results)
    x = [r.horizon for r in rows]

    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    _plot_series(ax, x, [r.total_A for r in rows], PALETTE["A"], "Strategy A", "o")
    _plot_series(ax, x, [r.total_B for r in rows], PALETTE["B"], "Strategy B", "s")
    _plot_series(ax, x, [r.total_C for r in rows], PALETTE["C"], "Strategy C", "D")
    ax.set_title("Experiment 3  Total discounted cost under different planning horizons")
    ax.set_xlabel("Planning horizon H")
    ax.set_ylabel("Total discounted cost")
    ax.set_xticks(x)
    _style_axes(ax)
    ax.legend(loc="upper left", ncol=3)
    _save(fig, out_png, out_pdf)


def plot_cost_gap_vs_horizon(results: List[Exp3HorizonResult], out_png: str, out_pdf: str) -> None:
    _apply_theme()
    rows = _sorted(results)
    x = [r.horizon for r in rows]

    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    _plot_series(ax, x, [r.gap_A_minus_B for r in rows], PALETTE["A"], "A − B", "o")
    _plot_series(ax, x, [r.gap_A_minus_C for r in rows], PALETTE["C"], "A − C", "D")
    _plot_series(ax, x, [r.gap_B_minus_C for r in rows], PALETTE["B"], "B − C", "s")
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color=PALETTE["RESET"], alpha=0.9)
    ax.set_title("Experiment 3  Strategy cost gaps across planning horizons")
    ax.set_xlabel("Planning horizon H")
    ax.set_ylabel("Discounted cost gap")
    ax.set_xticks(x)
    _style_axes(ax)
    ax.legend(loc="upper left", ncol=3)
    _save(fig, out_png, out_pdf)


def plot_total_demand_vs_horizon(results: List[Exp3HorizonResult], out_png: str, out_pdf: str) -> None:
    _apply_theme()
    rows = _sorted(results)
    x = [r.horizon for r in rows]

    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    _plot_series(ax, x, [r.demand_A for r in rows], PALETTE["A"], "Strategy A", "o")
    _plot_series(ax, x, [r.demand_B for r in rows], PALETTE["B"], "Strategy B", "s")
    _plot_series(ax, x, [r.demand_C for r in rows], PALETTE["C"], "Strategy C", "D")
    ax.set_title("Experiment 3  Total realised demand under different planning horizons")
    ax.set_xlabel("Planning horizon H")
    ax.set_ylabel("Total realised demand")
    ax.set_xticks(x)
    _style_axes(ax)
    ax.legend(loc="upper left", ncol=3)
    _save(fig, out_png, out_pdf)


def plot_action_timing_vs_horizon(results: List[Exp3HorizonResult], out_png: str, out_pdf: str) -> None:
    _apply_theme()
    rows = _sorted(results)
    x = [r.horizon for r in rows]

    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    _plot_series(ax, x, [r.activation_B for r in rows], PALETTE["B"], "B activation", "s")
    _plot_series(ax, x, [r.activation_C for r in rows], PALETTE["C"], "C activation", "D")
    _plot_series(ax, x, [r.withdrawal_C for r in rows], PALETTE["A"], "C withdrawal", "o")
    ax.set_title("Experiment 3  Action timing across planning horizons")
    ax.set_xlabel("Planning horizon H")
    ax.set_ylabel("Time period")
    ax.set_xticks(x)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(ax)
    ax.legend(loc="upper left", ncol=3)
    _save(fig, out_png, out_pdf)
