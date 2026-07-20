from __future__ import annotations

from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from experiments_2.sensitivity_runner import Exp2ScenarioResult


PALETTE = {
    "A": "#16324F",
    "B": "#2F6C7A",
    "C": "#B06C49",
    "GRID": "#D7DEE7",
    "TEXT": "#243447",
    "EDGE": "#BFC9D4",
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
            "axes.titlesize": 14.5,
            "axes.labelsize": 12.0,
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
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PALETTE["EDGE"])
    ax.spines["bottom"].set_color(PALETTE["EDGE"])


def _sorted_unique(vals: Iterable[float]) -> List[float]:
    return sorted({float(v) for v in vals})


def _matrix(results: List[Exp2ScenarioResult], attr: str, spans: List[float], shifts: List[float]) -> np.ndarray:
    mat = np.zeros((len(spans), len(shifts)), dtype=float)
    lookup: Dict[tuple[float, float], Exp2ScenarioResult] = {
        (float(r.span), float(r.level_shift)): r for r in results
    }
    for i, span in enumerate(spans):
        for j, shift in enumerate(shifts):
            r = lookup[(float(span), float(shift))]
            mat[i, j] = float(getattr(r, attr))
    return mat


# def _annotate(ax, data: np.ndarray, fmt: str = ".0f") -> None:
#     nrows, ncols = data.shape
#     normed = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-12)
#     for i in range(nrows):
#         for j in range(ncols):
#             txt_color = "white" if normed[i, j] > 0.62 else PALETTE["TEXT"]
#             ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8.2, color=txt_color)


def _heatmap(
    ax,
    data: np.ndarray,
    spans: List[float],
    shifts: List[float],
    title: str,
    cmap: str = "viridis",
    fmt: str = ".0f",
    norm=None,
):
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels([f"{int(round(s * 100))}%" for s in shifts], rotation=40, ha="right")
    ax.set_yticks(range(len(spans)))
    ax.set_yticklabels([f"±{int(round(v * 100))}%" for v in spans])
    ax.set_xlabel("Tail baseline shift for M1→D1 (periods 16–80)")
    ax.set_ylabel("Tail span around shifted baseline")
    _style_axes(ax)
    # _annotate(ax, data, fmt=fmt)
    return im


def _heatmap_colorbar(fig, im, ax, label: str) -> None:
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.04)
    cbar.outline.set_edgecolor("#C7D0DA")
    cbar.ax.tick_params(labelsize=9.2, colors=PALETTE["TEXT"])
    cbar.ax.set_ylabel(label, rotation=270, labelpad=15)


def plot_total_cost_heatmaps(results: List[Exp2ScenarioResult], out_png: str, out_pdf: str) -> None:
    _apply_theme()
    spans = _sorted_unique(r.span for r in results)
    shifts = _sorted_unique(r.level_shift for r in results)
    matA = _matrix(results, "total_A", spans, shifts)
    matB = _matrix(results, "total_B", spans, shifts)
    matC = _matrix(results, "total_C", spans, shifts)

    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.7))
    im0 = _heatmap(axes[0], matA, spans, shifts, "Strategy A total discounted cost", cmap="cividis")
    im1 = _heatmap(axes[1], matB, spans, shifts, "Strategy B total discounted cost", cmap="cividis")
    im2 = _heatmap(axes[2], matC, spans, shifts, "Strategy C total discounted cost", cmap="cividis")
    _heatmap_colorbar(fig, im0, axes[0], "Cost")
    _heatmap_colorbar(fig, im1, axes[1], "Cost")
    _heatmap_colorbar(fig, im2, axes[2], "Cost")
    fig.suptitle("Experiment 2  Sensitivity to downside shifts in the last-65-period M1→D1 tariff", y=1.02, fontsize=16)
    _save(fig, out_png, out_pdf)


def plot_gap_heatmaps(results: List[Exp2ScenarioResult], out_png: str, out_pdf: str) -> None:
    _apply_theme()
    spans = _sorted_unique(r.span for r in results)
    shifts = _sorted_unique(r.level_shift for r in results)
    matAB = _matrix(results, "gap_A_minus_B", spans, shifts)
    matAC = _matrix(results, "gap_A_minus_C", spans, shifts)
    matBC = _matrix(results, "gap_B_minus_C", spans, shifts)

    gap_max = float(max(np.abs(matAB).max(), np.abs(matAC).max(), np.abs(matBC).max()))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-gap_max, vmax=gap_max)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.7))
    im0 = _heatmap(axes[0], matAB, spans, shifts, "A − B cost gap (positive means B cheaper)", cmap="RdBu_r", norm=norm)
    im1 = _heatmap(axes[1], matAC, spans, shifts, "A − C cost gap (positive means C cheaper)", cmap="RdBu_r", norm=norm)
    im2 = _heatmap(axes[2], matBC, spans, shifts, "B − C cost gap (positive means C cheaper)", cmap="RdBu_r", norm=norm)
    _heatmap_colorbar(fig, im0, axes[0], "Gap")
    _heatmap_colorbar(fig, im1, axes[1], "Gap")
    _heatmap_colorbar(fig, im2, axes[2], "Gap")
    fig.suptitle("Experiment 2  Strategy-gap sensitivity under downside last-65-period tail-tariff perturbations", y=1.02, fontsize=16)
    _save(fig, out_png, out_pdf)


def plot_activation_heatmaps(results: List[Exp2ScenarioResult], out_png: str, out_pdf: str) -> None:
    _apply_theme()
    spans = _sorted_unique(r.span for r in results)
    shifts = _sorted_unique(r.level_shift for r in results)
    matB = _matrix(results, "activation_B", spans, shifts)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.8))
    im0 = _heatmap(ax, matB, spans, shifts, "Strategy B activation period", cmap="magma", fmt=".0f")
    _heatmap_colorbar(fig, im0, ax, "Period")
    fig.suptitle("Experiment 2  Strategy-B action timing under downside last-65-period tail-tariff perturbations", y=1.02, fontsize=16)
    _save(fig, out_png, out_pdf)


def _profile_line(ax, x, y, color: str, label: str, marker: str) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.25,
        marker=marker,
        markersize=6.4,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.45,
        label=label,
    )


def plot_profile_lines(results: List[Exp2ScenarioResult], out_png: str, out_pdf: str) -> None:
    _apply_theme()
    spans = _sorted_unique(r.span for r in results)
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.0))
    axes = axes.ravel()

    for ax, span in zip(axes, spans):
        sub = [r for r in results if abs(r.span - span) < 1e-12]
        sub = sorted(sub, key=lambda x: x.level_shift)
        x = [100.0 * r.level_shift for r in sub]
        _profile_line(ax, x, [r.total_A for r in sub], PALETTE["A"], "A", "o")
        _profile_line(ax, x, [r.total_B for r in sub], PALETTE["B"], "B", "s")
        _profile_line(ax, x, [r.total_C for r in sub], PALETTE["C"], "C", "D")
        ax.set_title(f"Tail span = ±{int(round(span * 100))}%")
        ax.set_xlabel("Tail baseline shift (%)")
        ax.set_ylabel("Total discounted cost")
        ax.grid(True, color=PALETTE["GRID"], linewidth=0.8, alpha=0.75)
        _style_axes(ax)
        ax.legend(loc="upper left")

    fig.suptitle("Experiment 2  Cost profiles across downside last-65-period tail baseline shifts", y=1.01, fontsize=16)
    _save(fig, out_png, out_pdf)
