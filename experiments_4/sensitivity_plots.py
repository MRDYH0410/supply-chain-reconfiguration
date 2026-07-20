from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from experiments_4.sensitivity_runner import Exp4ScenarioResult


PALETTE = {
    "A": "#16324F",  # deep navy
    "B": "#2F6C7A",  # muted teal
    "C": "#B06C49",  # warm bronze
    "PROFILE_GRAY": "#A9C7E8",
    "PROFILE_LOW": "#145DA0",   # low anchor: blue
    "PROFILE_MID": "#D97706",   # middle anchor: amber
    "PROFILE_HIGH": "#A61B29",  # high anchor: burgundy
    "GRID": "#D7DEE7",
    "TEXT": "#243447",
    "EDGE": "#E9EEF4",
}


def _pct(x: float) -> float:
    return 100.0 * float(x)


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#B8C2CC",
            "axes.labelcolor": PALETTE["TEXT"],
            "xtick.color": PALETTE["TEXT"],
            "ytick.color": PALETTE["TEXT"],
            "text.color": PALETTE["TEXT"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 15,
            "axes.labelsize": 12.5,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#DCE3EA",
            "legend.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig, png_path: str, pdf_path: str) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _style_2d_axes(ax) -> None:
    ax.grid(True, color=PALETTE["GRID"], linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BCC6D0")
    ax.spines["bottom"].set_color("#BCC6D0")


def _line(ax, xs, ys, color: str, label: str, marker: str = "o") -> None:
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2.35,
        marker=marker,
        markersize=6.8,
        markerfacecolor="white",
        markeredgewidth=1.55,
        markeredgecolor=color,
        solid_capstyle="round",
        label=label,
        zorder=3,
    )


def _format_3d_axes(ax) -> None:
    ax.view_init(elev=20, azim=-58)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((0.985, 0.989, 0.993, 1.0))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["linewidth"] = 0.7
        axis._axinfo["grid"]["color"] = "#D9E1EA"
        axis._axinfo["axisline"]["linewidth"] = 0.8
        axis._axinfo["axisline"]["color"] = "#B8C2CC"
    ax.tick_params(pad=3)


def _activation_label_3d(tau: float, b_t: float, c_t: float) -> str:
    return f"τ{tau:.0f}  B{b_t:.0f}  C{c_t:.0f}"


def _nearest_tariff_profile(results: List[Exp4ScenarioResult], target: float) -> Exp4ScenarioResult:
    if not results:
        raise ValueError("At least one tariff profile is required.")
    return min(results, key=lambda r: abs(float(r.t2_tariff) - float(target)))


def plot_tariff_profiles(results: List[Exp4ScenarioResult], png_path: str, pdf_path: str) -> None:
    """Plot the full profile ensemble while keeping key paths legible at journal size.

    All non-reference paths are retained as a neutral context layer. The 5%, 50%,
    and 100% period-2 anchors are then redrawn with high-contrast colours, distinct
    line patterns, sparse markers, and a white halo so that they remain visible when
    the panel is reduced and placed beside Experiment 5.
    """
    _apply_theme()
    ordered = sorted(results, key=lambda r: float(r.t2_tariff))
    fig, ax = plt.subplots(figsize=(10.4, 5.9))

    # Context layer: retain every simulated path without letting the ensemble
    # overwhelm the three reference profiles used in the paper discussion.
    for r in ordered:
        ax.plot(
            range(1, len(r.jo_kc_series) + 1),
            [_pct(v) for v in r.jo_kc_series],
            color=PALETTE["PROFILE_GRAY"],
            alpha=0.34,
            linewidth=0.95,
            solid_capstyle="round",
            zorder=1,
        )

    reference_specs = [
        (0.05, PALETTE["PROFILE_LOW"], "-", "o"),
        (0.50, PALETTE["PROFILE_MID"], (0, (6.0, 2.3)), "s"),
        (1.00, PALETTE["PROFILE_HIGH"], (0, (1.5, 1.7)), "D"),
    ]
    halo = [
        path_effects.Stroke(linewidth=4.4, foreground="white", alpha=0.95),
        path_effects.Normal(),
    ]
    for target, color, linestyle, marker in reference_specs:
        r = _nearest_tariff_profile(ordered, target)
        ax.plot(
            range(1, len(r.jo_kc_series) + 1),
            [_pct(v) for v in r.jo_kc_series],
            color=color,
            linestyle=linestyle,
            linewidth=2.55,
            marker=marker,
            markevery=(1, 5),
            markersize=5.4,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.25,
            solid_capstyle="round",
            label=f"τ₂={int(round(_pct(r.t2_tariff)))}%",
            path_effects=halo,
            zorder=4,
        )

    # The period-2 intervention point is shown subtly rather than annotated,
    # preserving space in the small two-panel layout.
    ax.axvline(2, color="#7D8792", linewidth=0.9, linestyle=(0, (2.5, 2.5)), alpha=0.60, zorder=0)
    ax.set_xlabel("Time period")
    ax.set_ylabel("Tariff rate (%)")
    ax.margins(x=0.01)
    _style_2d_axes(ax)
    ax.legend(
        loc="upper left",
        ncol=3,
        fontsize=10.0,
        handlelength=2.8,
        columnspacing=1.15,
        borderpad=0.55,
    )
    _save(fig, png_path, pdf_path)


def plot_total_costs(results: List[Exp4ScenarioResult], png_path: str, pdf_path: str) -> None:
    _apply_theme()
    xs = [_pct(r.t2_tariff) for r in results]
    fig, ax = plt.subplots(figsize=(10.1, 5.9))
    _line(ax, xs, [r.total_A for r in results], PALETTE["A"], "Strategy A", marker="o")
    _line(ax, xs, [r.total_B for r in results], PALETTE["B"], "Strategy B", marker="s")
    _line(ax, xs, [r.total_C for r in results], PALETTE["C"], "Strategy C", marker="D")
    ax.set_title("Experiment 4  Discounted total cost across period-2 M1→D1 tariff anchors")
    ax.set_xlabel("Period-2 tariff on M1→D1 (%)")
    ax.set_ylabel("Discounted total cost")
    _style_2d_axes(ax)
    ax.legend(loc="upper left", ncol=3, fontsize=10.5)
    _save(fig, png_path, pdf_path)


def plot_total_demand(results: List[Exp4ScenarioResult], png_path: str, pdf_path: str) -> None:
    _apply_theme()
    xs = [_pct(r.t2_tariff) for r in results]
    fig, ax = plt.subplots(figsize=(10.1, 5.9))
    _line(ax, xs, [r.demand_A for r in results], PALETTE["A"], "Strategy A", marker="o")
    _line(ax, xs, [r.demand_B for r in results], PALETTE["B"], "Strategy B", marker="s")
    _line(ax, xs, [r.demand_C for r in results], PALETTE["C"], "Strategy C", marker="D")
    ax.set_title("Experiment 4  Total realised demand across period-2 M1→D1 tariff anchors")
    ax.set_xlabel("Period-2 tariff on M1→D1 (%)")
    ax.set_ylabel("Total realised demand")
    _style_2d_axes(ax)
    ax.legend(loc="upper left", ncol=3, fontsize=10.5)
    _save(fig, png_path, pdf_path)


def plot_action_timing(results: List[Exp4ScenarioResult], png_path: str, pdf_path: str) -> None:
    _apply_theme()
    xs = [_pct(r.t2_tariff) for r in results]
    fig, ax = plt.subplots(figsize=(10.1, 5.9))
    _line(ax, xs, [r.activation_B for r in results], PALETTE["B"], "B activation", marker="s")
    _line(ax, xs, [r.activation_C for r in results], PALETTE["C"], "C activation", marker="D")
    _line(ax, xs, [r.withdrawal_C for r in results], PALETTE["A"], "C withdrawal", marker="o")
    ax.set_title("Experiment 4  Action timing across period-2 M1→D1 tariff anchors")
    ax.set_xlabel("Period-2 tariff on M1→D1 (%)")
    ax.set_ylabel("Period")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_2d_axes(ax)
    ax.legend(loc="upper left", ncol=3, fontsize=10.5)
    _save(fig, png_path, pdf_path)


def plot_activation_scatter(results: List[Exp4ScenarioResult], png_path: str, pdf_path: str) -> None:
    _apply_theme()
    xs = [_pct(r.t2_tariff) for r in results]
    ys = [int(r.activation_B) for r in results]
    zs = [int(r.activation_C) for r in results]

    fig = plt.figure(figsize=(11.0, 7.1))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        xs,
        ys,
        zs,
        c=xs,
        cmap="cividis",
        s=82,
        depthshade=True,
        edgecolors="#F7FAFC",
        linewidths=0.8,
        alpha=0.97,
    )
    for idx, (x, y, z) in enumerate(zip(xs, ys, zs)):
        dx = 1.0 if idx % 2 == 0 else -3.4
        dy = 0.06 if idx % 3 == 0 else -0.08
        dz = 0.05 if idx % 2 == 0 else -0.08
        ax.text(
            x + dx,
            y + dy,
            z + dz,
            _activation_label_3d(x, y, z),
            fontsize=6.7,
            color=PALETTE["TEXT"],
            alpha=0.88,
        )
    ax.set_title("Experiment 4  Three-dimensional activation map for Strategy B and Strategy C", pad=14)
    ax.set_xlabel("Period-2 tariff on M1→D1 (%)", labelpad=10)
    ax.set_ylabel("Strategy B activation period", labelpad=10)
    ax.set_zlabel("Strategy C activation period", labelpad=10)
    ax.set_xlim(min(xs) - 3.0, max(xs) + 4.0)
    ax.set_ylim(min(ys) - 0.18, max(ys) + 0.35)
    ax.set_zlim(min(zs) - 0.18, max(zs) + 0.35)
    _format_3d_axes(ax)
    cbar = fig.colorbar(sc, ax=ax, pad=0.08, fraction=0.03)
    cbar.outline.set_edgecolor("#C7D0DA")
    cbar.ax.set_ylabel("Tariff anchor (%)", rotation=270, labelpad=15)
    _save(fig, png_path, pdf_path)
