from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt


@dataclass
class FixedScenarioPlotSpec:
    periods: List[int]
    tariff_schedule: Dict[str, Dict[str, List[float]]]

    mean_period_cost_A: List[float]
    mean_period_cost_B: List[float]
    mean_period_cost_C: List[float]

    mean_cum_discounted_A: List[float]
    mean_cum_discounted_B: List[float]
    mean_cum_discounted_C: List[float]

    std_period_cost_A: List[float]
    std_period_cost_B: List[float]
    std_period_cost_C: List[float]

    std_cum_discounted_A: List[float]
    std_cum_discounted_B: List[float]
    std_cum_discounted_C: List[float]

    total_discounted_A: float
    total_discounted_B: float
    total_discounted_C: float

    activation_period_B: int
    activation_period_C: int
    withdrawal_period_C: int


@dataclass
class DemandDashboardPlotSpec:
    periods: List[int]
    tariff_schedule: Dict[str, Dict[str, List[float]]]

    mean_period_demand_A: List[float]
    mean_period_demand_B: List[float]
    mean_period_demand_C: List[float]

    mean_cum_demand_A: List[float]
    mean_cum_demand_B: List[float]
    mean_cum_demand_C: List[float]

    std_period_demand_A: List[float]
    std_period_demand_B: List[float]
    std_period_demand_C: List[float]

    std_cum_demand_A: List[float]
    std_cum_demand_B: List[float]
    std_cum_demand_C: List[float]

    total_demand_A: float
    total_demand_B: float
    total_demand_C: float

    activation_period_B: int
    activation_period_C: int
    withdrawal_period_C: int


@dataclass
class ActionTracePlotSpec:
    periods: List[int]
    u_B: List[float]
    u_C: List[float]
    v_C: List[float]
    a_M2_B: List[float]
    a_M2_C: List[float]
    age_M2_B: List[float]
    age_M2_C: List[float]


@dataclass
class CapacityTracePlotSpec:
    periods: List[int]

    cap_M1_A: List[float]
    cap_M2_A: List[float]
    y_M1_A: List[float]
    y_M2_A: List[float]

    cap_M1_B: List[float]
    cap_M2_B: List[float]
    y_M1_B: List[float]
    y_M2_B: List[float]

    cap_M1_C: List[float]
    cap_M2_C: List[float]
    y_M1_C: List[float]
    y_M2_C: List[float]


def _save_fig(fig, png_path: str, pdf_path: str) -> None:
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _draw_tariff_panel(ax, periods: List[int], tariff_schedule: Dict[str, Dict[str, List[float]]]) -> None:
    jo = list(tariff_schedule.keys())[0]
    jn = list(tariff_schedule.keys())[1]
    kc = list(tariff_schedule[jo].keys())[0]
    ka = list(tariff_schedule[jo].keys())[1]

    ax.plot(periods, tariff_schedule[jo][kc], label=f"{jo}->{kc}")
    ax.plot(periods, tariff_schedule[jn][kc], label=f"{jn}->{kc}")
    ax.plot(periods, tariff_schedule[jo][ka], label=f"{jo}->{ka}")
    ax.plot(periods, tariff_schedule[jn][ka], label=f"{jn}->{ka}")
    ax.set_title("Fixed 30-period tariff schedule")
    ax.set_xlabel("Time period")
    ax.set_ylabel("Tariff rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _add_strategy_markers(ax, activation_period_B: int, activation_period_C: int, withdrawal_period_C: int) -> None:
    if activation_period_B > 0:
        ax.axvline(activation_period_B, linestyle="--", linewidth=1.0)
    if activation_period_C > 0:
        ax.axvline(activation_period_C, linestyle=":", linewidth=1.0)
    if withdrawal_period_C > 0:
        ax.axvline(withdrawal_period_C, linestyle="-.", linewidth=1.0)


def plot_fixed_baseline_dashboard(spec: FixedScenarioPlotSpec, png_path: str, pdf_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    t = spec.periods

    _draw_tariff_panel(ax1, t, spec.tariff_schedule)

    ax2.plot(t, spec.mean_period_cost_A, label="Strategy A")
    ax2.plot(t, spec.mean_period_cost_B, label="Strategy B")
    ax2.plot(t, spec.mean_period_cost_C, label="Strategy C")
    ax2.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_period_cost_A, spec.std_period_cost_A)],
        [a + b for a, b in zip(spec.mean_period_cost_A, spec.std_period_cost_A)],
        alpha=0.15,
    )
    ax2.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_period_cost_B, spec.std_period_cost_B)],
        [a + b for a, b in zip(spec.mean_period_cost_B, spec.std_period_cost_B)],
        alpha=0.15,
    )
    ax2.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_period_cost_C, spec.std_period_cost_C)],
        [a + b for a, b in zip(spec.mean_period_cost_C, spec.std_period_cost_C)],
        alpha=0.15,
    )
    _add_strategy_markers(ax2, spec.activation_period_B, spec.activation_period_C, spec.withdrawal_period_C)
    ax2.set_title("Per-period cost")
    ax2.set_xlabel("Time period")
    ax2.set_ylabel("Cost")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, spec.mean_cum_discounted_A, label=f"A total={spec.total_discounted_A:.1f}")
    ax3.plot(t, spec.mean_cum_discounted_B, label=f"B total={spec.total_discounted_B:.1f}")
    ax3.plot(t, spec.mean_cum_discounted_C, label=f"C total={spec.total_discounted_C:.1f}")
    ax3.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_cum_discounted_A, spec.std_cum_discounted_A)],
        [a + b for a, b in zip(spec.mean_cum_discounted_A, spec.std_cum_discounted_A)],
        alpha=0.15,
    )
    ax3.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_cum_discounted_B, spec.std_cum_discounted_B)],
        [a + b for a, b in zip(spec.mean_cum_discounted_B, spec.std_cum_discounted_B)],
        alpha=0.15,
    )
    ax3.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_cum_discounted_C, spec.std_cum_discounted_C)],
        [a + b for a, b in zip(spec.mean_cum_discounted_C, spec.std_cum_discounted_C)],
        alpha=0.15,
    )
    _add_strategy_markers(ax3, spec.activation_period_B, spec.activation_period_C, spec.withdrawal_period_C)
    ax3.set_title("Cumulative discounted cost")
    ax3.set_xlabel("Time period")
    ax3.set_ylabel("Cum discounted cost")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4.axis("off")
    lines = [
        "Experiment 1 baseline redesign",
        "Single fixed tariff scenario",
        "H = 30 periods",
        f"Strategy B activation period = {spec.activation_period_B if spec.activation_period_B > 0 else 'none'}",
        f"Strategy C activation period = {spec.activation_period_C if spec.activation_period_C > 0 else 'none'}",
        f"Strategy C withdrawal period = {spec.withdrawal_period_C if spec.withdrawal_period_C > 0 else 'none'}",
        "",
        "Main comparison",
        f"  A cumulative discounted total = {spec.total_discounted_A:.2f}",
        f"  B cumulative discounted total = {spec.total_discounted_B:.2f}",
        f"  C cumulative discounted total = {spec.total_discounted_C:.2f}",
    ]
    ax4.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )

    fig.suptitle("Experiment 1  Fixed 30-period tariff baseline", fontsize=14, y=0.98)
    fig.tight_layout()
    _save_fig(fig, png_path, pdf_path)


def plot_demand_dashboard(spec: DemandDashboardPlotSpec, png_path: str, pdf_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    t = spec.periods

    _draw_tariff_panel(ax1, t, spec.tariff_schedule)

    ax2.plot(t, spec.mean_period_demand_A, label="Strategy A")
    ax2.plot(t, spec.mean_period_demand_B, label="Strategy B")
    ax2.plot(t, spec.mean_period_demand_C, label="Strategy C")
    ax2.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_period_demand_A, spec.std_period_demand_A)],
        [a + b for a, b in zip(spec.mean_period_demand_A, spec.std_period_demand_A)],
        alpha=0.15,
    )
    ax2.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_period_demand_B, spec.std_period_demand_B)],
        [a + b for a, b in zip(spec.mean_period_demand_B, spec.std_period_demand_B)],
        alpha=0.15,
    )
    ax2.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_period_demand_C, spec.std_period_demand_C)],
        [a + b for a, b in zip(spec.mean_period_demand_C, spec.std_period_demand_C)],
        alpha=0.15,
    )
    _add_strategy_markers(ax2, spec.activation_period_B, spec.activation_period_C, spec.withdrawal_period_C)
    ax2.set_title("Per-period realised demand")
    ax2.set_xlabel("Time period")
    ax2.set_ylabel("Demand volume")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, spec.mean_cum_demand_A, label=f"A total={spec.total_demand_A:.1f}")
    ax3.plot(t, spec.mean_cum_demand_B, label=f"B total={spec.total_demand_B:.1f}")
    ax3.plot(t, spec.mean_cum_demand_C, label=f"C total={spec.total_demand_C:.1f}")
    ax3.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_cum_demand_A, spec.std_cum_demand_A)],
        [a + b for a, b in zip(spec.mean_cum_demand_A, spec.std_cum_demand_A)],
        alpha=0.15,
    )
    ax3.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_cum_demand_B, spec.std_cum_demand_B)],
        [a + b for a, b in zip(spec.mean_cum_demand_B, spec.std_cum_demand_B)],
        alpha=0.15,
    )
    ax3.fill_between(
        t,
        [a - b for a, b in zip(spec.mean_cum_demand_C, spec.std_cum_demand_C)],
        [a + b for a, b in zip(spec.mean_cum_demand_C, spec.std_cum_demand_C)],
        alpha=0.15,
    )
    _add_strategy_markers(ax3, spec.activation_period_B, spec.activation_period_C, spec.withdrawal_period_C)
    ax3.set_title("Cumulative realised demand")
    ax3.set_xlabel("Time period")
    ax3.set_ylabel("Cum demand volume")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4.axis("off")
    lines = [
        "Experiment 1 demand comparison",
        "Single fixed tariff scenario",
        "H = 30 periods",
        f"Strategy B activation period = {spec.activation_period_B if spec.activation_period_B > 0 else 'none'}",
        f"Strategy C activation period = {spec.activation_period_C if spec.activation_period_C > 0 else 'none'}",
        f"Strategy C withdrawal period = {spec.withdrawal_period_C if spec.withdrawal_period_C > 0 else 'none'}",
        "",
        "Main comparison",
        f"  A realised total demand = {spec.total_demand_A:.2f}",
        f"  B realised total demand = {spec.total_demand_B:.2f}",
        f"  C realised total demand = {spec.total_demand_C:.2f}",
    ]
    ax4.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )

    fig.suptitle("Experiment 1  Fixed 30-period realised demand baseline", fontsize=14, y=0.98)
    fig.tight_layout()
    _save_fig(fig, png_path, pdf_path)


def plot_action_trace(spec: ActionTracePlotSpec, png_path: str, pdf_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ax1, ax2, ax3 = axes

    t = spec.periods

    ax1.step(t, spec.u_B, where="post", label="u_t for Strategy B")
    ax1.step(t, spec.u_C, where="post", label="u_t for Strategy C")
    ax1.step(t, spec.v_C, where="post", label="v_t for Strategy C")
    ax1.set_ylabel("Decision")
    ax1.set_title("Structural actions")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.step(t, spec.a_M2_B, where="post", label="a_M2 under Strategy B")
    ax2.step(t, spec.a_M2_C, where="post", label="a_M2 under Strategy C")
    ax2.set_ylabel("Plant status")
    ax2.set_title("Candidate plant operational status")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, spec.age_M2_B, label="age_M2 under Strategy B")
    ax3.plot(t, spec.age_M2_C, label="age_M2 under Strategy C")
    ax3.set_xlabel("Time period")
    ax3.set_ylabel("Age")
    ax3.set_title("Candidate plant age")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Experiment 1  Structural action trace", fontsize=14, y=0.98)
    fig.tight_layout()
    _save_fig(fig, png_path, pdf_path)


def plot_capacity_trace(spec: CapacityTracePlotSpec, png_path: str, pdf_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    t = spec.periods

    panels = [
        (
            axes[0],
            "Strategy A  plant capacity and realised production",
            spec.cap_M1_A,
            spec.cap_M2_A,
            spec.y_M1_A,
            spec.y_M2_A,
        ),
        (
            axes[1],
            "Strategy B  plant capacity and realised production",
            spec.cap_M1_B,
            spec.cap_M2_B,
            spec.y_M1_B,
            spec.y_M2_B,
        ),
        (
            axes[2],
            "Strategy C  plant capacity and realised production",
            spec.cap_M1_C,
            spec.cap_M2_C,
            spec.y_M1_C,
            spec.y_M2_C,
        ),
    ]

    for ax, title, cap1, cap2, y1, y2 in panels:
        ax.plot(t, cap1, linewidth=2.0, label="M1 effective capacity")
        ax.plot(t, cap2, linewidth=2.0, label="M2 effective capacity")
        ax.step(t, y1, where="mid", linestyle="--", linewidth=1.8, label="M1 realised production")
        ax.step(t, y2, where="mid", linestyle="--", linewidth=1.8, label="M2 realised production")
        ax.set_ylabel("Thousand vehicles")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    axes[2].set_xlabel("Time period")

    fig.suptitle("Experiment 1  Two-plant capacity and realised production trace", fontsize=14, y=0.98)
    fig.tight_layout()
    _save_fig(fig, png_path, pdf_path)
