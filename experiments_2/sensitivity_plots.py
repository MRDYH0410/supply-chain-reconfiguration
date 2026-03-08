from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def _set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def _shade_phases(ax, spec: LinePlotSpec, ymax: float) -> None:
    p1s, p1e = spec.phase1_periods
    p2s, p2e = spec.phase2_periods
    p3s, p3e = spec.phase3_periods

    if p1s <= p1e:
        ax.axvspan(p1s - 0.5, p1e + 0.5, alpha=0.08)
        ax.text((p1s + p1e) / 2.0, ymax * 0.97, "Phase I", ha="center", va="top", fontsize=9)
    if p2s <= p2e:
        ax.axvspan(p2s - 0.5, p2e + 0.5, alpha=0.05)
        ax.text((p2s + p2e) / 2.0, ymax * 0.97, "Phase II", ha="center", va="top", fontsize=9)
    if p3s <= p3e:
        ax.axvspan(p3s - 0.5, p3e + 0.5, alpha=0.03)
        ax.text((p3s + p3e) / 2.0, ymax * 0.97, "Phase III", ha="center", va="top", fontsize=9)

    ax.axvline(spec.phase2_periods[0] - 0.5, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(spec.phase3_periods[0] - 0.5, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(spec.change_period_1 - 0.5, linestyle=":", linewidth=1.2, alpha=0.95)
    ax.axvline(spec.change_period_2 - 0.5, linestyle=":", linewidth=1.2, alpha=0.95)

    ax.text(spec.change_period_1, ymax * 0.90, f"update 1\nt={spec.change_period_1}", ha="center", va="top", fontsize=8)
    ax.text(spec.change_period_2, ymax * 0.90, f"update 2\nt={spec.change_period_2}", ha="center", va="top", fontsize=8)


def plot_single_tariff_path_lineplot(spec: LinePlotSpec, out_png: str, out_pdf: str) -> None:
    _set_pub_style()

    H = len(spec.meanA)
    x = np.arange(1, H + 1)
    y_max = max(
        max(spec.meanA) if spec.meanA else 0.0,
        max(spec.meanB) if spec.meanB else 0.0,
        max(spec.meanC) if spec.meanC else 0.0,
    )
    y_max = max(1.0, y_max * 1.16)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    _shade_phases(ax, spec, y_max)

    A = np.array(spec.meanA, dtype=float)
    B = np.array(spec.meanB, dtype=float)
    C = np.array(spec.meanC, dtype=float)
    sA = np.array(spec.stdA, dtype=float)
    sB = np.array(spec.stdB, dtype=float)
    sC = np.array(spec.stdC, dtype=float)

    ax.plot(x, A, marker="o", linewidth=1.7, label=f"Strategy A  total={spec.totalA:.1f}")
    ax.plot(x, B, marker="s", linewidth=1.7, label=f"Strategy B  total={spec.totalB:.1f}")
    ax.plot(x, C, marker="^", linewidth=1.7, label=f"Strategy C  total={spec.totalC:.1f}")

    ax.fill_between(x, A - sA, A + sA, alpha=0.12)
    ax.fill_between(x, B - sB, B + sB, alpha=0.12)
    ax.fill_between(x, C - sC, C + sC, alpha=0.12)

    ax.set_xlim(1, H)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks(x)
    ax.set_xlabel("Time period")
    ax.set_ylabel("Mean period cost")
    ax.set_title(f"{spec.path_id} | {spec.path_label}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_tariff_path_lineplots(specs: List[LinePlotSpec], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for spec in specs:
        stem = spec.path_label.replace(" ", "_").replace("|", "").replace(">", "")
        png_path = os.path.join(output_dir, f"{spec.path_id}_{stem}.png")
        pdf_path = os.path.join(output_dir, f"{spec.path_id}_{stem}.pdf")
        plot_single_tariff_path_lineplot(spec=spec, out_png=png_path, out_pdf=pdf_path)


def plot_tariff_path_contact_sheet(specs: List[LinePlotSpec], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    specs = list(specs)
    if not specs:
        return

    fig, axes = plt.subplots(4, 4, figsize=(18.0, 12.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, spec in zip(axes, specs):
        H = len(spec.meanA)
        x = np.arange(1, H + 1)
        y_max = max(
            max(spec.meanA) if spec.meanA else 0.0,
            max(spec.meanB) if spec.meanB else 0.0,
            max(spec.meanC) if spec.meanC else 0.0,
        )
        y_max = max(1.0, y_max * 1.10)

        _shade_phases(ax, spec, y_max)
        A = np.array(spec.meanA, dtype=float)
        B = np.array(spec.meanB, dtype=float)
        C = np.array(spec.meanC, dtype=float)

        ax.plot(x, A, linewidth=1.2)
        ax.plot(x, B, linewidth=1.2)
        ax.plot(x, C, linewidth=1.2)
        ax.set_title(f"{spec.path_id}  {spec.path_label}", fontsize=9)
        ax.set_xlim(1, H)
        ax.set_ylim(0.0, y_max)
        ax.set_xticks([1, H // 2, H])
        ax.grid(axis="y", alpha=0.18)

    for ax in axes[len(specs):]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], linewidth=1.7, marker="o", label="Strategy A"),
        plt.Line2D([], [], linewidth=1.7, marker="s", label="Strategy B"),
        plt.Line2D([], [], linewidth=1.7, marker="^", label="Strategy C"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _extract_xy(points: List[RampSummaryPoint]):
    return np.array([p.ramp_kappa for p in points], dtype=float)


def plot_ramp_phase_length(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs = _extract_xy(points)
    ys = np.array([p.phase2_length for p in points], dtype=float)
    ramp_age = np.array([p.ramp_full_age for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(xs, ys, marker="o", linewidth=1.8, label="Detected Phase II length")
    ax.plot(xs, ramp_age, marker="s", linewidth=1.5, linestyle="--", label="Theoretical full-ramp age")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Periods")
    ax.set_title("Ramp-up ability and Phase II duration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ramp_total_cost(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs = _extract_xy(points)
    yA = np.array([p.mean_total_cost_A for p in points], dtype=float)
    yB = np.array([p.mean_total_cost_B for p in points], dtype=float)
    yC = np.array([p.mean_total_cost_C for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(xs, yA, marker="o", linewidth=1.8, label="Strategy A")
    ax.plot(xs, yB, marker="s", linewidth=1.8, label="Strategy B")
    ax.plot(xs, yC, marker="^", linewidth=1.8, label="Strategy C")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Mean total cost across 16 tariff paths")
    ax.set_title("Total-cost response to candidate ramp-up ability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ramp_phase2_cost(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs = _extract_xy(points)
    yA = np.array([p.phase2_mean_cost_A for p in points], dtype=float)
    yB = np.array([p.phase2_mean_cost_B for p in points], dtype=float)
    yC = np.array([p.phase2_mean_cost_C for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(xs, yA, marker="o", linewidth=1.8, label="Strategy A")
    ax.plot(xs, yB, marker="s", linewidth=1.8, label="Strategy B")
    ax.plot(xs, yC, marker="^", linewidth=1.8, label="Strategy C")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Mean Phase II period cost across 16 tariff paths")
    ax.set_title("Phase II cost exposure under different ramp-up speeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ramp_cost_gaps(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs = _extract_xy(points)
    gBA = np.array([p.mean_gap_B_minus_A for p in points], dtype=float)
    gCA = np.array([p.mean_gap_C_minus_A for p in points], dtype=float)
    g2BA = np.array([p.phase2_gap_B_minus_A for p in points], dtype=float)
    g2CA = np.array([p.phase2_gap_C_minus_A for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.axhline(0.0, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot(xs, gBA, marker="o", linewidth=1.8, label="B − A overall")
    ax.plot(xs, gCA, marker="^", linewidth=1.8, label="C − A overall")
    ax.plot(xs, g2BA, marker="s", linewidth=1.4, linestyle=":", label="B − A in Phase II")
    ax.plot(xs, g2CA, marker="D", linewidth=1.4, linestyle=":", label="C − A in Phase II")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Cost gap")
    ax.set_title("Strategic value of reconfiguration versus no reconfiguration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ramp_win_share(points: List[RampSummaryPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs = _extract_xy(points)
    wA = np.array([p.win_share_A for p in points], dtype=float)
    wB = np.array([p.win_share_B for p in points], dtype=float)
    wC = np.array([p.win_share_C for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(xs, wA, marker="o", linewidth=1.8, label="Strategy A")
    ax.plot(xs, wB, marker="s", linewidth=1.8, label="Strategy B")
    ax.plot(xs, wC, marker="^", linewidth=1.8, label="Strategy C")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Share of 16 tariff paths won")
    ax.set_title("Strategy dominance across tariff paths")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _extract_path_xy(points: List[PathRampPoint]):
    return np.array([p.ramp_kappa for p in points], dtype=float)


def plot_path_total_cost_vs_kappa(points: List[PathRampPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs = _extract_path_xy(points)
    yA = np.array([p.mean_cost_A for p in points], dtype=float)
    yB = np.array([p.mean_cost_B for p in points], dtype=float)
    yC = np.array([p.mean_cost_C for p in points], dtype=float)
    sA = np.array([p.std_cost_A for p in points], dtype=float)
    sB = np.array([p.std_cost_B for p in points], dtype=float)
    sC = np.array([p.std_cost_C for p in points], dtype=float)

    title = f"{points[0].path_id} | {points[0].path_label}"
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(xs, yA, marker="o", linewidth=1.8, label="Strategy A")
    ax.plot(xs, yB, marker="s", linewidth=1.8, label="Strategy B")
    ax.plot(xs, yC, marker="^", linewidth=1.8, label="Strategy C")
    ax.fill_between(xs, yA - sA, yA + sA, alpha=0.12)
    ax.fill_between(xs, yB - sB, yB + sB, alpha=0.12)
    ax.fill_between(xs, yC - sC, yC + sC, alpha=0.12)
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Mean total cost")
    ax.set_title(f"{title}\nTotal cost by ramp-up ability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_path_phase2_cost_vs_kappa(points: List[PathRampPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs = _extract_path_xy(points)
    yA = np.array([p.phase2_mean_cost_A for p in points], dtype=float)
    yB = np.array([p.phase2_mean_cost_B for p in points], dtype=float)
    yC = np.array([p.phase2_mean_cost_C for p in points], dtype=float)

    title = f"{points[0].path_id} | {points[0].path_label}"
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(xs, yA, marker="o", linewidth=1.8, label="Strategy A")
    ax.plot(xs, yB, marker="s", linewidth=1.8, label="Strategy B")
    ax.plot(xs, yC, marker="^", linewidth=1.8, label="Strategy C")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Mean Phase II period cost")
    ax.set_title(f"{title}\nPhase II cost by ramp-up ability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_path_phase2_length_vs_kappa(points: List[PathRampPoint], out_png: str, out_pdf: str) -> None:
    _set_pub_style()
    if not points:
        return
    xs = _extract_path_xy(points)
    phase2_len = np.array([p.phase2_length for p in points], dtype=float)
    phase3_start = np.array([p.phase3_start for p in points], dtype=float)

    title = f"{points[0].path_id} | {points[0].path_label}"
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(xs, phase2_len, marker="o", linewidth=1.8, label="Phase II length")
    ax.plot(xs, phase3_start, marker="s", linewidth=1.5, linestyle="--", label="Phase III start period")
    ax.set_xlabel("Candidate ramp-up parameter κ")
    ax.set_ylabel("Periods")
    ax.set_title(f"{title}\nPhase timing by ramp-up ability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)