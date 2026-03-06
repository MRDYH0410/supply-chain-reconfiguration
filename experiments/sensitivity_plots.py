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
        png_path = os.path.join(output_dir, f"{spec.path_id}_{spec.path_label.replace(' ', '_').replace('|', '').replace('>', '')}.png")
        pdf_path = os.path.join(output_dir, f"{spec.path_id}_{spec.path_label.replace(' ', '_').replace('|', '').replace('>', '')}.pdf")
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