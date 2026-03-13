from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from experiments_2.sensitivity_runner import (
    JOINT_STATE_TO_ID,
    STATE_ORDER,
    PhaseStructure,
    TariffLevelConfig,
    TariffPathSpec,
    TrainEvalConfig,
    evaluate_three_strategies,
    joint_state_full_label,
    make_joint_tariff_scenario_base,
    make_tariff_path_scenario,
)
from experiments_3.sensitivity_runner import detect_phase_structure_via_strategy_b


@dataclass(frozen=True)
class DurationProfile:
    profile_id: str
    profile_name: str
    phase2_high_scale: float
    phase2_low_scale: float
    phase3_high_scale: float
    phase3_low_scale: float
    note: str = ""

    @property
    def short_label(self) -> str:
        return (
            f"{self.profile_id} | "
            f"P2(H={self.phase2_high_scale:.2f}, L={self.phase2_low_scale:.2f}) | "
            f"P3(H={self.phase3_high_scale:.2f}, L={self.phase3_low_scale:.2f})"
        )


@dataclass(frozen=True)
class DurationAllocation:
    phase1_length: int
    phase2_length: int
    phase3_length: int
    phase2_weight: float
    phase3_weight: float
    phase2_start: int
    phase2_end: int
    phase3_start: int
    phase3_end: int


def build_default_duration_profiles() -> List[DurationProfile]:
    """
    Default grid for experiment 4.

    Interpretation
    - phase2_high_scale / phase2_low_scale control the persistence weight of H and L inside Phase II
    - phase3_high_scale / phase3_low_scale control the persistence weight of H and L inside Phase III

    Because the current codebase uses one common phase boundary for the joint tariff state,
    the H/L persistence of the two market sides is aggregated into a single joint-state weight
    by averaging the two sides. HH therefore receives the full high weight, LL the full low
    weight, and HL/LH the midpoint between them.
    """
    return [
        DurationProfile(
            "D01",
            "baseline",
            1.00,
            1.00,
            1.00,
            1.00,
            "Reference timing inherited from the baseline phase split.",
        ),
        DurationProfile(
            "D02",
            "phase2_high_longer",
            1.35,
            0.85,
            1.00,
            1.00,
            "In Phase II, high-tariff states persist longer while low-tariff states compress.",
        ),
        DurationProfile(
            "D03",
            "phase2_low_longer",
            0.85,
            1.35,
            1.00,
            1.00,
            "In Phase II, low-tariff states persist longer while high-tariff states compress.",
        ),
        DurationProfile(
            "D04",
            "phase3_high_longer",
            1.00,
            1.00,
            1.35,
            0.85,
            "In Phase III, high-tariff states persist longer while low-tariff states compress.",
        ),
        DurationProfile(
            "D05",
            "phase3_low_longer",
            1.00,
            1.00,
            0.85,
            1.35,
            "In Phase III, low-tariff states persist longer while high-tariff states compress.",
        ),
        DurationProfile(
            "D06",
            "high_persistent_both",
            1.25,
            0.90,
            1.25,
            0.90,
            "High-tariff persistence is strengthened in both post-opening phases.",
        ),
        DurationProfile(
            "D07",
            "low_persistent_both",
            0.90,
            1.25,
            0.90,
            1.25,
            "Low-tariff persistence is strengthened in both post-opening phases.",
        ),
        DurationProfile(
            "D08",
            "front_high_back_low",
            1.30,
            0.85,
            0.85,
            1.30,
            "Tariff pressure persists longer during ramp-up but relaxes for longer after ramp-up.",
        ),
    ]


def _state_persistence_weight(state: str, high_scale: float, low_scale: float) -> float:
    weights = [float(high_scale) if ch == "H" else float(low_scale) for ch in str(state)]
    return float(sum(weights) / len(weights)) if weights else 1.0


def allocate_phase_lengths_for_path(
    base_phase_structure: PhaseStructure,
    horizon: int,
    profile: DurationProfile,
    phase2_state: str,
    phase3_state: str,
) -> DurationAllocation:
    # The tariff-path tree in experiments 1–3 is governed by change_period_1 and change_period_2.
    # Experiment 4 therefore perturbs the lengths of the second and third tariff-state blocks,
    # while keeping the total horizon fixed.
    phase1_length = max(0, int(base_phase_structure.change_period_1) - 1)
    base_phase2_length = max(1, int(base_phase_structure.change_period_2 - base_phase_structure.change_period_1))
    base_phase3_length = max(1, int(horizon - base_phase_structure.change_period_2 + 1))
    remainder = max(2, int(horizon) - phase1_length)

    phase2_weight = _state_persistence_weight(
        phase2_state,
        high_scale=profile.phase2_high_scale,
        low_scale=profile.phase2_low_scale,
    )
    phase3_weight = _state_persistence_weight(
        phase3_state,
        high_scale=profile.phase3_high_scale,
        low_scale=profile.phase3_low_scale,
    )

    desired_phase2 = float(base_phase2_length) * float(phase2_weight)
    desired_phase3 = float(base_phase3_length) * float(phase3_weight)
    total_desired = max(1.0e-12, desired_phase2 + desired_phase3)

    phase2_length = int(round(remainder * desired_phase2 / total_desired))
    phase2_length = max(1, min(remainder - 1, phase2_length))
    phase3_length = max(1, remainder - phase2_length)

    phase2_start = int(base_phase_structure.change_period_1)
    phase2_end = phase2_start + phase2_length - 1
    phase3_start = phase2_end + 1
    phase3_end = int(horizon)

    return DurationAllocation(
        phase1_length=phase1_length,
        phase2_length=phase2_length,
        phase3_length=phase3_length,
        phase2_weight=float(phase2_weight),
        phase3_weight=float(phase3_weight),
        phase2_start=int(phase2_start),
        phase2_end=int(phase2_end),
        phase3_start=int(phase3_start),
        phase3_end=int(phase3_end),
    )


def build_duration_sensitive_path_specs(
    base_phase_structure: PhaseStructure,
    horizon: int,
    profile: DurationProfile,
) -> List[TariffPathSpec]:
    phase1_state = "HL"
    specs: List[TariffPathSpec] = []
    counter = 1
    H = int(horizon)

    for s2 in STATE_ORDER:
        for s3 in STATE_ORDER:
            alloc = allocate_phase_lengths_for_path(
                base_phase_structure=base_phase_structure,
                horizon=H,
                profile=profile,
                phase2_state=s2,
                phase3_state=s3,
            )
            cp1 = int(alloc.phase2_start)
            cp2 = int(alloc.phase3_start)

            xi_path: List[int] = []
            for t in range(1, H + 1):
                if t < cp1:
                    state = phase1_state
                elif t < cp2:
                    state = s2
                else:
                    state = s3
                xi_path.append(JOINT_STATE_TO_ID[state])

            specs.append(
                TariffPathSpec(
                    path_id=f"P{counter:02d}",
                    phase1_state=phase1_state,
                    phase2_state=s2,
                    phase3_state=s3,
                    phase1_periods=(1, max(0, cp1 - 1)),
                    phase2_periods=(cp1, max(cp1, cp2 - 1)),
                    phase3_periods=(cp2, H),
                    change_period_1=cp1,
                    change_period_2=cp2,
                    xi_path=xi_path,
                )
            )
            counter += 1
    return specs


__all__ = [
    "DurationAllocation",
    "DurationProfile",
    "TariffLevelConfig",
    "TrainEvalConfig",
    "allocate_phase_lengths_for_path",
    "build_default_duration_profiles",
    "build_duration_sensitive_path_specs",
    "detect_phase_structure_via_strategy_b",
    "evaluate_three_strategies",
    "joint_state_full_label",
    "make_joint_tariff_scenario_base",
    "make_tariff_path_scenario",
]