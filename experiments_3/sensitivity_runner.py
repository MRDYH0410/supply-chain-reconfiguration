from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import copy

from scenarios.scenario import SupplyChainScenario

from experiments_2.sensitivity_runner import (
    JOINT_STATE_TO_ID,
    PhaseStructure,
    TariffLevelConfig,
    TrainEvalConfig,
    build_tariff_path_specs_from_phase_structure,
    detect_phase_structure_via_strategy_b as _detect_phase_structure_via_strategy_b_exp2,
    evaluate_three_strategies,
    extract_tariff_level_values,
    joint_state_full_label,
    make_scenario_base,
)


@dataclass(frozen=True)
class TariffProfile:
    profile_id: str
    profile_name: str
    low_scale: float
    high_scale: float
    note: str = ""

    @property
    def short_label(self) -> str:
        return f"{self.profile_id} | L×{self.low_scale:.2f}, H×{self.high_scale:.2f}"


def _fallback_phase_structure_for_profile(scenario: SupplyChainScenario) -> PhaseStructure:
    H = int(scenario.H)
    activation_period = 1
    ramp_full_period = min(H, activation_period + int(max(1, scenario.ramp_full_age)))
    phase2_start = max(1, min(H, activation_period))
    phase3_start = max(phase2_start + 1, min(H, ramp_full_period))
    change_period_1 = max(phase2_start, min(H, (phase2_start + phase3_start) // 2))
    change_period_2 = max(phase3_start, min(H, (phase3_start + H) // 2))
    return PhaseStructure(
        phase1_periods=(1, phase2_start - 1),
        phase2_periods=(phase2_start, max(phase2_start, phase3_start - 1)),
        phase3_periods=(phase3_start, H),
        phase2_start=phase2_start,
        phase3_start=phase3_start,
        change_period_1=change_period_1,
        change_period_2=change_period_2,
        activation_period_B=activation_period,
        ramp_full_period_B=ramp_full_period,
    )


def detect_phase_structure_via_strategy_b(
    seeds: List[int],
    cfg: TrainEvalConfig,
    *,
    level_cfg: Optional[TariffLevelConfig] = None,
    trace_episode_seed_base: int = 880000,
    scenario: Optional[SupplyChainScenario] = None,
) -> PhaseStructure:
    """
    Safe wrapper around experiment 2 phase detection.

    When quick test settings fail to activate the candidate plant, the experiment 2
    helper tries to read a checkpoint trained under a different observation shape,
    which may raise a state-dict mismatch. In that case we fall back to a
    deterministic phase split based on the current scenario ramp target.
    """
    try:
        return _detect_phase_structure_via_strategy_b_exp2(
            seeds=seeds,
            cfg=cfg,
            level_cfg=level_cfg,
            trace_episode_seed_base=trace_episode_seed_base,
            scenario=scenario,
        )
    except RuntimeError as e:
        if scenario is None:
            scenario = make_scenario_base(validate=False)
        print(f"[phase detection] checkpoint-based recovery skipped due to shape mismatch: {e}")
        print("[phase detection] using deterministic fallback split derived from the current ramp target.")
        return _fallback_phase_structure_for_profile(scenario)


@dataclass(frozen=True)
class TariffProfileSpec:
    profile: TariffProfile
    tariff_levels: Dict[str, Dict[str, Dict[str, float]]]


def build_default_tariff_profiles() -> List[TariffProfile]:
    """
    Default grid for experiment 3.

    Interpretation
    - low_scale multiplies all low-tariff states
    - high_scale multiplies all high-tariff states

    This keeps the 16 path structure unchanged while changing the numerical
    intensity behind H and L. Order constraints low <= high are enforced edge by edge.
    """
    return [
        TariffProfile("T01", "baseline", 1.00, 1.00, "Reference tariff levels from the story case."),
        TariffProfile("T02", "milder_all", 0.85, 0.85, "Both low and high states are reduced proportionally."),
        TariffProfile("T03", "stronger_all", 1.15, 1.15, "Both low and high states are strengthened proportionally."),
        TariffProfile("T04", "wider_gap", 0.90, 1.15, "Lower low state and higher high state increase tariff spread."),
        TariffProfile("T05", "compressed_gap", 1.05, 0.95, "Higher low state and lower high state compress the spread."),
        TariffProfile("T06", "high_state_spike", 1.00, 1.30, "Only the H state becomes much harsher."),
    ]


def _clip_tau(value: float, lo: float = 0.005, hi: float = 0.95) -> float:
    return max(lo, min(hi, float(value)))


def _ordered_low_high(raw_low: float, raw_high: float, eps: float = 0.005) -> tuple[float, float]:
    low = _clip_tau(raw_low)
    high = _clip_tau(raw_high)
    if low > high - eps:
        center = min(0.94, max(0.01, 0.5 * (low + high)))
        low = max(0.005, center - 0.5 * eps)
        high = min(0.95, center + 0.5 * eps)
        if low > high - eps:
            low = max(0.005, min(low, high - eps))
            high = min(0.95, max(high, low + eps))
    return float(low), float(high)


def _build_profiled_tariff_levels(
    base: SupplyChainScenario,
    level_cfg: TariffLevelConfig,
    profile: TariffProfile,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    base_levels = extract_tariff_level_values(base, level_cfg)
    out: Dict[str, Dict[str, Dict[str, float]]] = {
        "original_side": {"low": {}, "high": {}},
        "new_side": {"low": {}, "high": {}},
    }

    for side in ["original_side", "new_side"]:
        for market_id in base.markets:
            raw_low = float(base_levels[side]["low"][market_id]) * float(profile.low_scale)
            raw_high = float(base_levels[side]["high"][market_id]) * float(profile.high_scale)
            low_val, high_val = _ordered_low_high(raw_low, raw_high)
            out[side]["low"][market_id] = low_val
            out[side]["high"][market_id] = high_val
    return out


def _build_joint_state_tau_from_profile(
    base: SupplyChainScenario,
    level_cfg: TariffLevelConfig,
    profile: TariffProfile,
) -> Dict[str, Dict[str, Dict[int, float]]]:
    levels = _build_profiled_tariff_levels(base, level_cfg, profile)
    new_tau: Dict[str, Dict[str, Dict[int, float]]] = {
        base.legacy_plant_id: {},
        base.candidate_plant_id: {},
    }
    for market_id in base.markets:
        orig_low = levels["original_side"]["low"][market_id]
        orig_high = levels["original_side"]["high"][market_id]
        new_low = levels["new_side"]["low"][market_id]
        new_high = levels["new_side"]["high"][market_id]

        new_tau[base.legacy_plant_id][market_id] = {
            JOINT_STATE_TO_ID["HH"]: orig_high,
            JOINT_STATE_TO_ID["HL"]: orig_high,
            JOINT_STATE_TO_ID["LH"]: orig_low,
            JOINT_STATE_TO_ID["LL"]: orig_low,
        }
        new_tau[base.candidate_plant_id][market_id] = {
            JOINT_STATE_TO_ID["HH"]: new_high,
            JOINT_STATE_TO_ID["HL"]: new_low,
            JOINT_STATE_TO_ID["LH"]: new_high,
            JOINT_STATE_TO_ID["LL"]: new_low,
        }
    return new_tau


def describe_tariff_profile(
    profile: TariffProfile,
    *,
    level_cfg: Optional[TariffLevelConfig] = None,
    base_scenario: Optional[SupplyChainScenario] = None,
) -> TariffProfileSpec:
    base = copy.deepcopy(base_scenario) if base_scenario is not None else make_scenario_base(validate=False)
    if level_cfg is None:
        level_cfg = TariffLevelConfig()
    return TariffProfileSpec(
        profile=profile,
        tariff_levels=_build_profiled_tariff_levels(base, level_cfg, profile),
    )


def make_joint_tariff_scenario_with_profile(
    profile: TariffProfile,
    *,
    validate: bool = False,
    level_cfg: Optional[TariffLevelConfig] = None,
    base_scenario: Optional[SupplyChainScenario] = None,
) -> SupplyChainScenario:
    base = copy.deepcopy(base_scenario) if base_scenario is not None else make_scenario_base(validate=False)
    scenario = copy.deepcopy(base)
    if level_cfg is None:
        level_cfg = TariffLevelConfig()
    scenario.regimes = [1, 2, 3, 4]
    scenario.P = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    scenario.tau = _build_joint_state_tau_from_profile(base, level_cfg, profile)
    scenario.xi_1 = JOINT_STATE_TO_ID["HL"]
    scenario.xi_2_forced = JOINT_STATE_TO_ID["HL"]
    if validate:
        scenario.validate_assumptions()
    return scenario


def make_profiled_tariff_path_scenario(
    spec,
    profile: TariffProfile,
    *,
    validate: bool = False,
    level_cfg: Optional[TariffLevelConfig] = None,
    base_scenario: Optional[SupplyChainScenario] = None,
) -> SupplyChainScenario:
    scenario = make_joint_tariff_scenario_with_profile(
        profile=profile,
        validate=False,
        level_cfg=level_cfg,
        base_scenario=base_scenario,
    )
    fixed_path = list(spec.xi_path)
    scenario.xi_1 = int(fixed_path[0])
    scenario.xi_2_forced = int(fixed_path[1]) if len(fixed_path) >= 2 else None

    def _fixed_sample_regime_path(seed=None):
        return list(fixed_path)

    scenario.sample_regime_path = _fixed_sample_regime_path  # type: ignore[attr-defined]
    if validate:
        scenario.validate_assumptions()
    return scenario