from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import copy
import math

import torch

from scenarios.story_case import build_story_case
from scenarios.scenario import SupplyChainScenario

from rl.env.sc_reconfig_env import SCReconfigEnv, Strategy
from rl.env.rollout import rollout_episode
from rl.policies.baselines import NoReconfigPolicy
from rl.policies.networks import ActorCritic
from rl.algorithms.ppo import PPOTrainer, PPOConfig
from rl.utils.metrics import summarize_episode


@dataclass
class TrainEvalConfig:
    hidden: int = 64
    device: str = "cpu"
    iterations: int = 20
    episodes_per_iter: int = 10
    eval_episodes: int = 20
    phase_detection_eval_episodes: int = 20


@dataclass
class StrategyStats:
    mean_total_cost: float
    std_total_cost: float
    mean_breakdown: Dict[str, float]
    std_breakdown: Dict[str, float]
    mean_cost_by_t: List[float]
    std_cost_by_t: List[float]
    mean_a_M2_by_t: List[float]
    mean_age_M2_by_t: List[float]
    mean_rho_M2_by_t: List[float]


@dataclass(frozen=True)
class TariffLevelConfig:
    original_side_low_regime: int = 1
    original_side_high_regime: int = 2
    new_side_low_regime: int = 1
    new_side_high_regime: int = 3


@dataclass(frozen=True)
class PhaseStructure:
    phase1_periods: Tuple[int, int]
    phase2_periods: Tuple[int, int]
    phase3_periods: Tuple[int, int]
    phase2_start: int
    phase3_start: int
    change_period_1: int
    change_period_2: int
    activation_period_B: int
    ramp_full_period_B: int


@dataclass(frozen=True)
class TariffPathSpec:
    path_id: str
    phase1_state: str
    phase2_state: str
    phase3_state: str
    phase1_periods: Tuple[int, int]
    phase2_periods: Tuple[int, int]
    phase3_periods: Tuple[int, int]
    change_period_1: int
    change_period_2: int
    xi_path: List[int]

    @property
    def path_label(self) -> str:
        return f"{self.phase1_state} -> {self.phase2_state} -> {self.phase3_state}"


TraceInfoList = List[Dict[str, Any]]


JOINT_STATE_TO_ID: Dict[str, int] = {
    "HH": 1,
    "HL": 2,
    "LH": 3,
    "LL": 4,
}

ID_TO_JOINT_STATE: Dict[int, str] = {v: k for k, v in JOINT_STATE_TO_ID.items()}
STATE_ORDER: List[str] = ["HH", "HL", "LH", "LL"]


def joint_state_full_label(state: str) -> str:
    mapping = {
        "HH": "Original high | New high",
        "HL": "Original high | New low",
        "LH": "Original low | New high",
        "LL": "Original low | New low",
    }
    return mapping[state]


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) <= 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(v)


def _avg_breakdown(summaries: List[Dict[str, float]]) -> Dict[str, float]:
    if not summaries:
        return {}
    keys = list(summaries[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = sum(d.get(k, 0.0) for d in summaries) / len(summaries)
    return out


def _std_breakdown(summaries: List[Dict[str, float]], mean: Dict[str, float]) -> Dict[str, float]:
    if not summaries:
        return {}
    out: Dict[str, float] = {}
    for k, m in mean.items():
        if len(summaries) <= 1:
            out[k] = 0.0
        else:
            v = sum((d.get(k, 0.0) - m) ** 2 for d in summaries) / (len(summaries) - 1)
            out[k] = math.sqrt(v)
    return out


def _mean_std_series(series_list: List[List[float]]) -> Tuple[List[float], List[float]]:
    if not series_list:
        return [], []
    H = len(series_list[0])
    mean = [0.0] * H
    std = [0.0] * H
    n = len(series_list)
    for t in range(H):
        vals = [s[t] for s in series_list]
        m = sum(vals) / n
        mean[t] = m
        if n <= 1:
            std[t] = 0.0
        else:
            v = sum((x - m) ** 2 for x in vals) / (n - 1)
            std[t] = math.sqrt(v)
    return mean, std


def _train_policy(
    scenario: SupplyChainScenario,
    strategy: Strategy,
    seed: int,
    cfg: TrainEvalConfig,
) -> ActorCritic:
    env = SCReconfigEnv(scenario=scenario, strategy=strategy, seed=seed)
    policy = ActorCritic(obs_dim=env.obs_dim(), hidden=cfg.hidden)
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        config=PPOConfig(gamma=scenario.gamma),
        seed=seed,
        device=cfg.device,
    )
    for it in range(cfg.iterations):
        episode_seeds: List[int] = [seed * 100000 + it * 1000 + e for e in range(cfg.episodes_per_iter)]
        trainer.train_one_iteration(episode_seeds)
    return policy


def _load_policy_from_checkpoint_if_available(
    scenario: SupplyChainScenario,
    strategy: Strategy,
    device: str,
    ckpt_path: str,
):
    path = Path(ckpt_path)
    if not path.exists():
        return None

    ckpt = torch.load(path, map_location=device)
    hidden = int(ckpt.get("hidden", 64))
    env = SCReconfigEnv(scenario=scenario, strategy=strategy, seed=0)
    policy = ActorCritic(obs_dim=env.obs_dim(), hidden=hidden)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def _masked_binary_argmax(logit1: float, mask: Tuple[int, int]) -> int:
    logits = [0.0, float(logit1)]
    if int(mask[0]) == 0:
        logits[0] = -1.0e18
    if int(mask[1]) == 0:
        logits[1] = -1.0e18
    return int(1 if logits[1] > logits[0] else 0)


@torch.no_grad()
def _act_greedy(policy, obs_np, mask) -> Tuple[Tuple[int, int], float, float]:
    if isinstance(policy, NoReconfigPolicy):
        return (0, 0), 0.0, 0.0
    obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    u1, v1, val = policy.forward(obs)
    a_u = _masked_binary_argmax(float(u1.item()), mask.u_mask)
    a_v = _masked_binary_argmax(float(v1.item()), mask.v_mask)
    return (a_u, a_v), 0.0, float(val.item())


def _rollout_trace_episode_greedy(
    scenario: SupplyChainScenario,
    strategy: Strategy,
    policy,
    seed: int,
    episode_seed: int,
) -> TraceInfoList:
    env = SCReconfigEnv(scenario=scenario, strategy=strategy, seed=seed)
    obs = env.reset(episode_seed=int(episode_seed))
    infos: TraceInfoList = []
    done = False
    while not done:
        mask = env.get_action_mask()
        action, _, _ = _act_greedy(policy, obs, mask)
        out = env.step(action)
        infos.append(out.info)
        obs = out.obs
        done = out.done
    return infos[: int(scenario.H)]


def _eval_policy_metrics(
    scenario: SupplyChainScenario,
    strategy: Strategy,
    policy,
    seed: int,
    n_episodes: int,
) -> Tuple[float, Dict[str, float], List[float], List[float], List[float], List[float]]:
    env = SCReconfigEnv(scenario=scenario, strategy=strategy, seed=seed)
    H = int(scenario.H)
    sums = []
    cost_by_t_sum = [0.0] * H
    a2_by_t_sum = [0.0] * H
    age2_by_t_sum = [0.0] * H
    rho2_by_t_sum = [0.0] * H
    for e in range(n_episodes):
        traj = rollout_episode(env, policy, episode_seed=seed * 10000 + e)
        infos = [tr.info for tr in traj]
        sums.append(summarize_episode(infos))
        for idx, info in enumerate(infos[:H]):
            cb = info.get("cost_breakdown", {})
            cost_by_t_sum[idx] += float(cb.get("C_total", 0.0))
            a2_by_t_sum[idx] += float(info.get("a_M2", 0.0))
            age2_by_t_sum[idx] += float(info.get("age_M2", 0.0))
            rho2_by_t_sum[idx] += float(info.get("rho_M2", 0.0))
    denom = max(1, n_episodes)
    cost_by_t_mean = [x / denom for x in cost_by_t_sum]
    a2_by_t_mean = [x / denom for x in a2_by_t_sum]
    age2_by_t_mean = [x / denom for x in age2_by_t_sum]
    rho2_by_t_mean = [x / denom for x in rho2_by_t_sum]
    mean_cost = sum(s.total_cost for s in sums) / max(1, len(sums))
    bd_per_ep = [{k: v for k, v in s.cost_breakdown_sum.items()} for s in sums]
    mean_bd = _avg_breakdown(bd_per_ep)
    return mean_cost, mean_bd, cost_by_t_mean, a2_by_t_mean, age2_by_t_mean, rho2_by_t_mean


def _rollout_trace_episode(
    scenario: SupplyChainScenario,
    strategy: Strategy,
    policy,
    seed: int,
    episode_seed: int,
) -> TraceInfoList:
    env = SCReconfigEnv(scenario=scenario, strategy=strategy, seed=seed)
    traj = rollout_episode(env, policy, episode_seed=int(episode_seed))
    infos = [tr.info for tr in traj]
    return infos[: int(scenario.H)]


def evaluate_three_strategies(
    scenario: SupplyChainScenario,
    seeds: List[int],
    cfg: TrainEvalConfig,
    *,
    return_traces: bool = False,
    trace_episode_seed: int = 12345,
    trace_seed: Optional[int] = None,
):
    a_costs, a_bds = [], []
    a_ts, a_a2, a_age2, a_rho2 = [], [], [], []
    for s in seeds:
        polA = NoReconfigPolicy()
        cost, bd, ts, a2, age2, rho2 = _eval_policy_metrics(scenario, Strategy.A, polA, s, cfg.eval_episodes)
        a_costs.append(cost)
        a_bds.append(bd)
        a_ts.append(ts)
        a_a2.append(a2)
        a_age2.append(age2)
        a_rho2.append(rho2)
    a_mean, a_std = _mean_std(a_costs)
    a_bd_mean = _avg_breakdown(a_bds)
    a_bd_std = _std_breakdown(a_bds, a_bd_mean)
    a_ts_mean, a_ts_std = _mean_std_series(a_ts)
    a_a2_mean, _ = _mean_std_series(a_a2)
    a_age2_mean, _ = _mean_std_series(a_age2)
    a_rho2_mean, _ = _mean_std_series(a_rho2)

    b_costs, b_bds = [], []
    b_ts, b_a2, b_age2, b_rho2 = [], [], [], []
    trace_seed_used = int(seeds[0]) if trace_seed is None else int(trace_seed)
    polB_trace = None
    for s in seeds:
        polB = _train_policy(scenario, Strategy.B, s, cfg)
        if int(s) == trace_seed_used:
            polB_trace = polB
        cost, bd, ts, a2, age2, rho2 = _eval_policy_metrics(scenario, Strategy.B, polB, s, cfg.eval_episodes)
        b_costs.append(cost)
        b_bds.append(bd)
        b_ts.append(ts)
        b_a2.append(a2)
        b_age2.append(age2)
        b_rho2.append(rho2)
    b_mean, b_std = _mean_std(b_costs)
    b_bd_mean = _avg_breakdown(b_bds)
    b_bd_std = _std_breakdown(b_bds, b_bd_mean)
    b_ts_mean, b_ts_std = _mean_std_series(b_ts)
    b_a2_mean, _ = _mean_std_series(b_a2)
    b_age2_mean, _ = _mean_std_series(b_age2)
    b_rho2_mean, _ = _mean_std_series(b_rho2)

    c_costs, c_bds = [], []
    c_ts, c_a2, c_age2, c_rho2 = [], [], [], []
    polC_trace = None
    for s in seeds:
        polC = _train_policy(scenario, Strategy.C, s, cfg)
        if int(s) == trace_seed_used:
            polC_trace = polC
        cost, bd, ts, a2, age2, rho2 = _eval_policy_metrics(scenario, Strategy.C, polC, s, cfg.eval_episodes)
        c_costs.append(cost)
        c_bds.append(bd)
        c_ts.append(ts)
        c_a2.append(a2)
        c_age2.append(age2)
        c_rho2.append(rho2)
    c_mean, c_std = _mean_std(c_costs)
    c_bd_mean = _avg_breakdown(c_bds)
    c_bd_std = _std_breakdown(c_bds, c_bd_mean)
    c_ts_mean, c_ts_std = _mean_std_series(c_ts)
    c_a2_mean, _ = _mean_std_series(c_a2)
    c_age2_mean, _ = _mean_std_series(c_age2)
    c_rho2_mean, _ = _mean_std_series(c_rho2)

    stats: Dict[str, StrategyStats] = {
        "A": StrategyStats(a_mean, a_std, a_bd_mean, a_bd_std, a_ts_mean, a_ts_std, a_a2_mean, a_age2_mean, a_rho2_mean),
        "B": StrategyStats(b_mean, b_std, b_bd_mean, b_bd_std, b_ts_mean, b_ts_std, b_a2_mean, b_age2_mean, b_rho2_mean),
        "C": StrategyStats(c_mean, c_std, c_bd_mean, c_bd_std, c_ts_mean, c_ts_std, c_a2_mean, c_age2_mean, c_rho2_mean),
    }
    if not return_traces:
        return stats
    traces: Dict[str, TraceInfoList] = {}
    traces["A"] = _rollout_trace_episode(scenario, Strategy.A, NoReconfigPolicy(), seed=trace_seed_used, episode_seed=trace_episode_seed)
    if polB_trace is None:
        polB_trace = _train_policy(scenario, Strategy.B, int(seeds[0]), cfg)
    traces["B"] = _rollout_trace_episode(scenario, Strategy.B, polB_trace, seed=trace_seed_used, episode_seed=trace_episode_seed)
    if polC_trace is None:
        polC_trace = _train_policy(scenario, Strategy.C, int(seeds[0]), cfg)
    traces["C"] = _rollout_trace_episode(scenario, Strategy.C, polC_trace, seed=trace_seed_used, episode_seed=trace_episode_seed)
    return stats, traces


def make_scenario_base(validate: bool = False) -> SupplyChainScenario:
    return build_story_case(validate=validate)


def extract_tariff_level_values(
    base: SupplyChainScenario,
    level_cfg: TariffLevelConfig,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {
        "original_side": {"low": {}, "high": {}},
        "new_side": {"low": {}, "high": {}},
    }
    for market_id in base.markets:
        out["original_side"]["low"][market_id] = float(base.tau[base.legacy_plant_id][market_id][level_cfg.original_side_low_regime])
        out["original_side"]["high"][market_id] = float(base.tau[base.legacy_plant_id][market_id][level_cfg.original_side_high_regime])
        out["new_side"]["low"][market_id] = float(base.tau[base.candidate_plant_id][market_id][level_cfg.new_side_low_regime])
        out["new_side"]["high"][market_id] = float(base.tau[base.candidate_plant_id][market_id][level_cfg.new_side_high_regime])
    return out


def _build_joint_state_tau(
    base: SupplyChainScenario,
    level_cfg: TariffLevelConfig,
) -> Dict[str, Dict[str, Dict[int, float]]]:
    levels = extract_tariff_level_values(base, level_cfg)
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


def make_joint_tariff_scenario_base(
    *,
    validate: bool = False,
    level_cfg: Optional[TariffLevelConfig] = None,
) -> SupplyChainScenario:
    base = make_scenario_base(validate=False)
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
    scenario.tau = _build_joint_state_tau(base, level_cfg)
    scenario.xi_1 = JOINT_STATE_TO_ID["HL"]
    scenario.xi_2_forced = JOINT_STATE_TO_ID["HL"]
    if validate:
        scenario.validate_assumptions()
    return scenario


def make_fixed_hl_reference_scenario(
    *,
    validate: bool = False,
    level_cfg: Optional[TariffLevelConfig] = None,
) -> SupplyChainScenario:
    scenario = make_joint_tariff_scenario_base(validate=False, level_cfg=level_cfg)
    fixed_path = [JOINT_STATE_TO_ID["HL"]] * int(scenario.H)

    def _fixed_sample_regime_path(seed=None):
        return list(fixed_path)

    scenario.sample_regime_path = _fixed_sample_regime_path  # type: ignore[attr-defined]
    if validate:
        scenario.validate_assumptions()
    return scenario


def _first_activation_period(trace: TraceInfoList) -> Optional[int]:
    for info in trace:
        if int(info.get("u_t", 0)) == 1:
            return int(info["t"])
    return None


def _first_ramp_full_period(trace: TraceInfoList, threshold: float) -> Optional[int]:
    for info in trace:
        if int(info.get("a_M2_pre", 0)) == 1 and float(info.get("rho_M2_pre", 0.0)) >= float(threshold):
            return int(info["t"])
    return None


def _mode_or_median_int(values: List[int]) -> int:
    if not values:
        raise ValueError("empty integer list")
    counts = Counter(int(v) for v in values)
    best_count = max(counts.values())
    best_vals = sorted(v for v, c in counts.items() if c == best_count)
    if len(best_vals) == 1:
        return int(best_vals[0])
    vals_sorted = sorted(int(v) for v in values)
    return int(vals_sorted[(len(vals_sorted) - 1) // 2])


def detect_phase_structure_via_strategy_b(
    seeds: List[int],
    cfg: TrainEvalConfig,
    *,
    level_cfg: Optional[TariffLevelConfig] = None,
    trace_episode_seed_base: int = 880000,
) -> PhaseStructure:
    scenario = make_fixed_hl_reference_scenario(validate=False, level_cfg=level_cfg)
    activation_times: List[int] = []
    ramp_full_times: List[int] = []

    for seed in seeds:
        policy_b = _train_policy(scenario, Strategy.B, int(seed), cfg)
        for e in range(cfg.phase_detection_eval_episodes):
            trace = _rollout_trace_episode_greedy(
                scenario=scenario,
                strategy=Strategy.B,
                policy=policy_b,
                seed=int(seed),
                episode_seed=int(trace_episode_seed_base + 1000 * int(seed) + e),
            )
            t_open = _first_activation_period(trace)
            if t_open is not None:
                activation_times.append(int(t_open))
            t_full = _first_ramp_full_period(trace, threshold=float(scenario.ramp_full_threshold))
            if t_full is not None:
                ramp_full_times.append(int(t_full))

    if not activation_times:
        ckpt_policy = _load_policy_from_checkpoint_if_available(
            scenario=scenario,
            strategy=Strategy.B,
            device=cfg.device,
            ckpt_path="checkpoints/policy_B.pt",
        )
        if ckpt_policy is not None:
            print("[phase detection] quick training did not activate M2, trying checkpoints/policy_B.pt")
            for e in range(max(5, cfg.phase_detection_eval_episodes)):
                trace = _rollout_trace_episode_greedy(
                    scenario=scenario,
                    strategy=Strategy.B,
                    policy=ckpt_policy,
                    seed=0,
                    episode_seed=int(trace_episode_seed_base + 50000 + e),
                )
                t_open = _first_activation_period(trace)
                if t_open is not None:
                    activation_times.append(int(t_open))
                t_full = _first_ramp_full_period(trace, threshold=float(scenario.ramp_full_threshold))
                if t_full is not None:
                    ramp_full_times.append(int(t_full))

    H = int(scenario.H)

    if not activation_times:
        activation_period = 1
        ramp_full_period = min(H, activation_period + int(scenario.ramp_full_age))
        print(
            "[phase detection] Strategy B still did not activate the candidate plant. "
            f"Using fallback phase split: Phase II starts at t=1 and Phase III starts at t={ramp_full_period}."
        )
    else:
        activation_period = _mode_or_median_int(activation_times)
        if ramp_full_times:
            ramp_full_period = _mode_or_median_int(ramp_full_times)
        else:
            ramp_full_period = activation_period + int(scenario.ramp_full_age)

    phase2_start = max(1, min(H, int(activation_period)))
    phase3_start = max(phase2_start + 1, min(H, int(ramp_full_period)))

    change_period_1 = max(phase2_start, min(H, (phase2_start + phase3_start) // 2))
    change_period_2 = max(phase3_start, min(H, (phase3_start + H) // 2))

    phase1_periods = (1, phase2_start - 1)
    phase2_periods = (phase2_start, max(phase2_start, phase3_start - 1))
    phase3_periods = (phase3_start, H)

    return PhaseStructure(
        phase1_periods=phase1_periods,
        phase2_periods=phase2_periods,
        phase3_periods=phase3_periods,
        phase2_start=phase2_start,
        phase3_start=phase3_start,
        change_period_1=change_period_1,
        change_period_2=change_period_2,
        activation_period_B=activation_period,
        ramp_full_period_B=ramp_full_period,
    )


def build_tariff_path_specs_from_phase_structure(
    phase_structure: PhaseStructure,
    horizon: int,
) -> List[TariffPathSpec]:
    phase1_state = "HL"
    specs: List[TariffPathSpec] = []
    counter = 1
    cp1 = int(phase_structure.change_period_1)
    cp2 = int(phase_structure.change_period_2)
    H = int(horizon)
    for s2 in STATE_ORDER:
        for s3 in STATE_ORDER:
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
                    phase1_periods=phase_structure.phase1_periods,
                    phase2_periods=phase_structure.phase2_periods,
                    phase3_periods=phase_structure.phase3_periods,
                    change_period_1=cp1,
                    change_period_2=cp2,
                    xi_path=xi_path,
                )
            )
            counter += 1
    return specs


def make_tariff_path_scenario(
    spec: TariffPathSpec,
    *,
    validate: bool = False,
    level_cfg: Optional[TariffLevelConfig] = None,
) -> SupplyChainScenario:
    scenario = make_joint_tariff_scenario_base(validate=False, level_cfg=level_cfg)
    fixed_path = list(spec.xi_path)
    scenario.xi_1 = int(fixed_path[0])
    scenario.xi_2_forced = int(fixed_path[1]) if len(fixed_path) >= 2 else None

    def _fixed_sample_regime_path(seed=None):
        return list(fixed_path)

    scenario.sample_regime_path = _fixed_sample_regime_path  # type: ignore[attr-defined]
    if validate:
        scenario.validate_assumptions()
    return scenario