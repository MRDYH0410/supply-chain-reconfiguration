from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

from scenarios.scenario import SupplyChainScenario
from problem.operational_lp import solve_operational_lp
from problem.costs import compute_period_cost

from .dynamics import StructuralState, next_structural_state
from .masks import compute_action_mask, ActionMask

from problem.triggers import (
    TriggerConfig,
    trigger_score_strategy_b,
    trigger_score_strategy_c,
)


class Strategy(str, Enum):
    A = "A"
    B = "B"
    C = "C"


@dataclass
class StepOutput:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class SCReconfigEnv:
    def __init__(
        self,
        scenario: SupplyChainScenario,
        strategy: Strategy,
        seed: int = 0,
        episode_seed: Optional[int] = None,
        activation_mode: str = "rl",
        trigger_cfg: Optional[TriggerConfig] = None,
    ):
        self.scenario = scenario
        self.strategy = Strategy(strategy)
        self.base_seed = int(seed)
        self.episode_seed = episode_seed

        self.activation_mode = str(activation_mode).lower()
        if self.activation_mode not in ("rl", "trigger"):
            raise ValueError("activation_mode must be 'rl' or 'trigger'")
        self.trigger_cfg = trigger_cfg if trigger_cfg is not None else TriggerConfig(gamma=scenario.gamma)

        self._rng = np.random.default_rng(self.base_seed)
        self._regime_path: List[int] = []
        self._state: Optional[StructuralState] = None

    def _encode_obs(self, state: StructuralState) -> np.ndarray:
        regimes = self.scenario.regimes
        xi_onehot = np.zeros(len(regimes), dtype=np.float32)
        xi_onehot[regimes.index(state.xi_t)] = 1.0

        M1 = self.scenario.legacy_plant_id
        M2 = self.scenario.candidate_plant_id

        a_m1 = float(state.a.get(M1, 0))
        a_m2 = float(state.a.get(M2, 0))
        age_norm = float(state.age) / max(1.0, float(self.scenario.H))
        t_norm = float(state.t) / max(1.0, float(self.scenario.H))

        return np.concatenate(
            [xi_onehot, np.array([a_m1, a_m2, age_norm, t_norm], dtype=np.float32)],
            axis=0,
        )

    def obs_dim(self) -> int:
        return len(self.scenario.regimes) + 4

    def reset(self, episode_seed: Optional[int] = None) -> np.ndarray:
        if episode_seed is None:
            episode_seed = self.episode_seed
        if episode_seed is None:
            episode_seed = int(self._rng.integers(0, 10**9))

        self._regime_path = self.scenario.sample_regime_path(seed=int(episode_seed))
        xi_1 = self._regime_path[0]

        a0 = dict(self.scenario.initial_a)
        age0 = int(self.scenario.initial_age)

        self._state = StructuralState(
            t=1,
            xi_t=int(xi_1),
            a=a0,
            age=age0,
            u_prev=0,
        )
        return self._encode_obs(self._state)

    def get_action_mask(self) -> ActionMask:
        assert self._state is not None
        if self.strategy == Strategy.A:
            return ActionMask(u_mask=(1, 0), v_mask=(1, 0))
        if self.strategy == Strategy.B:
            return compute_action_mask(
                scenario=self.scenario,
                strategy="B",
                t=self._state.t,
                xi_t=self._state.xi_t,
                a=self._state.a,
                age=self._state.age,
            )
        return compute_action_mask(
            scenario=self.scenario,
            strategy="C",
            t=self._state.t,
            xi_t=self._state.xi_t,
            a=self._state.a,
            age=self._state.age,
        )

    def step(self, action: Tuple[int, int]) -> StepOutput:
        assert self._state is not None

        # --- snapshot PRE-action structural state (for per-period timeline prints) ---
        t = self._state.t
        xi_t = self._state.xi_t
        u_prev = int(self._state.u_prev)
        a_pre = dict(self._state.a)
        age_pre = int(self._state.age)

        u_t, v_t = int(action[0]), int(action[1])

        if self.strategy == Strategy.A:
            u_t, v_t = 0, 0
        if self.strategy == Strategy.B:
            v_t = 0

        M2 = self.scenario.candidate_plant_id

        trigger_info = None
        if self.activation_mode == "trigger" and int(self._state.a.get(M2, 0)) == 0:
            cfg = self.trigger_cfg
            if self.strategy == Strategy.B:
                res = trigger_score_strategy_b(
                    scenario=self.scenario,
                    t=t,
                    xi_t=xi_t,
                    a=self._state.a,
                    age=self._state.age,
                    u_prev=self._state.u_prev,
                    cfg=cfg,
                )
            elif self.strategy == Strategy.C:
                res = trigger_score_strategy_c(
                    scenario=self.scenario,
                    t=t,
                    xi_t=xi_t,
                    a=self._state.a,
                    age=self._state.age,
                    u_prev=self._state.u_prev,
                    cfg=cfg,
                )
            else:
                res = None

            if res is not None:
                u_t = 1 if res.psi >= 0 else 0
                trigger_info = {
                    "J_NR": float(res.J_NR),
                    "J_ALT": float(res.J_ALT),
                    "psi": float(res.psi),
                    "best_theta": res.best_theta,
                }

        op = solve_operational_lp(
            scenario=self.scenario,
            t=t,
            xi_t=xi_t,
            a=self._state.a,
            age=self._state.age,
        )

        cb = compute_period_cost(
            scenario=self.scenario,
            t=t,
            xi_t=xi_t,
            a=self._state.a,
            age=self._state.age,
            u_prev=self._state.u_prev,
            v_t=v_t,
            op=op,
        )
        reward = float(cb.reward)

        done = (t >= self.scenario.H)
        if not done:
            xi_next = int(self._regime_path[t])
            self._state = next_structural_state(
                scenario=self.scenario,
                state=self._state,
                u_t=u_t,
                v_t=v_t,
                xi_next=xi_next,
            )
            obs_next = self._encode_obs(self._state)
        else:
            obs_next = self._encode_obs(self._state)

        # --- snapshot POST-transition structural state ---
        a_post = dict(self._state.a)
        age_post = int(self._state.age)

        info = {
            "t": t,
            "xi_t": xi_t,
            "u_prev": u_prev,
            "u_t": u_t,
            "v_t": v_t,

            # PRE and POST structural snapshots (clean timeline interpretation)
            "a_M1_pre": int(a_pre.get(self.scenario.legacy_plant_id, 0)),
            "a_M2_pre": int(a_pre.get(self.scenario.candidate_plant_id, 0)),
            "age_M2_pre": age_pre,
            "rho_M2_pre": float(self.scenario.ramp_factor(self.scenario.candidate_plant_id, age_pre)),
            "a_M1_post": int(a_post.get(self.scenario.legacy_plant_id, 0)),
            "a_M2_post": int(a_post.get(self.scenario.candidate_plant_id, 0)),
            "age_M2_post": age_post,
            "rho_M2_post": float(self.scenario.ramp_factor(self.scenario.candidate_plant_id, age_post)),

            # Backward-compatible aliases (used by older metrics code)
            "a_M2": int(a_pre.get(self.scenario.candidate_plant_id, 0)),
            "age_M2": age_pre,
            "rho_M2": float(self.scenario.ramp_factor(self.scenario.candidate_plant_id, age_pre)),

            "lp_status": op.status,
            "lp_message": op.message,
            "cost_breakdown": cb.__dict__,
            "d": op.d,
            "L": op.L,
            "trigger": trigger_info,
        }

        return StepOutput(obs=obs_next, reward=reward, done=done, info=info)