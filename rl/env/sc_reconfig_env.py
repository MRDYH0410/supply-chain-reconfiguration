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


class Strategy(str, Enum):
    A = "A"  # No relocation
    B = "B"  # Partial relocation
    C = "C"  # Full relocation


@dataclass
class StepOutput:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class SCReconfigEnv:
    """Environment for structural reconfiguration RL with embedded operational LP.

    This environment is designed to match Chapter 3 and 4 logic
    - state includes regime xi_t and structural state (a_M1, a_M2, age)
    - action is (u_t, v_t) with strategy specific masks
    - operational decisions are obtained by solving the LP each period
    - reward is negative total cost assembled from operational + structural components

    Strategy separation
    - Strategy A is evaluated with fixed actions (u=0, v=0) and no policy training
    - Strategy B trains policy on activation u in Phase I only
    - Strategy C trains policy on activation u in Phase I and withdrawal v in Phase II with feasibility mask
    """

    def __init__(
        self,
        scenario: SupplyChainScenario,
        strategy: Strategy,
        seed: int = 0,
        episode_seed: Optional[int] = None,
    ):
        self.scenario = scenario
        self.strategy = Strategy(strategy)
        self.base_seed = int(seed)
        self.episode_seed = episode_seed

        self._rng = np.random.default_rng(self.base_seed)
        self._regime_path: List[int] = []
        self._state: Optional[StructuralState] = None

    # ---------- observation encoding ----------
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

        # feature vector
        return np.concatenate(
            [xi_onehot, np.array([a_m1, a_m2, age_norm, t_norm], dtype=np.float32)],
            axis=0,
        )

    def obs_dim(self) -> int:
        return len(self.scenario.regimes) + 4

    # ---------- reset and step ----------
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
            return ActionMask(u_mask=(1, 0), v_mask=(1, 0))  # fixed u=0,v=0
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
        u_t, v_t = int(action[0]), int(action[1])

        # enforce strategy A and B rules defensively
        if self.strategy == Strategy.A:
            u_t, v_t = 0, 0
        if self.strategy == Strategy.B:
            v_t = 0

        t = self._state.t
        xi_t = self._state.xi_t

        # 1 solve operational LP under current structure
        op = solve_operational_lp(
            scenario=self.scenario,
            t=t,
            xi_t=xi_t,
            a=self._state.a,
            age=self._state.age,
        )

        # 2 assemble total cost and reward
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

        # 3 transition regime and structural state
        done = (t >= self.scenario.H)
        if not done:
            xi_next = int(self._regime_path[t])  # path is 0-indexed, t is 1-indexed
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

        info = {
            "t": t,
            "xi_t": xi_t,
            "u_t": u_t,
            "v_t": v_t,
            "lp_status": op.status,
            "lp_message": op.message,
            "cost_breakdown": cb.__dict__,
            "d": op.d,
            "L": op.L,
        }

        return StepOutput(obs=obs_next, reward=reward, done=done, info=info)
