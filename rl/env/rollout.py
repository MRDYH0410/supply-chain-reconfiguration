from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from .sc_reconfig_env import SCReconfigEnv
from .masks import ActionMask


@dataclass
class Transition:
    obs: np.ndarray
    action: Tuple[int, int]
    logp: float
    value: float
    reward: float
    done: bool
    mask_u: Tuple[int, int]
    mask_v: Tuple[int, int]
    info: Dict[str, Any]


def rollout_episode(
    env: SCReconfigEnv,
    policy,
    episode_seed: Optional[int] = None,
) -> List[Transition]:
    obs = env.reset(episode_seed=episode_seed)
    transitions: List[Transition] = []

    done = False
    while not done:
        mask: ActionMask = env.get_action_mask()
        action, logp, value = policy.act(obs, mask)
        out = env.step(action)

        transitions.append(
            Transition(
                obs=obs,
                action=action,
                logp=float(logp),
                value=float(value),
                reward=float(out.reward),
                done=bool(out.done),
                mask_u=mask.u_mask,
                mask_v=mask.v_mask,
                info=out.info,
            )
        )
        obs = out.obs
        done = out.done

    return transitions
