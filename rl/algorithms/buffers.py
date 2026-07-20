from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor
    actions_u: torch.Tensor
    actions_v: torch.Tensor
    old_logp: torch.Tensor
    returns: torch.Tensor
    adv: torch.Tensor
    masks_u: torch.Tensor
    masks_v: torch.Tensor


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        next_value = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def transitions_to_batch(transitions: List[Any], gamma: float, lam: float) -> Batch:
    obs = np.stack([tr.obs for tr in transitions], axis=0).astype(np.float32)
    actions_u = np.array([tr.action[0] for tr in transitions], dtype=np.int64)
    actions_v = np.array([tr.action[1] for tr in transitions], dtype=np.int64)
    old_logp = np.array([tr.logp for tr in transitions], dtype=np.float32)
    rewards = np.array([tr.reward for tr in transitions], dtype=np.float32)
    dones = np.array([tr.done for tr in transitions], dtype=np.float32)
    values = np.array([tr.value for tr in transitions], dtype=np.float32)

    adv, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    masks_u = np.stack([np.array(tr.mask_u, dtype=np.int64) for tr in transitions], axis=0)
    masks_v = np.stack([np.array(tr.mask_v, dtype=np.int64) for tr in transitions], axis=0)

    return Batch(
        obs=torch.tensor(obs),
        actions_u=torch.tensor(actions_u),
        actions_v=torch.tensor(actions_v),
        old_logp=torch.tensor(old_logp),
        returns=torch.tensor(returns, dtype=torch.float32),
        adv=torch.tensor(adv, dtype=torch.float32),
        masks_u=torch.tensor(masks_u),
        masks_v=torch.tensor(masks_v),
    )
