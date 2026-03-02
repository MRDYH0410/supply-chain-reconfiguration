from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from rl.algorithms.buffers import transitions_to_batch
from rl.env.rollout import rollout_episode
from rl.utils.metrics import summarize_episode
from rl.utils.seeding import set_global_seed


@dataclass
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    train_epochs: int = 5
    minibatch_size: int = 128


class PPOTrainer:
    def __init__(self, env, policy: nn.Module, config: PPOConfig, seed: int = 0, device: str = "cpu"):
        self.env = env
        self.policy = policy.to(device)
        self.cfg = config
        self.device = device
        set_global_seed(seed)
        self.opt = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

    def train_one_iteration(self, episode_seeds: List[int]) -> Dict[str, float]:
        all_transitions: List[Any] = []
        episode_infos = []

        for s in episode_seeds:
            traj = rollout_episode(self.env, self.policy, episode_seed=s)
            all_transitions.extend(traj)
            episode_infos.append([tr.info for tr in traj])

        batch = transitions_to_batch(all_transitions, gamma=self.cfg.gamma, lam=self.cfg.gae_lambda)
        obs = batch.obs.to(self.device)
        actions_u = batch.actions_u.to(self.device)
        actions_v = batch.actions_v.to(self.device)
        old_logp = batch.old_logp.to(self.device)
        returns = batch.returns.to(self.device)
        adv = batch.adv.to(self.device)
        masks_u = batch.masks_u.to(self.device)
        masks_v = batch.masks_v.to(self.device)

        N = obs.shape[0]
        idxs = torch.randperm(N)

        for _ in range(self.cfg.train_epochs):
            for start in range(0, N, self.cfg.minibatch_size):
                mb = idxs[start : start + self.cfg.minibatch_size]
                mb_obs = obs[mb]
                mb_au = actions_u[mb]
                mb_av = actions_v[mb]
                mb_old = old_logp[mb]
                mb_ret = returns[mb]
                mb_adv = adv[mb]
                mb_mu = masks_u[mb]
                mb_mv = masks_v[mb]

                logp, ent, v = self.policy.evaluate_actions(mb_obs, mb_au, mb_av, mb_mu, mb_mv)
                ratio = torch.exp(logp - mb_old)

                clip = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                pg_loss = -(torch.min(ratio * mb_adv, clip * mb_adv)).mean()

                v_loss = ((v - mb_ret) ** 2).mean()

                ent_loss = -ent.mean()

                loss = pg_loss + self.cfg.vf_coef * v_loss + self.cfg.ent_coef * ent_loss

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt.step()

        # metrics
        ep_summaries = [summarize_episode(infos) for infos in episode_infos]
        avg_cost = sum(s.total_cost for s in ep_summaries) / max(1, len(ep_summaries))
        avg_reward = sum(s.total_reward for s in ep_summaries) / max(1, len(ep_summaries))

        return {"avg_cost": float(avg_cost), "avg_reward": float(avg_reward), "steps": float(N)}
