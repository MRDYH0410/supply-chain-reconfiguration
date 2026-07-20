from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.env.masks import ActionMask
from .distributions import sample_binary_action, logprob_binary_action


class ActorCritic(nn.Module):
    """Actor critic for binary structural actions (u_t, v_t)."""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # logits for action=1 for u and v (action=0 logit is 0)
        self.logit_u1 = nn.Linear(hidden, 1)
        self.logit_v1 = nn.Linear(hidden, 1)

        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        u1 = self.logit_u1(h).squeeze(-1)
        v1 = self.logit_v1(h).squeeze(-1)
        v = self.value_head(h).squeeze(-1)
        return u1, v1, v

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, mask: ActionMask) -> Tuple[Tuple[int, int], float, float]:
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        u1, v1, val = self.forward(obs)

        a_u, logp_u = sample_binary_action(u1, mask.u_mask)
        a_v, logp_v = sample_binary_action(v1, mask.v_mask)

        action = (int(a_u.item()), int(a_v.item()))
        logp = float((logp_u + logp_v).item())
        value = float(val.item())
        return action, logp, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions_u: torch.Tensor,
        actions_v: torch.Tensor,
        masks_u: torch.Tensor,
        masks_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logp, entropy (approx), and value for PPO update."""
        u1, v1, val = self.forward(obs)

        # masks_u, masks_v are shape [B,2] int tensors
        logps = []
        ents = []
        for i in range(obs.shape[0]):
            mu = (int(masks_u[i,0].item()), int(masks_u[i,1].item()))
            mv = (int(masks_v[i,0].item()), int(masks_v[i,1].item()))
            lp_u = logprob_binary_action(u1[i], actions_u[i], mu)
            lp_v = logprob_binary_action(v1[i], actions_v[i], mv)
            logps.append(lp_u + lp_v)

            # simple entropy proxy: categorical entropy for both actions
            from .distributions import masked_binary_categorical_logits
            import torch.nn.functional as F
            logits_u = masked_binary_categorical_logits(u1[i], mu)
            logits_v = masked_binary_categorical_logits(v1[i], mv)
            pu = F.softmax(logits_u, dim=-1)
            pv = F.softmax(logits_v, dim=-1)
            ent = -(pu * (pu.clamp_min(1e-12)).log()).sum() - (pv * (pv.clamp_min(1e-12)).log()).sum()
            ents.append(ent)

        logp = torch.stack(logps, dim=0)
        entropy = torch.stack(ents, dim=0)
        return logp, entropy, val
