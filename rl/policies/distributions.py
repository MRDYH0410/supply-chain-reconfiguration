from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


def masked_binary_categorical_logits(logit1: torch.Tensor, mask: Tuple[int, int]) -> torch.Tensor:
    """Return 2-logit tensor for actions {0,1}, applying mask by -inf to invalid."""
    # logits for 0 and 1
    logits = torch.stack([torch.zeros_like(logit1), logit1], dim=-1)
    allow0, allow1 = int(mask[0]), int(mask[1])
    if allow0 == 0:
        logits[..., 0] = -1e9
    if allow1 == 0:
        logits[..., 1] = -1e9
    return logits


def sample_binary_action(logit1: torch.Tensor, mask: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample action in {0,1} and return (action, logprob)."""
    logits = masked_binary_categorical_logits(logit1, mask)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    a = dist.sample()
    logp = dist.log_prob(a)
    return a, logp


def logprob_binary_action(logit1: torch.Tensor, action: torch.Tensor, mask: Tuple[int, int]) -> torch.Tensor:
    logits = masked_binary_categorical_logits(logit1, mask)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    return dist.log_prob(action)
