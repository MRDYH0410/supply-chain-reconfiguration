from __future__ import annotations

import os
from typing import List

from scenarios.story_case import build_story_case
from rl.env.sc_reconfig_env import SCReconfigEnv, Strategy
from rl.policies.networks import ActorCritic
from rl.algorithms.ppo import PPOTrainer, PPOConfig
from rl.experiments.config import TrainConfig
from rl.utils.logging import CSVLogger


def train(cfg: TrainConfig) -> None:
    scenario = build_story_case()

    if cfg.strategy.upper() == "B":
        env = SCReconfigEnv(scenario=scenario, strategy=Strategy.B, seed=cfg.seed)
    elif cfg.strategy.upper() == "C":
        env = SCReconfigEnv(scenario=scenario, strategy=Strategy.C, seed=cfg.seed)
    else:
        raise ValueError("strategy must be 'B' or 'C'")

    policy = ActorCritic(obs_dim=env.obs_dim(), hidden=cfg.hidden)
    trainer = PPOTrainer(env=env, policy=policy, config=PPOConfig(gamma=scenario.gamma), seed=cfg.seed, device=cfg.device)

    log = CSVLogger(
        path=os.path.join("runs", f"train_{cfg.strategy.lower()}_seed{cfg.seed}.csv"),
        fieldnames=["iter", "avg_cost", "avg_reward", "steps"],
    )

    for it in range(cfg.iterations):
        seeds: List[int] = [cfg.seed * 100000 + it * 1000 + e for e in range(cfg.episodes_per_iter)]
        stats = trainer.train_one_iteration(seeds)
        stats["iter"] = it
        log.log(stats)
        print(f"iter {it}  avg_cost {stats['avg_cost']:.3f}  avg_reward {stats['avg_reward']:.3f}  steps {int(stats['steps'])}")

    log.close()


if __name__ == "__main__":
    cfg = TrainConfig(strategy="B", seed=0, iterations=20, episodes_per_iter=10)
    train(cfg)
