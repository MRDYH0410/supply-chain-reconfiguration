from __future__ import annotations

from typing import List, Dict, Any

from scenarios.story_case import build_story_case
from rl.env.sc_reconfig_env import SCReconfigEnv, Strategy
from rl.policies.baselines import NoReconfigPolicy, ActivateAtFirstChancePolicy
from rl.env.rollout import rollout_episode
from rl.utils.metrics import summarize_episode


def evaluate_baselines(n_episodes: int = 20, seed: int = 0) -> None:
    scenario = build_story_case()

    # Strategy A baseline is no reconfig
    envA = SCReconfigEnv(scenario=scenario, strategy=Strategy.A, seed=seed)
    polA = NoReconfigPolicy()

    envB = SCReconfigEnv(scenario=scenario, strategy=Strategy.B, seed=seed)
    polB = ActivateAtFirstChancePolicy()

    envC = SCReconfigEnv(scenario=scenario, strategy=Strategy.C, seed=seed)
    polC = ActivateAtFirstChancePolicy()

    def run(env, pol, name: str):
        sums = []
        for e in range(n_episodes):
            traj = rollout_episode(env, pol, episode_seed=seed * 10000 + e)
            sums.append(summarize_episode([tr.info for tr in traj]))
        avg_cost = sum(s.total_cost for s in sums) / n_episodes
        print(f"{name}: avg_total_cost={avg_cost:.3f}")

    run(envA, polA, "Strategy A (NoReconfig)")
    run(envB, polB, "Strategy B (ActivateOnce)")
    run(envC, polC, "Strategy C (ActivateOnce)")

if __name__ == "__main__":
    evaluate_baselines(n_episodes=10, seed=0)
