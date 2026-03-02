from pathlib import Path
from typing import List, Dict

from scenarios.story_case import build_story_case

from rl.env.sc_reconfig_env import SCReconfigEnv, Strategy
from rl.env.rollout import rollout_episode
from rl.policies.baselines import NoReconfigPolicy, ActivateAtFirstChancePolicy
from rl.policies.networks import ActorCritic
from rl.algorithms.ppo import PPOTrainer, PPOConfig
from rl.utils.metrics import summarize_episode, EpisodeSummary
from rl.utils.logging import CSVLogger

try:
    import torch
except Exception:
    torch = None


# =========================
# 你只需要改这里的参数
# =========================
SEED = 0

BASELINE_EPISODES = 10

TRAIN_DEVICE = "cpu"
HIDDEN = 64

TRAIN_ITERATIONS_B = 20
TRAIN_EPISODES_PER_ITER_B = 10

TRAIN_ITERATIONS_C = 20
TRAIN_EPISODES_PER_ITER_C = 10

EVAL_TRAINED_EPISODES = 20

CKPT_B = Path("checkpoints/policy_B.pt")
CKPT_C = Path("checkpoints/policy_C.pt")


def validate_scenario() -> None:
    sc = build_story_case()
    sc.print_diagnose()
    sc.validate_assumptions()
    print("\nok: scenario validated\n")


def _avg_episode_summaries(summaries: List[EpisodeSummary]) -> Dict[str, float]:
    n = max(1, len(summaries))
    avg_total_cost = sum(s.total_cost for s in summaries) / n
    avg_total_reward = sum(s.total_reward for s in summaries) / n

    keys = list(summaries[0].cost_breakdown_sum.keys()) if summaries else []
    avg_breakdown = {k: sum(s.cost_breakdown_sum[k] for s in summaries) / n for k in keys}

    out = {"avg_total_cost": float(avg_total_cost), "avg_total_reward": float(avg_total_reward)}
    for k, v in avg_breakdown.items():
        out[f"avg_{k}"] = float(v)
    return out


def run_baselines(n_episodes: int, seed: int) -> None:
    scenario = build_story_case()

    envA = SCReconfigEnv(scenario=scenario, strategy=Strategy.A, seed=seed)
    polA = NoReconfigPolicy()

    envB = SCReconfigEnv(scenario=scenario, strategy=Strategy.B, seed=seed)
    polB = ActivateAtFirstChancePolicy()

    envC = SCReconfigEnv(scenario=scenario, strategy=Strategy.C, seed=seed)
    polC = ActivateAtFirstChancePolicy()

    def run(env, pol, name: str) -> None:
        sums = []
        for e in range(n_episodes):
            traj = rollout_episode(env, pol, episode_seed=seed * 10000 + e)
            sums.append(summarize_episode([tr.info for tr in traj]))
        avg = _avg_episode_summaries(sums)

        print(f"{name}")
        print(f"  avg_total_cost  {avg['avg_total_cost']:.3f}")
        print(f"  avg_total_reward {avg['avg_total_reward']:.3f}")
        print(
            f"  avg_C_in {avg.get('avg_C_in',0):.3f}  avg_C_out {avg.get('avg_C_out',0):.3f}  "
            f"avg_C_fix {avg.get('avg_C_fix',0):.3f}  avg_C_qual {avg.get('avg_C_qual',0):.3f}  "
            f"avg_C_loss {avg.get('avg_C_loss',0):.3f}  avg_Salvage {avg.get('avg_Salvage',0):.3f}"
        )

    print("=== Baseline evaluation (no training) ===")
    run(envA, polA, "Strategy A  No reconfiguration")
    run(envB, polB, "Strategy B  Partial reconfiguration (activate once baseline)")
    run(envC, polC, "Strategy C  Full reconfiguration (activate once baseline)")
    print("")


def train_strategy(strategy: str, seed: int, iterations: int, episodes_per_iter: int, hidden: int, device: str, save: Path) -> None:
    if torch is None:
        print("torch not available -> skip training")
        return

    scenario = build_story_case()

    if strategy.upper() == "B":
        env = SCReconfigEnv(scenario=scenario, strategy=Strategy.B, seed=seed)
    elif strategy.upper() == "C":
        env = SCReconfigEnv(scenario=scenario, strategy=Strategy.C, seed=seed)
    else:
        raise ValueError("strategy must be B or C")

    policy = ActorCritic(obs_dim=env.obs_dim(), hidden=hidden)
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        config=PPOConfig(gamma=scenario.gamma),
        seed=seed,
        device=device,
    )

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    save.parent.mkdir(parents=True, exist_ok=True)

    log_path = runs_dir / f"train_{strategy.lower()}_seed{seed}.csv"
    logger = CSVLogger(path=log_path, fieldnames=["iter", "avg_cost", "avg_reward", "steps"])

    print(f"=== Training Strategy {strategy.upper()} ===")
    for it in range(iterations):
        episode_seeds: List[int] = [seed * 100000 + it * 1000 + e for e in range(episodes_per_iter)]
        stats = trainer.train_one_iteration(episode_seeds)
        stats["iter"] = it
        logger.log(stats)
        print(
            f"iter {it:03d}  avg_cost {stats['avg_cost']:.3f}  "
            f"avg_reward {stats['avg_reward']:.3f}  steps {int(stats['steps'])}"
        )

    logger.close()

    torch.save(
        {
            "strategy": strategy.upper(),
            "seed": seed,
            "obs_dim": env.obs_dim(),
            "hidden": hidden,
            "state_dict": policy.state_dict(),
        },
        save,
    )
    print(f"saved model to {save}\n")


def eval_trained(strategy: str, seed: int, n_episodes: int, device: str, load: Path) -> None:
    if torch is None:
        print("torch not available -> skip trained evaluation")
        return

    scenario = build_story_case()

    if strategy.upper() == "B":
        env = SCReconfigEnv(scenario=scenario, strategy=Strategy.B, seed=seed)
    elif strategy.upper() == "C":
        env = SCReconfigEnv(scenario=scenario, strategy=Strategy.C, seed=seed)
    else:
        raise ValueError("strategy must be B or C")

    ckpt = torch.load(load, map_location=device)
    hidden = int(ckpt.get("hidden", 64))
    policy = ActorCritic(obs_dim=env.obs_dim(), hidden=hidden)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    sums = []
    for e in range(n_episodes):
        traj = rollout_episode(env, policy, episode_seed=seed * 10000 + e)
        sums.append(summarize_episode([tr.info for tr in traj]))

    avg = _avg_episode_summaries(sums)

    print(f"=== Trained policy evaluation Strategy {strategy.upper()} ===")
    print(f"  avg_total_cost  {avg['avg_total_cost']:.3f}")
    print(f"  avg_total_reward {avg['avg_total_reward']:.3f}")
    print(
        f"  avg_C_in {avg.get('avg_C_in',0):.3f}  avg_C_out {avg.get('avg_C_out',0):.3f}  "
        f"avg_C_fix {avg.get('avg_C_fix',0):.3f}  avg_C_qual {avg.get('avg_C_qual',0):.3f}  "
        f"avg_C_loss {avg.get('avg_C_loss',0):.3f}  avg_Salvage {avg.get('avg_Salvage',0):.3f}"
    )
    print("")


def main() -> None:
    # 1) scenario self-check
    validate_scenario()

    # 2) run three strategies baseline (no training)
    run_baselines(n_episodes=BASELINE_EPISODES, seed=SEED)

    # 3) train Strategy B and C separately
    train_strategy(
        strategy="B",
        seed=SEED,
        iterations=TRAIN_ITERATIONS_B,
        episodes_per_iter=TRAIN_EPISODES_PER_ITER_B,
        hidden=HIDDEN,
        device=TRAIN_DEVICE,
        save=CKPT_B,
    )
    train_strategy(
        strategy="C",
        seed=SEED,
        iterations=TRAIN_ITERATIONS_C,
        episodes_per_iter=TRAIN_EPISODES_PER_ITER_C,
        hidden=HIDDEN,
        device=TRAIN_DEVICE,
        save=CKPT_C,
    )

    # 4) evaluate trained policies
    eval_trained(strategy="B", seed=SEED, n_episodes=EVAL_TRAINED_EPISODES, device=TRAIN_DEVICE, load=CKPT_B)
    eval_trained(strategy="C", seed=SEED, n_episodes=EVAL_TRAINED_EPISODES, device=TRAIN_DEVICE, load=CKPT_C)


if __name__ == "__main__":
    main()