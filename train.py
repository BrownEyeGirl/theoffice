import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agents import AGENTS


def train(name, timesteps, seed, save_path):

    env = AGENTS[name]()
    check_env(env)

    # Seed the environment's own random events (demand, drift, etc.)
    env.reset(seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99,
        seed=seed
    )

    print(f"\nTraining {name} agent (seed {seed})...")
    model.learn(total_timesteps=timesteps)

    model.save(save_path)
    print(f"Model saved as {save_path}.zip")

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", default="all", choices=["all", *AGENTS.keys()]
    )
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="One or more training seeds, e.g. --seeds 0 1 2 3 4"
    )
    args = parser.parse_args()

    names = list(AGENTS) if args.agent == "all" else [args.agent]

    Path("models").mkdir(exist_ok=True)

    for name in names:
        for seed in args.seeds:

            # A single seed saves to models/<agent>.zip so evaluate.py still
            # works unchanged. Several seeds save one file per seed.
            if len(args.seeds) == 1:
                save_path = f"models/{name}"
            else:
                save_path = f"models/{name}_seed{seed}"

            train(name, args.timesteps, seed, save_path)

    print("\nTraining complete!")