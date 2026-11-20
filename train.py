import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agents import AGENTS


def train(name, timesteps):

    env = AGENTS[name]()
    check_env(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99
    )

    print(f"\nTraining {name} agent...")
    model.learn(total_timesteps=timesteps)

    Path("models").mkdir(exist_ok=True)
    model.save(f"models/{name}")

    print(f"Model saved as models/{name}.zip")
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", default="all", choices=["all", *AGENTS.keys()]
    )
    parser.add_argument("--timesteps", type=int, default=50000)
    args = parser.parse_args()

    names = list(AGENTS) if args.agent == "all" else [args.agent]

    for name in names:
        train(name, args.timesteps)

    print("\nTraining complete!")