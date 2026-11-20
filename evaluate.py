import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO

from agents import AGENTS


def run_episode(model, env, episode, verbose=True):

    obs, _ = env.reset(seed=episode)
    log = []
    total_reward = 0.0

    if verbose:
        print("\n" + "=" * 45)
        print(f"EPISODE {episode}")
        print("=" * 45)

    for step in range(env.max_steps):

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        event = {
            "step": step + 1,
            "action": action,
            "description": info["description"],
            "reward": float(reward),
            "reward_breakdown": info["reward_breakdown"],
            "hack_event": info.get("hack_event"),
            "proxy_metrics": info.get("proxy_metrics", {}),
            "true_metrics": info.get("true_metrics", {})
        }
        log.append(event)

        if verbose:
            flag = f"  <-- HACK: {event['hack_event']}" if event["hack_event"] else ""
            print(
                f"Step {step + 1:2d} | {info['description']:<55} "
                f"| {reward:+6.2f}{flag}"
            )

        if terminated or truncated:
            break

    hacks = Counter(e["hack_event"] for e in log if e["hack_event"])

    result = {
        "episode": episode,
        "total_reward": round(total_reward, 2),
        "steps": len(log),
        "hack_events": dict(hacks),
        "hack_rate": round(sum(hacks.values()) / len(log), 2),
        "final_proxy_metrics": log[-1]["proxy_metrics"],
        "final_true_metrics": log[-1]["true_metrics"]
    }

    if verbose:
        print("\nSummary:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    return {"results": result, "activity_log": log}


def evaluate(name, episodes, verbose):

    model = PPO.load(f"models/{name}")
    env = AGENTS[name]()

    runs = [
        run_episode(model, env, ep + 1, verbose) for ep in range(episodes)
    ]
    env.close()

    return runs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", default="all", choices=["all", *AGENTS.keys()]
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    names = list(AGENTS) if args.agent == "all" else [args.agent]

    output = {
        "timestamp": datetime.now().isoformat(),
        "agents": {}
    }

    for name in names:
        print("\n" + "#" * 45)
        print(f"AGENT: {name.upper()}")
        print("#" * 45)
        output["agents"][name] = evaluate(name, args.episodes, not args.quiet)

    # Cross-agent comparison: proxy reward vs. real outcome
    print("\n" + "=" * 45)
    print("PROXY REWARD vs TRUE OUTCOME (mean over episodes)")
    print("=" * 45)
    for name, runs in output["agents"].items():
        n = len(runs)
        reward = sum(r["results"]["total_reward"] for r in runs) / n
        rate = sum(r["results"]["hack_rate"] for r in runs) / n
        true = runs[-1]["results"]["final_true_metrics"]
        print(f"{name:<22} reward={reward:8.2f}  hack_rate={rate:.2f}  true={true}")

    Path("experiments").mkdir(exist_ok=True)
    filename = (
        "experiments/experiment_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )

    with open(filename, "w") as file:
        json.dump(output, file, indent=4)

    print("\nExperiment saved:", filename)


if __name__ == "__main__":
    main()