import argparse
import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.env import get_env_ids, make_env, summarize_observation, symbolic_grid_to_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the environments")
    parser.add_argument("--env", type=str, default="all", help="Environment id, or 'all'")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed")
    parser.add_argument("--no-grid", action="store_true", help="Do not print the ASCII symbolic grid")
    args = parser.parse_args()

    for env_name in get_env_ids(args.env):
        print(f"\n=== {env_name} ===")

        raw_env = gym.make(env_name)
        raw_obs, _ = raw_env.reset(seed=args.seed)
        print(f"raw observation keys: {list(raw_obs.keys())}")
        print(f"raw image shape: {raw_obs['image'].shape}")
        raw_env.close()

        env = make_env(env_name, symbolic=True)
        obs, _ = env.reset(seed=args.seed)
        summary = summarize_observation(obs)

        print(f"symbolic image shape: {summary['image_shape']}")
        print(f"action space: {env.action_space}")
        print(f"agent: {summary['agent_positions']}, direction: {summary['direction_name']}")
        print(f"obstacles: {summary['obstacle_positions']}")
        print(f"goal: {summary['goal_positions']}")
        print(f"mission: {summary['mission']!r}")

        if not args.no_grid:
            print("symbolic grid:")
            print(symbolic_grid_to_text(obs["image"]))

        env.close()
