import argparse

from minigridrl.env import ACTION_NAMES, make_env, symbolic_grid_to_text


def main():
    parser = argparse.ArgumentParser(description="Print a text trace of a random rollout")
    parser.add_argument("--env", default="MiniGrid-Dynamic-Obstacles-5x5-v0", help="Environment id")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed")
    parser.add_argument("--steps", type=int, default=10, help="Maximum number of steps")
    args = parser.parse_args()

    env = make_env(args.env, symbolic=True)
    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0

    print("initial grid:")
    print(symbolic_grid_to_text(obs["image"]))

    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(
            f"\nstep={step} action={ACTION_NAMES.get(int(action), action)} "
            f"reward={reward:.3f} total={total_reward:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )
        print(symbolic_grid_to_text(obs["image"]))
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
