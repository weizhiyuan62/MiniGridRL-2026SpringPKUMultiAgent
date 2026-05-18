import argparse
import time

from minigridrl.env import ACTION_NAMES, make_env


def main():
    parser = argparse.ArgumentParser(description="Render a random policy in a MiniGrid environment")
    parser.add_argument("--env", default="MiniGrid-Dynamic-Obstacles-5x5-v0", help="Environment id")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed")
    parser.add_argument("--steps", type=int, default=100, help="Maximum number of steps")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between steps in seconds")
    parser.add_argument(
        "--symbolic",
        action="store_true",
        help="Use SymbolicObsWrapper while rendering. Rendering still shows the underlying environment.",
    )
    args = parser.parse_args()

    env = make_env(args.env, symbolic=args.symbolic, render_mode="human")
    env.reset(seed=args.seed)

    total_reward = 0.0
    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        print(
            f"step={step:03d} action={ACTION_NAMES.get(int(action), action)} "
            f"reward={reward:.3f} total={total_reward:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )

        time.sleep(args.delay)
        if terminated or truncated:
            print("episode ended")
            break

    env.close()


if __name__ == "__main__":
    main()
