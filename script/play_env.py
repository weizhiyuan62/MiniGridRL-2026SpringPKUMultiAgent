import argparse

from minigridrl.env import ACTION_NAMES, make_env, summarize_observation, symbolic_grid_to_text


KEY_TO_ACTION = {
    "a": 0,
    "left": 0,
    "d": 1,
    "right": 1,
    "w": 2,
    "up": 2,
}


def main():
    parser = argparse.ArgumentParser(description="Manually interact with a MiniGrid environment")
    parser.add_argument("--env", default="MiniGrid-Dynamic-Obstacles-5x5-v0", help="Environment id")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed")
    parser.add_argument("--symbolic", action="store_true", help="Print symbolic grid after each step")
    args = parser.parse_args()

    env = make_env(args.env, symbolic=True, render_mode="human")
    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    done = False

    print("Controls: a/left = turn left, d/right = turn right, w/up = forward, r = reset, q = quit")
    _print_state(obs, total_reward, args.symbolic)

    while True:
        key = input("> ").strip().lower()
        if key == "q":
            break
        if key == "r":
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            _print_state(obs, total_reward, args.symbolic)
            continue
        if done:
            print("Episode ended. Press r to reset, or q to quit.")
            continue
        if key not in KEY_TO_ACTION:
            print("Unknown command. Use a, d, w, r, or q.")
            continue

        action = KEY_TO_ACTION[key]
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(
            f"action={ACTION_NAMES[action]} reward={reward:.3f} total={total_reward:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )
        _print_state(obs, total_reward, args.symbolic)

        if terminated or truncated:
            done = True
            print("Episode ended. Press r to reset, or q to quit.")

    env.close()


def _print_state(obs, total_reward, show_grid):
    summary = summarize_observation(obs)
    print(
        f"agent={summary['agent_positions']} direction={summary['direction_name']} "
        f"obstacles={summary['obstacle_positions']} goal={summary['goal_positions']} "
        f"total={total_reward:.3f}"
    )
    if show_grid:
        print(symbolic_grid_to_text(obs["image"]))


if __name__ == "__main__":
    main()
