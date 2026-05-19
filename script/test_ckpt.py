import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from minigridrl.env import ACTION_NAMES, make_env
from minigridrl.utils.obs import obs_cnn_shape, obs_to_cnn_input, obs_to_tabular_state


def main():
    parser = argparse.ArgumentParser(description="Test and render a trained checkpoint in MiniGrid")
    parser.add_argument("--ckpt", required=True, help="Path to a q_learning .pkl or PPO .pt checkpoint")
    parser.add_argument("--env", required=True, help="Environment id used for evaluation")
    parser.add_argument("--algo", choices=["auto", "q_learning", "ppo"], default="auto", help="Checkpoint algorithm")
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed")
    parser.add_argument("--steps", type=int, default=500, help="Maximum rollout steps")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between rendered steps in seconds")
    parser.add_argument("--device", default="cpu", help="Torch device for PPO checkpoints")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the PPO policy instead of taking the greedy action",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    checkpoint = _load_checkpoint(ckpt_path, args.device)
    algo = checkpoint.get("algo") if args.algo == "auto" else args.algo
    if algo not in {"q_learning", "ppo"}:
        raise ValueError(f"Cannot infer checkpoint algorithm from {ckpt_path}; pass --algo explicitly")

    env = make_env(args.env, symbolic=True, render_mode="human")
    try:
        policy = _build_policy(algo, checkpoint, env, args.device, args.stochastic)
        obs, _ = env.reset(seed=args.seed)
        env.render()

        total_reward = 0.0
        for step in range(1, args.steps + 1):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            env.render()

            print(
                f"step={step:03d} action={ACTION_NAMES.get(int(action), action)} "
                f"reward={reward:.3f} total={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated}"
            )

            time.sleep(args.delay)
            if terminated or truncated:
                print(f"episode ended after {step} steps, total_reward={total_reward:.3f}")
                break
        else:
            print(f"rollout stopped at step limit, total_reward={total_reward:.3f}")
    finally:
        env.close()


def _load_checkpoint(ckpt_path, device):
    if ckpt_path.suffix == ".pkl":
        with ckpt_path.open("rb") as file:
            return pickle.load(file)

    if ckpt_path.suffix == ".pt":
        import torch

        try:
            return torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location=device)

    raise ValueError(f"Unsupported checkpoint suffix {ckpt_path.suffix!r}; expected .pkl or .pt")


def _build_policy(algo, checkpoint, env, device, stochastic):
    if algo == "q_learning":
        return _build_q_learning_policy(checkpoint, env.action_space.n)
    return _build_ppo_policy(checkpoint, env, device, stochastic)


def _build_q_learning_policy(checkpoint, action_dim):
    q_table = checkpoint["q_table"]

    def policy(obs):
        state = obs_to_tabular_state(obs)
        q_values = q_table.get(state)
        if q_values is None:
            q_values = np.zeros(action_dim, dtype=np.float32)
        return int(np.argmax(q_values))

    return policy


def _build_ppo_policy(checkpoint, env, device, stochastic):
    import torch

    from minigridrl.models.ppo import ActorCritic

    config = checkpoint.get("config", {})
    hidden_dim = config.get("hidden_dim", 64)
    cnn_channels = config.get("cnn_channels", 16)
    model = ActorCritic(obs_cnn_shape(env), env.action_space.n, hidden_dim, cnn_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def policy(obs):
        image, direction = obs_to_cnn_input(obs)
        image_tensor = torch.as_tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
        direction_tensor = torch.as_tensor([direction], dtype=torch.long, device=device)
        with torch.no_grad():
            if stochastic:
                action, _, _, _ = model.get_action_and_value(image_tensor, direction_tensor)
                return int(action.item())
            logits, _ = model(image_tensor, direction_tensor)
            return int(torch.argmax(logits, dim=-1).item())

    return policy


if __name__ == "__main__":
    main()
