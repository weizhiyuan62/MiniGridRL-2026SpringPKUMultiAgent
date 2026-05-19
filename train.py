import argparse
import pickle
from pathlib import Path

from minigridrl.env import ENV_IDS, make_env


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on MiniGrid Dynamic Obstacles")
    parser.add_argument("--algo", choices=["q_learning", "ppo"], required=True, help="Algorithm to train")
    parser.add_argument("--env", choices=ENV_IDS+["all"], default=ENV_IDS[0], help="Environment id")
    parser.add_argument("--total-steps", type=int, default=50_000, help="Maximum total training steps")
    parser.add_argument("--seed", type=int, default=23, help="Random seed")
    parser.add_argument("--log-dir", default="results", help="Directory for CSV logs")
    parser.add_argument("--debug-log-dir", default="logs", help="Directory for PPO debug CSV logs")
    parser.add_argument("--save-dir", default="checkpoints", help="Directory for saved models")
    parser.add_argument("--plot-dir", default=None, help="Directory for curve PNGs. Defaults to log-dir")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting training curves after training")

    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Q-learning initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Q-learning final epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, default=40_000, help="Q-learning epsilon decay steps")

    parser.add_argument("--rollout-steps", type=int, default=1024, help="PPO rollout length")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO optimization epochs per rollout")
    parser.add_argument("--minibatch-size", type=int, default=256, help="PPO minibatch size")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping coefficient")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="PPO GAE lambda")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="PPO entropy bonus coefficient")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PPO hidden layer width")
    parser.add_argument("--cnn-channels", type=int, default=16, help="PPO CNN channel width")
    parser.add_argument("--device", default="cpu", help="PPO torch device, e.g. cpu or mps")
    args = parser.parse_args()

    for select_env in _selected_envs(args.env):
        _train_one_env(select_env, args)


def _selected_envs(env_arg):
    if env_arg == "all":
        return ENV_IDS
    return [env_arg]


def _train_one_env(env_id, args):
    env = make_env(env_id, symbolic=True)
    try:
        short_env_name = _short_env_name(env_id)

        trainer = {
            "q_learning": _run_q_learning,
            "ppo": _run_ppo,
        }[args.algo]
        log_path = trainer(env, env_id, short_env_name, args)
    finally:
        env.close()

    if not args.no_plot:
        _plot_curves(log_path, args)


def _run_q_learning(env, env_id, short_env_name, args):
    from minigridrl.models.q_learning import QLearningConfig, train_q_learning

    log_path = _artifact_path(args.log_dir, args.algo, short_env_name, args, ".csv")
    config = QLearningConfig(
        total_steps=args.total_steps,
        learning_rate=args.lr if args.lr is not None else 0.2,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        seed=args.seed,
        log_path=str(log_path),
    )
    agent = train_q_learning(env, config)

    save_path = _artifact_path(args.save_dir, args.algo, short_env_name, args, ".pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as file:
        pickle.dump(
            {
                "algo": "q_learning",
                "env": env_id,
                "seed": args.seed,
                "config": config.__dict__,
                "q_table": dict(agent.q_table),
            },
            file,
        )
    return log_path


def _run_ppo(env, env_id, short_env_name, args):
    # Keep torch local so tabular Q-learning can run without importing it.
    import torch

    from minigridrl.models.ppo import PPOConfig, train_ppo

    log_path = _artifact_path(args.log_dir, args.algo, short_env_name, args, ".csv")
    config = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.lr if args.lr is not None else 3e-4,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        hidden_dim=args.hidden_dim,
        cnn_channels=args.cnn_channels,
        seed=args.seed,
        device=args.device,
        log_path=str(log_path),
        debug_log_path=str(_artifact_path(args.debug_log_dir, args.algo, short_env_name, args, ".debug.csv")),
    )
    model = train_ppo(env, config)

    save_path = _artifact_path(args.save_dir, args.algo, short_env_name, args, ".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algo": "ppo",
            "model_type": "cnn_actor_critic",
            "env": env_id,
            "seed": args.seed,
            "config": config.__dict__,
            "model_state_dict": model.state_dict(),
        },
        save_path,
    )
    return log_path


def _plot_curves(log_path, args):
    from minigridrl.utils.visualize import plot_training_curves

    if args.plot_dir is None:
        plot_path = log_path.with_suffix(".png")
    else:
        plot_path = _artifact_path(args.plot_dir, args.algo, log_path.parent.name, args, ".png")
    saved_plot_path = plot_training_curves(log_path, plot_path)
    print(f"Saved training curve to {saved_plot_path}")


def _artifact_path(root_dir, algo, short_env_name, args, suffix):
    return Path(root_dir) / _algo_dir_name(algo) / short_env_name / f"{_run_name(args)}{suffix}"


def _algo_dir_name(algo):
    return {"q_learning": "qlearn", "ppo": "ppo"}[algo]


def _run_name(args):
    name = f"steps_{args.total_steps}_seed_{args.seed}"
    if args.algo == "ppo":
        name += f"_entropy_{_format_float(args.entropy_coef)}"
    return name


def _format_float(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _short_env_name(env_id):
    return env_id.replace("MiniGrid-", "").replace("-v0", "").replace("-", "_")


if __name__ == "__main__":
    main()
