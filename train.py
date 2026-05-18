import argparse
import pickle
from pathlib import Path
import shutil

from minigridrl.env import ENV_IDS, make_env


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on MiniGrid Dynamic Obstacles")
    parser.add_argument("--algo", choices=["q_learning", "ppo"], required=True, help="Algorithm to train")
    parser.add_argument("--env", choices=ENV_IDS+["all"], default=ENV_IDS[0], help="Environment id")
    parser.add_argument("--total-steps", type=int, default=50_000, help="Maximum total training steps")
    parser.add_argument("--seed", type=int, default=23, help="Random seed")
    parser.add_argument("--log-dir", default="results", help="Directory for CSV logs")
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
    parser.add_argument("--hidden-dim", type=int, default=128, help="PPO hidden layer width")
    parser.add_argument("--device", default="cpu", help="PPO torch device, e.g. cpu or mps")
    args = parser.parse_args()

    arg_env = args.env
    if arg_env == "all":
        envs = ENV_IDS
    else:
        envs = [arg_env]
    for select_env in envs:
        env = make_env(select_env, symbolic=True)
        log_dir = Path(args.log_dir)
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        short_env_name = _short_env_name(select_env)

        if args.algo == "q_learning":
            from minigridrl.models.q_learning import QLearningConfig, train_q_learning

            log_path = log_dir / f"q_learning_{short_env_name}_seed{args.seed}.csv"
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
            q_path = save_dir / f"q_learning_{short_env_name}_seed{args.seed}.pkl"
            with q_path.open("wb") as f:
                pickle.dump(
                    {
                        "algo": "q_learning",
                        "env": select_env,
                        "seed": args.seed,
                        "config": config.__dict__,
                        "q_table": dict(agent.q_table),
                    },
                    f,
                )
        else:
            # import torch at branch instead of top-level to avoid unnecessary dependency for q_learning
            import torch
            from minigridrl.models.ppo import PPOConfig, train_ppo
            
            log_path = log_dir / f"ppo_{short_env_name}_seed{args.seed}.csv"
            config = PPOConfig(
                total_steps=args.total_steps,
                rollout_steps=args.rollout_steps,
                update_epochs=args.update_epochs,
                minibatch_size=args.minibatch_size,
                learning_rate=args.lr if args.lr is not None else 3e-4,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_coef=args.clip_coef,
                hidden_dim=args.hidden_dim,
                seed=args.seed,
                device=args.device,
                log_path=str(log_path),
            )
            model = train_ppo(env, config)
            save_path = save_dir / f"ppo_{short_env_name}_seed{args.seed}.pt"
            if save_path.exists():
                shutil.rmtree(save_path)
            torch.save(
                {
                    "algo": "ppo",
                    "env": select_env,
                    "seed": args.seed,
                    "config": config.__dict__,
                    "model_state_dict": model.state_dict(),
                },
                save_path,
            )

        env.close()

        if not args.no_plot:
            from minigridrl.utils.visualize import plot_training_curves

            plot_dir = Path(args.plot_dir) if args.plot_dir is not None else log_dir
            plot_path = plot_dir / log_path.with_suffix(".png").name
            saved_plot_path = plot_training_curves(log_path, plot_path)
            print(f"Saved training curve to {saved_plot_path}")


def _short_env_name(env_id):
    return env_id.replace("MiniGrid-", "").replace("-v0", "").replace("-", "_")


if __name__ == "__main__":
    main()
