import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


ENV_IDS = [
    "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
    "MiniGrid-Dynamic-Obstacles-16x16-v0",
]

ALGOS = ["q_learning", "ppo"]


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple seeds and plot mean episode-return curves for Q-learning and PPO"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 23, 31, 42, 97], help="Random seeds to run")
    parser.add_argument("--env", choices=[*ENV_IDS, "all"], default="all", help="Environment id or all")
    parser.add_argument("--algo", choices=[*ALGOS, "all"], default="all", help="Algorithm or all")
    parser.add_argument("--total-steps", type=int, default=50_000, help="Default total steps for both algorithms")
    parser.add_argument("--q-total-steps", type=int, default=None, help="Override total steps for Q-learning")
    parser.add_argument("--ppo-total-steps", type=int, default=None, help="Override total steps for PPO")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="PPO entropy bonus coefficient")
    parser.add_argument("--results-dir", default="results", help="Directory containing train.py CSV outputs")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--logs-dir", default="logs", help="Directory for PPO debug logs")
    parser.add_argument("--output-dir", default="results/averaged", help="Directory for averaged plots and CSVs")
    parser.add_argument("--window", type=int, default=100, help="Episode rolling-average window before interpolation")
    parser.add_argument("--grid-step", type=int, default=1, help="Timestep spacing for seed averaging")
    parser.add_argument("--skip-train", action="store_true", help="Only aggregate existing CSV files")
    parser.add_argument("--force", action="store_true", help="Rerun training even if a seed CSV already exists")
    parser.add_argument("--no-std", action="store_true", help="Do not draw standard-deviation bands")
    args, train_extra_args = parser.parse_known_args()

    env_ids = ENV_IDS if args.env == "all" else [args.env]
    algos = ALGOS if args.algo == "all" else [args.algo]

    if not args.skip_train:
        _run_all_experiments(args, train_extra_args, env_ids, algos)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_all_envs(args, env_ids, algos, output_dir)


def _run_all_experiments(args, train_extra_args, env_ids, algos):
    for env_id in env_ids:
        short_env_name = _short_env_name(env_id)
        for algo in algos:
            total_steps = _total_steps_for_algo(args, algo)
            for seed in args.seeds:
                csv_path = _artifact_path(args.results_dir, algo, short_env_name, total_steps, seed, ".csv", args)
                if csv_path.exists() and not args.force:
                    print(f"[skip] {csv_path}")
                    continue

                cmd = [
                    sys.executable,
                    "train.py",
                    "--algo",
                    algo,
                    "--env",
                    env_id,
                    "--total-steps",
                    str(total_steps),
                    "--seed",
                    str(seed),
                    "--log-dir",
                    args.results_dir,
                    "--save-dir",
                    args.checkpoints_dir,
                    "--debug-log-dir",
                    args.logs_dir,
                    "--no-plot",
                ]
                if algo == "ppo":
                    cmd.extend(["--entropy-coef", str(args.entropy_coef)])
                cmd.extend(train_extra_args)
                print("[run]", " ".join(cmd))
                subprocess.run(cmd, check=True)


def _plot_all_envs(args, env_ids, algos, output_dir):
    plt = _import_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    for axis, env_id in zip(axes.flatten(), env_ids):
        _plot_env(axis, args, env_id, algos, output_dir)

    for axis in axes.flatten()[len(env_ids) :]:
        axis.axis("off")

    fig.suptitle("Mean Episode Return Across Seeds")
    fig.tight_layout()
    combined_path = output_dir / f"mean_episode_return_all_envs{_plot_suffix(algos, args)}.png"
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)
    print(f"Saved combined plot to {combined_path}")


def _plot_env(axis, args, env_id, algos, output_dir):
    short_env_name = _short_env_name(env_id)
    for algo in algos:
        total_steps = _total_steps_for_algo(args, algo)
        seed_curves = []
        grid = np.arange(1, total_steps + 1, args.grid_step, dtype=np.float32)

        for seed in args.seeds:
            csv_path = _artifact_path(args.results_dir, algo, short_env_name, total_steps, seed, ".csv", args)
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Missing {csv_path}. Run without --skip-train, or check total steps and seed list."
                )
            steps, returns = _read_episode_returns(csv_path)
            if len(steps) == 0:
                raise ValueError(f"No episode-return rows found in {csv_path}")
            smoothed_returns = _rolling_mean(returns, args.window)
            seed_curves.append(np.interp(grid, steps, smoothed_returns, left=smoothed_returns[0], right=smoothed_returns[-1]))

        curves = np.vstack(seed_curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        label = _algo_label(algo)
        axis.plot(grid, mean, linewidth=2.0, label=f"{label} mean")
        if not args.no_std:
            axis.fill_between(grid, mean - std, mean + std, alpha=0.18)

        avg_csv_path = output_dir / f"{_run_prefix(algo, short_env_name, args)}_mean_return.csv"
        _write_mean_csv(avg_csv_path, grid, mean, std)
        print(f"Saved averaged CSV to {avg_csv_path}")

    axis.set_title(short_env_name)
    axis.set_xlabel("Environment step")
    axis.set_ylabel(f"Episode return (rolling {args.window}, mean over {len(args.seeds)} seeds)")
    axis.grid(alpha=0.3)
    axis.legend()

    env_fig_path = output_dir / f"{short_env_name}_mean_episode_return{_plot_suffix(algos, args)}.png"
    _save_single_axis_plot(env_fig_path, args, env_id, algos)


def _save_single_axis_plot(path, args, env_id, algos):
    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(9, 5))
    short_env_name = _short_env_name(env_id)
    for algo in algos:
        total_steps = _total_steps_for_algo(args, algo)
        grid = np.arange(1, total_steps + 1, args.grid_step, dtype=np.float32)
        curves = []
        for seed in args.seeds:
            csv_path = _artifact_path(args.results_dir, algo, short_env_name, total_steps, seed, ".csv", args)
            steps, returns = _read_episode_returns(csv_path)
            smoothed_returns = _rolling_mean(returns, args.window)
            curves.append(np.interp(grid, steps, smoothed_returns, left=smoothed_returns[0], right=smoothed_returns[-1]))
        curves = np.vstack(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        axis.plot(grid, mean, linewidth=2.0, label=f"{_algo_label(algo)} mean")
        if not args.no_std:
            axis.fill_between(grid, mean - std, mean + std, alpha=0.18)

    axis.set_title(f"Mean Episode Return: {short_env_name}")
    axis.set_xlabel("Environment step")
    axis.set_ylabel(f"Episode return (rolling {args.window})")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved environment plot to {path}")


def _read_episode_returns(csv_path):
    steps = []
    returns = []
    with Path(csv_path).open(newline="") as file:
        for row in csv.DictReader(file):
            if not row.get("episode_return"):
                continue
            steps.append(float(row["step"]))
            returns.append(float(row["episode_return"]))
    return np.asarray(steps, dtype=np.float32), np.asarray(returns, dtype=np.float32)


def _rolling_mean(values, window):
    if window <= 1:
        return values
    means = np.empty_like(values, dtype=np.float32)
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += float(value)
        if index >= window:
            running_sum -= float(values[index - window])
        means[index] = running_sum / min(index + 1, window)
    return means


def _write_mean_csv(path, grid, mean, std):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "mean_episode_return", "std_episode_return"])
        writer.writeheader()
        for step, mean_value, std_value in zip(grid, mean, std):
            writer.writerow(
                {
                    "step": int(step),
                    "mean_episode_return": float(mean_value),
                    "std_episode_return": float(std_value),
                }
            )


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _artifact_path(root_dir, algo, short_env_name, total_steps, seed, suffix, args):
    return Path(root_dir) / _algo_dir_name(algo) / short_env_name / f"{_run_name(algo, total_steps, seed, args)}{suffix}"


def _total_steps_for_algo(args, algo):
    if algo == "q_learning" and args.q_total_steps is not None:
        return args.q_total_steps
    if algo == "ppo" and args.ppo_total_steps is not None:
        return args.ppo_total_steps
    return args.total_steps


def _short_env_name(env_id):
    return env_id.replace("MiniGrid-", "").replace("-v0", "").replace("-", "_")


def _algo_dir_name(algo):
    return {"q_learning": "qlearn", "ppo": "ppo"}[algo]


def _algo_label(algo):
    return {"q_learning": "Q-learning", "ppo": "PPO-Clip"}[algo]


def _run_prefix(algo, short_env_name, args):
    prefix = f"{_algo_dir_name(algo)}_{short_env_name}"
    if algo == "ppo":
        prefix += f"_entropy_{_format_float(args.entropy_coef)}"
    return prefix


def _run_name(algo, total_steps, seed, args):
    name = f"steps_{total_steps}_seed_{seed}"
    if algo == "ppo":
        name += f"_entropy_{_format_float(args.entropy_coef)}"
    return name


def _format_float(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _plot_suffix(algos, args):
    if "ppo" not in algos:
        return ""
    return f"_ppo_entropy_{_format_float(args.entropy_coef)}"


if __name__ == "__main__":
    main()
