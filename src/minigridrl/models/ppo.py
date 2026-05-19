from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from minigridrl.utils.logging import CSVLogger
from minigridrl.utils.obs import obs_cnn_shape, obs_to_cnn_input


@dataclass
class PPOConfig:
    total_steps: int = 50_000
    rollout_steps: int = 1024
    update_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    cnn_channels: int = 16
    seed: int = 23
    device: str = "cpu"
    log_path: str = "results/ppo.csv"
    debug_log_path: str | None = None


class ActorCritic(nn.Module):
    def __init__(self, image_shape, action_dim, hidden_dim, cnn_channels=16):
        super().__init__()
        channels, height, width = image_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        encoded_dim = cnn_channels * height * width
        self.shared = nn.Sequential(
            nn.Linear(encoded_dim + 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, image, direction):
        direction_one_hot = torch.nn.functional.one_hot(direction, num_classes=4).float()
        features = torch.cat([self.encoder(image), direction_one_hot], dim=-1)
        hidden = self.shared(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def get_action_and_value(self, image, direction, action=None):
        logits, value = self(image, direction)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def train_ppo(env, config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    model = ActorCritic(
        obs_cnn_shape(env),
        env.action_space.n,
        config.hidden_dim,
        config.cnn_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    obs, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_length = 0
    episode = 0
    global_step = 0

    fields = [
        "step",
        "episode",
        "episode_return",
        "episode_length",
        "success",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clipfrac",
        "explained_variance",
    ]
    debug_fields = [
        "iteration",
        "step",
        "episode",
        "rollout_steps",
        "rollout_episodes",
        "rollout_successes",
        "rollout_timeouts",
        "rollout_collisions",
        "action_left_frac",
        "action_right_frac",
        "action_forward_frac",
        "reward_mean",
        "reward_sum",
        "advantage_mean",
        "advantage_std",
        "return_mean",
        "value_mean",
        "next_value_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clipfrac",
        "explained_variance",
    ]
    with CSVLogger(config.log_path, fields) as logger:
        debug_logger = CSVLogger(config.debug_log_path, debug_fields) if config.debug_log_path else None
        try:
            iteration = 0
            while global_step < config.total_steps:
                rollout_steps = min(config.rollout_steps, config.total_steps - global_step)
                progress_state = {
                    "episode": episode,
                    "episode_return": episode_return,
                    "last_return": episode_return,
                }
                with tqdm(total=rollout_steps, desc=f"Iteration {iteration}", dynamic_ncols=True) as progress:
                    rollout = _collect_rollout(
                        env,
                        model,
                        obs,
                        config,
                        device,
                        global_step,
                        progress=progress,
                        progress_state=progress_state,
                    )
                obs = rollout.pop("last_obs")
                global_step += len(rollout["rewards"])

                for reward, done in zip(rollout["rewards"], rollout["dones"]):
                    episode_return += float(reward)
                    episode_length += 1
                    if done:
                        episode += 1
                        logger.log(
                            {
                                "step": min(global_step, config.total_steps),
                                "episode": episode,
                                "episode_return": episode_return,
                                "episode_length": episode_length,
                                "success": int(reward > 0),
                            }
                        )
                        episode_return = 0.0
                        episode_length = 0

                losses = _update(model, optimizer, rollout, config, device)
                logger.log(
                    {
                        "step": min(global_step, config.total_steps),
                        "episode": episode,
                        "policy_loss": losses["policy_loss"],
                        "value_loss": losses["value_loss"],
                        "entropy": losses["entropy"],
                        "approx_kl": losses["approx_kl"],
                        "clipfrac": losses["clipfrac"],
                        "explained_variance": losses["explained_variance"],
                    }
                )

                if debug_logger is not None:
                    debug_logger.log(
                        {
                            "iteration": iteration,
                            "step": min(global_step, config.total_steps),
                            "episode": episode,
                            **_rollout_debug_stats(rollout, losses, env.action_space.n),
                        }
                    )
                iteration += 1
        finally:
            if debug_logger is not None:
                debug_logger.close()

    return model


def _collect_rollout(env, model, obs, config, device, global_step, progress=None, progress_state=None):
    image_buf = []
    direction_buf = []
    actions = []
    logprobs = []
    rewards = []
    dones = []
    truncateds = []
    values = []
    next_values = []

    steps = min(config.rollout_steps, config.total_steps - global_step)
    if progress is not None:
        progress.set_postfix(
            {
                "episode": progress_state["episode"] if progress_state else 0,
                "return": f"{progress_state['last_return']:.3f}" if progress_state else "0.000",
            }
        )

    for _ in range(steps):
        image, direction = obs_to_cnn_input(obs)
        image_tensor = torch.as_tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
        direction_tensor = torch.as_tensor([direction], dtype=torch.long, device=device)
        with torch.no_grad():
            action, logprob, _, value = model.get_action_and_value(image_tensor, direction_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = terminated or truncated

        with torch.no_grad():
            next_image, next_direction = obs_to_cnn_input(next_obs)
            next_value = model(
                torch.as_tensor(next_image, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor([next_direction], dtype=torch.long, device=device),
            )[1].item()

        image_buf.append(image)
        direction_buf.append(direction)
        actions.append(int(action.item()))
        logprobs.append(float(logprob.item()))
        rewards.append(float(reward))
        dones.append(done)
        truncateds.append(truncated)
        values.append(float(value.item()))
        next_values.append(float(next_value))

        obs = next_obs
        if done:
            if progress_state is not None:
                progress_state["episode"] += 1
                progress_state["last_return"] = progress_state["episode_return"] + float(reward)
                progress_state["episode_return"] = 0.0
            obs, _ = env.reset()
        elif progress_state is not None:
            progress_state["episode_return"] += float(reward)

        if progress is not None:
            progress.set_postfix(
                {
                    "episode": progress_state["episode"] if progress_state else 0,
                    "return": f"{progress_state['last_return']:.3f}" if progress_state else "0.000",
                }
            )
            progress.update(1)

    advantages, returns = _compute_gae(rewards, dones, values, next_values, config)
    return {
        "images": np.asarray(image_buf, dtype=np.float32),
        "directions": np.asarray(direction_buf, dtype=np.int64),
        "actions": np.asarray(actions, dtype=np.int64),
        "logprobs": np.asarray(logprobs, dtype=np.float32),
        "rewards": rewards,
        "dones": dones,
        "truncateds": truncateds,
        "values": np.asarray(values, dtype=np.float32),
        "next_values": np.asarray(next_values, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "last_obs": obs,
    }


def _rollout_debug_stats(rollout, losses, action_dim):
    actions = np.asarray(rollout["actions"], dtype=np.int64)
    action_counts = np.bincount(actions, minlength=action_dim)
    action_fracs = action_counts / max(len(actions), 1)

    rewards = np.asarray(rollout["rewards"], dtype=np.float32)
    advantages = np.asarray(rollout["advantages"], dtype=np.float32)
    returns = np.asarray(rollout["returns"], dtype=np.float32)
    values = np.asarray(rollout["values"], dtype=np.float32)
    next_values = np.asarray(rollout["next_values"], dtype=np.float32)
    dones = np.asarray(rollout["dones"], dtype=bool)
    truncateds = np.asarray(rollout["truncateds"], dtype=bool)

    return {
        "rollout_steps": len(rewards),
        "rollout_episodes": int(np.sum(dones)),
        "rollout_successes": int(np.sum(rewards > 0)),
        "rollout_timeouts": int(np.sum(truncateds)),
        "rollout_collisions": int(np.sum(rewards < 0)),
        "action_left_frac": float(action_fracs[0]) if action_dim > 0 else 0.0,
        "action_right_frac": float(action_fracs[1]) if action_dim > 1 else 0.0,
        "action_forward_frac": float(action_fracs[2]) if action_dim > 2 else 0.0,
        "reward_mean": float(np.mean(rewards)) if len(rewards) else 0.0,
        "reward_sum": float(np.sum(rewards)),
        "advantage_mean": float(np.mean(advantages)) if len(advantages) else 0.0,
        "advantage_std": float(np.std(advantages)) if len(advantages) else 0.0,
        "return_mean": float(np.mean(returns)) if len(returns) else 0.0,
        "value_mean": float(np.mean(values)) if len(values) else 0.0,
        "next_value_mean": float(np.mean(next_values)) if len(next_values) else 0.0,
        **losses,
    }


def _compute_gae(rewards, dones, values, next_values, config):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + config.gamma * next_values[t] * next_nonterminal - values[t]
        last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def _update(model, optimizer, rollout, config, device):
    images = torch.as_tensor(rollout["images"], dtype=torch.float32, device=device)
    directions = torch.as_tensor(rollout["directions"], dtype=torch.long, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
    old_logprobs = torch.as_tensor(rollout["logprobs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    old_values = torch.as_tensor(rollout["values"], dtype=torch.float32, device=device)
    batch_size = images.shape[0]
    minibatch_size = min(config.minibatch_size, batch_size)
    indices = np.arange(batch_size)
    policy_losses = []
    value_losses = []
    entropies = []
    approx_kls = []
    clipfracs = []

    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            _, new_logprob, entropy, value = model.get_action_and_value(
                images[mb_idx],
                directions[mb_idx],
                actions[mb_idx],
            )

            logratio = new_logprob - old_logprobs[mb_idx]
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean()
            unclipped = -advantages[mb_idx] * ratio
            clipped = -advantages[mb_idx] * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
            policy_loss = torch.max(unclipped, clipped).mean()
            value_loss = 0.5 * ((returns[mb_idx] - value) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.mean().item()))
            approx_kls.append(float(approx_kl.item()))
            clipfracs.append(float(clipfrac.item()))

    explained_variance = _explained_variance(old_values.detach().cpu().numpy(), returns.detach().cpu().numpy())

    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "explained_variance": explained_variance,
    }


def _explained_variance(values, returns):
    return_var = np.var(returns)
    if return_var == 0:
        return 0.0
    return float(1.0 - np.var(returns - values) / return_var)
