from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from minigridrl.utils.logging import CSVLogger
from minigridrl.utils.obs import obs_to_vector, obs_vector_dim


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
    hidden_dim: int = 128
    seed: int = 23
    device: str = "cpu"
    log_path: str = "results/ppo.csv"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        hidden = self.shared(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def train_ppo(env, config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    model = ActorCritic(obs_vector_dim(env), env.action_space.n, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    obs, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_length = 0
    episode = 0
    global_step = 0

    fields = ["step", "episode", "episode_return", "episode_length", "success", "policy_loss", "value_loss"]
    with CSVLogger(config.log_path, fields) as logger:
        progress = tqdm(total=config.total_steps, desc="PPO")
        while global_step < config.total_steps:
            rollout = _collect_rollout(env, model, obs, config, device, global_step)
            obs = rollout.pop("last_obs")
            global_step += len(rollout["rewards"])
            progress.update(len(rollout["rewards"]))

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
                }
            )

        progress.close()

    return model


def _collect_rollout(env, model, obs, config, device, global_step):
    obs_buf = []
    actions = []
    logprobs = []
    rewards = []
    dones = []
    values = []

    steps = min(config.rollout_steps, config.total_steps - global_step)
    for _ in range(steps):
        obs_vec = torch.as_tensor(obs_to_vector(obs), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logprob, _, value = model.get_action_and_value(obs_vec)

        next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = terminated or truncated

        obs_buf.append(obs_to_vector(obs))
        actions.append(int(action.item()))
        logprobs.append(float(logprob.item()))
        rewards.append(float(reward))
        dones.append(done)
        values.append(float(value.item()))

        obs = next_obs
        if done:
            obs, _ = env.reset()

    with torch.no_grad():
        last_value = model(
            torch.as_tensor(obs_to_vector(obs), dtype=torch.float32, device=device).unsqueeze(0)
        )[1].item()

    advantages, returns = _compute_gae(rewards, dones, values, last_value, config)
    return {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "logprobs": np.asarray(logprobs, dtype=np.float32),
        "rewards": rewards,
        "dones": dones,
        "values": np.asarray(values, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "last_obs": obs,
    }


def _compute_gae(rewards, dones, values, last_value, config):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - float(dones[t])
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + config.gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def _update(model, optimizer, rollout, config, device):
    obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
    old_logprobs = torch.as_tensor(rollout["logprobs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    batch_size = obs.shape[0]
    minibatch_size = min(config.minibatch_size, batch_size)
    indices = np.arange(batch_size)
    policy_losses = []
    value_losses = []

    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            _, new_logprob, entropy, value = model.get_action_and_value(obs[mb_idx], actions[mb_idx])

            logratio = new_logprob - old_logprobs[mb_idx]
            ratio = logratio.exp()
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

    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
    }
