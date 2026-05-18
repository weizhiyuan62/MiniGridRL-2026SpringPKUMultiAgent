from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from minigridrl.utils.logging import CSVLogger
from minigridrl.utils.obs import obs_to_tabular_state


@dataclass
class QLearningConfig:
    # hyperparameters for the tabular Q-learning algorithm
    total_steps: int = 50_000
    learning_rate: float = 0.2
    gamma: float = 0.99         # discount factor for future rewards
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000
    seed: int = 23  # just select what you like, i like 23
    log_path: str = "results/q_learning.csv"


class QLearningAgent:
    def __init__(self, action_dim, config):
        self.action_dim = action_dim
        self.config = config
        # init q-table with 0. at first.
        self.q_table = defaultdict(lambda: np.zeros(action_dim, dtype=np.float32))  # (action_dim, )
        self.rng = np.random.default_rng(config.seed)

    def epsilon(self, step):
        frac = min(step / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, state, step):
        if self.rng.random() < self.epsilon(step):          # p(true) = epsilon, epsilon greedy...
            return int(self.rng.integers(self.action_dim))  # action sampled uniformly at random for exploration
        return int(np.argmax(self.q_table[state]))          # action with highest q-value for the current state

    def update(self, state, action, reward, next_state, done):
        # bootstrap update to q-table
        next_value = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.config.gamma * next_value
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.config.learning_rate * td_error
        return float(td_error)


def train_q_learning(env, config):
    agent = QLearningAgent(env.action_space.n, config)
    obs, _ = env.reset(seed=config.seed)
    # Convert MiniGrid observations into hashable symbolic states for the Q-table.
    state = obs_to_tabular_state(obs)
    episode_return = 0.0
    episode_length = 0
    episode = 0

    fields = [
        "step",
        "episode",
        "episode_return",
        "episode_length",
        "epsilon",
        "q_table_size",
        "success",
    ]
    with CSVLogger(config.log_path, fields) as logger:
        for step in tqdm(range(1, config.total_steps + 1), desc="Q-learning"):
            action = agent.act(state, step)
            # rollout one step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = obs_to_tabular_state(next_obs)
            agent.update(state, action, reward, next_state, done)   #(s_t, a_t, r_(s_t, a_t), s_{t+1}), 4-tuple transition for Q-learning update

            episode_return += float(reward)
            episode_length += 1
            state = next_state

            # done means the episode has ended
            if done:
                episode += 1
                # Log only completed episodes so success and return are comparable.
                logger.log(
                    {
                        "step": step,
                        "episode": episode,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "epsilon": agent.epsilon(step),
                        "q_table_size": len(agent.q_table),
                        "success": int(reward > 0),
                    }
                )
                # Start a fresh episode but keep the learned Q-table.
                obs, _ = env.reset()
                state = obs_to_tabular_state(obs)
                episode_return = 0.0
                episode_length = 0

    return agent
