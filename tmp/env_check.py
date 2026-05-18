import gymnasium as gym
import minigrid
from minigrid.wrappers import SymbolicObsWrapper

env_list = [
    "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
    "MiniGrid-Dynamic-Obstacles-16x16-v0"
]

if __name__ == "__main__":
    for env_name in env_list:
        print(f"Testing environment: {env_name}")
        env = gym.make(env_name)
        obs, info = env.reset(seed=0)
        print(obs.keys())
        print(obs["image"].shape)
        print(env.action_space)