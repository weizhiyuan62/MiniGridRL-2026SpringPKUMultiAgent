# MiniGridRL-2026SpringPKUMultiAgent

## Env Set
```zsh
# set the conda env 
conda create -n minigridrl python=3.10.6 -y
conda activate minigridrl

pip install -e .

```

After editable installation, project modules can be imported directly:
```python
from minigridrl.env import make_env
from minigridrl.models.q_learning import QLearningAgent
from minigridrl.models.ppo import ActorCritic
```

## Simulator Environment Debug Tools

Check raw and symbolic observations:
```zsh
python script/env_check.py
python script/env_check.py --env MiniGrid-Dynamic-Obstacles-8x8-v0
python script/env_check.py --env MiniGrid-Dynamic-Obstacles-5x5-v0 --no-grid
```

Print a text rollout with symbolic grids:
```zsh
python script/rollout_trace.py --env MiniGrid-Dynamic-Obstacles-5x5-v0 --steps 10
```

Watch a random policy in a MiniGrid render window:
```zsh
python script/watch_random_agent.py --env MiniGrid-Dynamic-Obstacles-5x5-v0 --steps 100 --delay 0.2
```

Manually interact with the environment:
```zsh
python script/play_env.py --env MiniGrid-Dynamic-Obstacles-5x5-v0 --symbolic
```

Controls for manual play: `a` turns left, `d` turns right, `w` moves forward, `r` resets, and `q` quits.

## Task 2 Training Entry

Train tabular Q-learning:
```zsh
# --env: can be set to a special env to debug. "all" means run the 4 env in order
python train.py \
  --algo q_learning \
  --env all \
  --total-steps 50000 \
  --seed 23
```

Train PPO:
```zsh
python train.py \
  --algo ppo \
  --env MiniGrid-Dynamic-Obstacles-5x5-v0 \
  --total-steps 50000 \
  --seed 0
```

Logs are written to `results/`, and saved checkpoints are written to `checkpoints/`.
