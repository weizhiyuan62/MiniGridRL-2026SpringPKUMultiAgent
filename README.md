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
## Task 2 Training Entry

### tabular Q-learning:
```zsh
# --env: can be set to a special env to debug. "all" means run the 4 env in order
python train.py \
  --algo q_learning \
  --env 'MiniGrid-Dynamic-Obstacles-5x5-v0' \
  --total-steps 50000 \
  --seed 23

python train.py \
  --algo q_learning \
  --env 'MiniGrid-Dynamic-Obstacles-Random-5x5-v0' \
  --total-steps 50000 \
  --seed 23

python train.py \
  --algo q_learning \
  --env 'MiniGrid-Dynamic-Obstacles-8x8-v0' \
  --total-steps 100000 \
  --seed 23
  
python train.py \
  --algo q_learning \
  --env 'MiniGrid-Dynamic-Obstacles-16x16-v0' \
  --total-steps 200000 \
  --seed 23
```

### PPO:
```zsh
python train.py \
  --algo ppo \
  --env MiniGrid-Dynamic-Obstacles-5x5-v0 \
  --total-steps 50000 \

python train.py \
  --algo ppo \
  --env 'MiniGrid-Dynamic-Obstacles-Random-5x5-v0' \
  --total-steps 50000

python train.py \
  --algo ppo \
  --env MiniGrid-Dynamic-Obstacles-8x8-v0 \
  --total-steps 50000

python train.py \
  --algo ppo \
  --env MiniGrid-Dynamic-Obstacles-16x16-v0 \
  --total-steps 50000
```

Logs are written to `results/`, and saved checkpoints are written to `checkpoints/`.
PPO also writes per-update debug logs to `logs/`, including rollout outcomes,
action fractions, reward statistics, advantage statistics, and PPO diagnostics.

### Multi-seed learning curves(for visualize)
```zsh
# Run Q-learning and PPO-Clip on all four environments with five seeds, 
# then plot the seed-averaged episode-return curves:
# --env: select env to do experiment
python script/run_seed_average.py \
  --seeds 1 23 31 42 97 \
  --env MiniGrid-Dynamic-Obstacles-16x16-v0 \
  --algo all \
  --total-steps 130000
  # In my experiment, I run 5*5 for 50,000 step, and the others for 100,000 steps
```

- The script writes averaged plots and per-step mean/std CSV files to
`results/averaged/`.
- PPO outputs include the entropy coefficient in the filename, for example
`steps_50000_seed_23_entropy_0p01.png`, so entropy sweeps do not overwrite each other.

### checkpoint test
```zsh
python script/test_ckpt.py \
  --ckpt checkpoints/qlearn/Dynamic_Obstacles_5x5/steps_50000_seed_23.pkl \
  --env MiniGrid-Dynamic-Obstacles-5x5-v0

python script/test_ckpt.py \
  --ckpt checkpoints/ppo/Dynamic_Obstacles_5x5/steps_50000_seed_0.pt \
  --env MiniGrid-Dynamic-Obstacles-5x5-v0 \
  --device cpu
```
