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
  --seed 0
```

Logs are written to `results/`, and saved checkpoints are written to `checkpoints/`.

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