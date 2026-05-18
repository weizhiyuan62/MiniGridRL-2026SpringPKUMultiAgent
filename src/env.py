import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from minigrid.core.constants import IDX_TO_OBJECT
from minigrid.wrappers import SymbolicObsWrapper


ENV_IDS = [
    "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
    "MiniGrid-Dynamic-Obstacles-16x16-v0",
]

DIRECTION_NAMES = {
    0: "right/east",
    1: "down/south",
    2: "left/west",
    3: "up/north",
}

ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
}

OBJECT_CHARS = {
    "unseen": "?",
    "empty": ".",
    "wall": "#",
    "floor": ".",
    "door": "D",
    "key": "K",
    "ball": "O",
    "box": "B",
    "goal": "G",
    "lava": "L",
    "agent": "A",
}


def make_env(env_id, symbolic=True, render_mode=None):
    """Create one project environment with the requested observation wrapper."""
    env = gym.make(env_id, render_mode=render_mode)
    if symbolic:
        env = SymbolicObsWrapper(env)
    return env


def get_env_ids(name):
    """Return all project env ids, or a single env id selected by name."""
    if name == "all":
        return ENV_IDS
    if name not in ENV_IDS:
        valid = ", ".join(["all", *ENV_IDS])
        raise ValueError(f"Unknown environment {name!r}. Valid choices: {valid}")
    return [name]


def symbolic_grid_to_text(image):
    """Render SymbolicObsWrapper image arrays as a compact ASCII grid."""
    rows = []
    width, height = image.shape[:2]
    for y in range(height):
        row = []
        for x in range(width):
            object_idx = int(image[x, y, 2])
            object_name = IDX_TO_OBJECT.get(object_idx, str(object_idx))
            row.append(OBJECT_CHARS.get(object_name, object_name[:1].upper()))
        rows.append(" ".join(row))
    return "\n".join(rows)


def summarize_observation(obs):
    """Return a short human-readable summary for a symbolic observation."""
    image = obs["image"]
    direction = int(obs["direction"])
    agent_positions = np.argwhere(image[:, :, 2] == _object_id("agent"))
    obstacle_positions = np.argwhere(image[:, :, 2] == _object_id("ball"))
    goal_positions = np.argwhere(image[:, :, 2] == _object_id("goal"))

    return {
        "image_shape": tuple(image.shape),
        "direction": direction,
        "direction_name": DIRECTION_NAMES.get(direction, str(direction)),
        "agent_positions": [tuple(map(int, pos)) for pos in agent_positions],
        "obstacle_positions": [tuple(map(int, pos)) for pos in obstacle_positions],
        "goal_positions": [tuple(map(int, pos)) for pos in goal_positions],
        "mission": obs.get("mission"),
    }


def _object_id(name):
    for idx, object_name in IDX_TO_OBJECT.items():
        if object_name == name:
            return idx
    raise KeyError(name)
