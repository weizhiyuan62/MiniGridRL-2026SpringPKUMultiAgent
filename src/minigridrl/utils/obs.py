import numpy as np


def obs_to_tabular_state(obs):
    """Convert a symbolic MiniGrid observation into a hashable Q-table key."""
    image = obs["image"]
    direction = int(obs["direction"])
    object_layer = image[:, :, 2].astype(np.int16).flatten()
    return (direction, *object_layer.tolist())


def obs_to_vector(obs):
    """Convert a symbolic MiniGrid observation into a normalized neural-net vector."""
    image = obs["image"].astype(np.float32)
    width, height = image.shape[:2]

    x_layer = image[:, :, 0] / max(width - 1, 1)
    y_layer = image[:, :, 1] / max(height - 1, 1)
    object_layer = image[:, :, 2] / 10.0

    direction = int(obs["direction"])
    direction_one_hot = np.zeros(4, dtype=np.float32)
    direction_one_hot[direction] = 1.0

    flat_image = np.stack([x_layer, y_layer, object_layer], axis=-1).flatten()
    return np.concatenate([flat_image, direction_one_hot]).astype(np.float32)


def obs_vector_dim(env):
    obs, _ = env.reset(seed=0)
    return int(obs_to_vector(obs).shape[0])
