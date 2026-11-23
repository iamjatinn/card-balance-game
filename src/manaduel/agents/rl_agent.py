"""Wrap a trained PPO model so it plays through the same Agent interface as every other bot."""

import numpy as np

from manaduel.agents.base import Agent
from manaduel.rl.obs import observation, observation_size


def infer_memory(model, config) -> bool:
    """Was this model trained with the opponent-memory features? Decided from the size of
    the observation it expects. Raises if it was trained for a different card set."""
    expected = int(model.observation_space.shape[0])
    if expected == observation_size(config, memory=False):
        return False
    if expected == observation_size(config, memory=True):
        return True
    raise ValueError(
        f"this model expects {expected} observation numbers, but this game gives "
        f"{observation_size(config, False)} (or {observation_size(config, True)} with memory); "
        "it was probably trained on a different card set"
    )


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, model, memory: bool | None = None):
        self.model = model
        self._memory = memory   # None = work it out from the model on first use

    def act(self, state, legal_actions, config):
        if self._memory is None:
            self._memory = infer_memory(self.model, config)
        obs = observation(state, state.current, config, memory=self._memory)
        mask = np.zeros(config.n_actions, dtype=bool)
        mask[legal_actions] = True  # the network may only pick legal moves
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


def load_ppo_model(path):
    # Imported here so the rest of the project works without the training libraries.
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(path, device="cpu")
