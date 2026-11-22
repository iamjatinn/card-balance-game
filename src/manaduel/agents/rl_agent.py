"""Wrap a trained PPO model so it plays through the same Agent interface as every other bot."""

import numpy as np

from manaduel.agents.base import Agent
from manaduel.rl.obs import observation


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, model):
        self.model = model

    def act(self, state, legal_actions, config):
        obs = observation(state, state.current, config)
        mask = np.zeros(config.n_actions, dtype=bool)
        mask[legal_actions] = True  # the network may only pick legal moves
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


def load_ppo_model(path):
    # Imported here so the rest of the project works without the training libraries.
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(path, device="cpu")
