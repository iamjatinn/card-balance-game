import random

from manaduel.agents.base import Agent


class RandomAgent(Agent):
    """Picks uniformly among the legal actions. The weakest sensible baseline."""

    name = "random"

    def __init__(self, seed: int | None = None):
        # Its own random generator, so the same seed replays the same game.
        self._rng = random.Random(seed)

    def act(self, state, legal_actions, config):
        return self._rng.choice(legal_actions)
