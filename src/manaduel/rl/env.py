"""Gymnasium environment: the learning agent plays against a randomly chosen bot.

From the agent's point of view a 'step' is one action of its own (play a card or
end the turn). When it ends its turn, the opponent bot plays its whole turn
inside the same step() call, so the agent only ever sees its own decisions.

Reward is sparse: +1 for a win, -1 for a loss, 0 for a draw, and 0 on every
other step. The agent has to work out which earlier moves caused the result.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from manaduel.agents.registry import BOT_FACTORIES
from manaduel.engine.rules import legal_actions, load_game_config, new_game, step as game_step
from manaduel.engine.state import is_terminal, winner
from manaduel.rl.obs import observation, observation_size


class ManaDuelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, opponents=None, strict=True):
        """strict=True: an illegal action raises an error, so a masking bug can never
        hide during training. strict=False: an illegal action counts as 'end turn'
        (only for Gymnasium's checker, which tries random actions)."""
        super().__init__()
        self.strict = strict
        self.config = config or load_game_config()
        self.opponents = dict(opponents or BOT_FACTORIES)  # name -> factory(seed)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(observation_size(self.config),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.config.n_actions)
        self.state = None
        self.learner = 0       # which seat the agent plays this episode (0 moves first)
        self.opponent = None

    # -- helpers ----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        return observation(self.state, self.learner, self.config)

    def _opponent_moves(self) -> None:
        """Let the opponent act until it is the learner's turn again (or the game ends)."""
        max_turns = self.config.rules.max_turns
        while not is_terminal(self.state, max_turns) and self.state.current != self.learner:
            legal = legal_actions(self.state, self.config)
            action = self.opponent.act(self.state, legal, self.config)
            self.state = game_step(self.state, action, self.config)

    def action_masks(self) -> np.ndarray:
        """True for each action that is legal right now (MaskablePPO uses this)."""
        mask = np.zeros(self.config.n_actions, dtype=bool)
        mask[legal_actions(self.state, self.config)] = True
        return mask

    # -- gymnasium API ----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        rng = self.np_random
        self.learner = int(rng.integers(2))
        names = list(self.opponents)
        name = names[int(rng.integers(len(names)))]
        self.opponent = self.opponents[name](int(rng.integers(2**31 - 1)))
        self.state = new_game(self.config, int(rng.integers(2**31 - 1)))
        self._opponent_moves()  # if we play second, the opponent goes first
        return self._obs(), {"opponent": name, "learner_side": self.learner}

    def step(self, action):
        action = int(action)
        if action not in legal_actions(self.state, self.config):
            if self.strict:
                raise ValueError(f"illegal action {action}")
            action = self.config.end_turn
        self.state = game_step(self.state, action, self.config)
        self._opponent_moves()

        terminated = is_terminal(self.state, self.config.rules.max_turns)
        reward = 0.0
        if terminated:
            w = winner(self.state, self.config.rules.max_turns)
            if w is not None:
                reward = 1.0 if w == self.learner else -1.0
        return self._obs(), reward, terminated, False, {}
