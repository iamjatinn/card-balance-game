"""The one interface every bot implements."""


class Agent:
    """An agent looks at the state and picks one of the legal actions.

    Random, heuristic, MCTS and RL bots all subclass this, so tournaments and
    the viewer never care which kind of bot is playing.
    """

    name = "agent"

    def act(self, state, legal_actions: list[int], config) -> int:
        raise NotImplementedError
