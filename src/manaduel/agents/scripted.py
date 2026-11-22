from manaduel.agents.base import Agent


class ScriptedAgent(Agent):
    """Plays a fixed list of moves by name, e.g. ["strike", "end_turn", ...].
    Used in tests to replay a known game exactly."""

    name = "scripted"

    def __init__(self, moves: list[str]):
        self._moves = list(moves)
        self._next = 0

    def act(self, state, legal_actions, config):
        move = self._moves[self._next]
        self._next += 1
        if move == "end_turn":
            return config.end_turn
        return next(i for i, card in enumerate(config.cards) if card.name == move)
