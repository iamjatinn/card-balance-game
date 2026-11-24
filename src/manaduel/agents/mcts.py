"""A Monte Carlo tree search bot that plays without seeing the opponent's hand.

Why this is not just textbook MCTS: the opponent's hand is hidden. Search that read the
real opponent hand would be cheating. This bot uses *information-set MCTS with
determinization* (single-observer ISMCTS):

  * every search iteration first GUESSES the opponent's hidden hand: it deals the opponent as many
    random cards as they really hold (the hand SIZE is public), each drawn by the cards' draw weights;
  * the card draws that come later are also re-randomised in every iteration, so the tree cannot learn
    "the next card I draw is a Fireball" from the game's random-number state;
  * the tree is keyed by the sequence of actions (not by states), and each iteration only follows
    moves that are legal in that iteration's guess (the usual "availability" correction for UCT);
  * after the tree, a rollout plays the game out with a cheap policy; the result (win 1, draw 0.5,
    loss 0, from the searching player's view) is passed back up. A node where the opponent moves
    picks the move that is best for the opponent.

The bot returns the most visited move at the root. A test checks the key honesty property: its move
does not change if the opponent's hand is swapped for a different hand of the same size.

Limits: it supports the endless-draw game (the original cards and the extended cards). In deck mode
the opponent's deck is also hidden, and that is not modelled; the constructor refuses.
The guess for the opponent's hand ignores what they have played so far (in this game that carries
almost no information), and rollouts are a rough model of how both sides play.
"""

import math
import random
from dataclasses import replace

from manaduel.agents.base import Agent
from manaduel.agents.heuristic import GreedyAgent
from manaduel.engine.rules import GameConfig, legal_actions, step
from manaduel.engine.state import GameState, is_terminal, winner


class _Node:
    __slots__ = ("children", "visits", "value", "avail")

    def __init__(self):
        self.children: dict[int, "_Node"] = {}
        self.visits = 0       # times this node was reached
        self.value = 0.0      # sum of results from the view of whoever chose the move into this node
        self.avail = 0        # times the parent was visited while this move was legal


class MCTSAgent(Agent):
    name = "mcts"

    def __init__(self, seed: int = 0, iterations: int = 200, exploration: float = 0.7,
                 rollout: str = "random"):
        """iterations: search iterations per decision (more = stronger, slower).
        exploration: the UCT constant (rewards are between 0 and 1).
        rollout: how the game is played out after the tree: "random" (any legal move) or "greedy"
        (the greedy bot plays both sides)."""
        if rollout not in ("random", "greedy"):
            raise ValueError("rollout must be 'random' or 'greedy'")
        self.iterations = iterations
        self.exploration = exploration
        self.rollout = rollout
        self._rng = random.Random(seed)
        self._greedy = GreedyAgent(seed) if rollout == "greedy" else None

    # -- the hidden information -------------------------------------------------------

    def _guess(self, state: GameState, me: int, config: GameConfig) -> GameState:
        """One possible world: the opponent's hand re-dealt at random (same size), the draw generator reset."""
        opp = state.players[1 - me]
        size = sum(opp.hand)
        weights = [c.weight for c in config.cards]
        counts = [0] * len(config.cards)
        for card in self._rng.choices(range(len(weights)), weights=weights, k=size):
            counts[card] += 1
        players = list(state.players)
        players[1 - me] = replace(opp, hand=tuple(counts))
        return replace(state, players=tuple(players), rng=self._rng.randrange(2**31))

    # -- search -----------------------------------------------------------------------

    def _rollout(self, state: GameState, config: GameConfig) -> GameState:
        max_turns = config.rules.max_turns
        while not is_terminal(state, max_turns):
            legal = legal_actions(state, config)
            if self._greedy is not None:
                action = self._greedy.act(state, legal, config)
            else:
                action = self._rng.choice(legal)
            state = step(state, action, config)
        return state

    def _select(self, node: _Node, legal: list[int]) -> int:
        """UCT among the children that are legal now. Values are stored from the view of whoever chose the move,
        so every chooser (root player or opponent) simply maximises its own result."""
        best, best_score = None, -math.inf
        for action in legal:
            child = node.children[action]
            mean = child.value / child.visits if child.visits else 0.5
            score = mean + self.exploration * math.sqrt(math.log(max(child.avail, 1)) / max(child.visits, 1))
            if score > best_score:
                best, best_score = action, score
        return best

    def act(self, state: GameState, legal: list[int], config: GameConfig) -> int:
        if config.uses_decks:
            raise NotImplementedError("MCTSAgent does not model hidden decks; use a game without deck_size")
        if len(legal) == 1:
            return legal[0]
        me = state.current
        max_turns = config.rules.max_turns
        root = _Node()
        for _ in range(self.iterations):
            world = self._guess(state, me, config)
            node, path = root, []
            while not is_terminal(world, max_turns):
                moves = legal_actions(world, config)
                for a in moves:
                    if a in node.children:
                        node.children[a].avail += 1
                untried = [a for a in moves if a not in node.children]
                chooser = world.current
                if untried:
                    action = self._rng.choice(untried)
                    child = _Node()
                    child.avail = 1
                    node.children[action] = child
                    path.append((child, chooser))
                    world = step(world, action, config)
                    break
                action = self._select(node, moves)
                child = node.children[action]
                path.append((child, chooser))
                node = child
                world = step(world, action, config)
            world = self._rollout(world, config)
            w = winner(world, max_turns)
            result = 0.5 if w is None else (1.0 if w == me else 0.0)
            for child, chooser in path:
                child.visits += 1
                child.value += result if chooser == me else 1.0 - result
        return max(legal, key=lambda a: (root.children[a].visits if a in root.children else -1, a == config.end_turn))
