"""One place that lists the hand-written bots. Each entry builds a fresh bot from a seed."""

from manaduel.agents.heuristic import AggressiveAgent, BoldAgent, CautiousAgent, DefensiveAgent, GreedyAgent
from manaduel.agents.random_agent import RandomAgent

# The four bots used for training, evaluation and the default tournament (unchanged since v0.1).
BOT_FACTORIES = {
    "random": RandomAgent,
    "aggressive": AggressiveAgent,
    "defensive": DefensiveAgent,
    "greedy": GreedyAgent,
}

# Extra bots for testing card sets that use the new mechanics: like greedy, but with different
# valuations, and they play every kind of card.
EXTRA_BOT_FACTORIES = {
    "cautious": CautiousAgent,
    "bold": BoldAgent,
}

ALL_BOT_FACTORIES = {**BOT_FACTORIES, **EXTRA_BOT_FACTORIES}
