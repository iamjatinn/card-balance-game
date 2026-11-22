"""One place that lists the hand-written bots. Each entry builds a fresh bot from a seed."""

from manaduel.agents.heuristic import AggressiveAgent, DefensiveAgent, GreedyAgent
from manaduel.agents.random_agent import RandomAgent

BOT_FACTORIES = {
    "random": RandomAgent,
    "aggressive": AggressiveAgent,
    "defensive": DefensiveAgent,
    "greedy": GreedyAgent,
}
