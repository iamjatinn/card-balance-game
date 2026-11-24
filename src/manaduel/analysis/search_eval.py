"""How strong is the tree-search bot, and how does its strength grow with thinking time?

The MCTS bot is slow (about a second or two per game), so the games are split into small jobs and run
on several processes. Every opponent gets the SAME games for every MCTS setting (same deal seeds,
seats swapped), so settings are compared on equal footing.

An opponent is named either as one of the hand-written bots ("greedy") or as "ppo:path/to/model.zip".
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from manaduel.sim.stats import wilson_interval


@dataclass(frozen=True)
class Row:
    iterations: int
    rollout: str
    opponent: str
    games: int
    points: float          # wins + 0.5 * draws, for the MCTS bot

    @property
    def score(self) -> float:
        return self.points / self.games

    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.points, self.games)


def _load_config(config_path):
    from pathlib import Path

    from manaduel.engine.rules import load_game_config

    return load_game_config(Path(config_path)) if config_path else load_game_config()


def _opponent_factory(name: str):
    from manaduel.agents.registry import ALL_BOT_FACTORIES

    if name.startswith("ppo:"):
        from manaduel.agents.rl_agent import PPOAgent, load_ppo_model

        model = load_ppo_model(name[4:])
        return lambda seed: PPOAgent(model)
    return ALL_BOT_FACTORIES[name]


def _job(job) -> float:
    """Worker: MCTS points over games [first, first + count) against one opponent."""
    config_path, iterations, rollout, opponent, first, count, seed = job
    import torch

    torch.set_num_threads(1)
    from manaduel.agents.mcts import MCTSAgent
    from manaduel.sim.tournament import play_match

    config = _load_config(config_path)
    factory = _opponent_factory(opponent)
    # play_match numbers games from 0; shifting the seed moves to a different set of games, and chunks
    # are even-sized so the seat swap pattern (every other game) stays aligned.
    result = play_match(config, "mcts", lambda s: MCTSAgent(seed=s, iterations=iterations, rollout=rollout),
                        opponent, factory, count, seed=seed + 7_000_021 * (first // count))
    return result.points_a


def evaluate(iterations_list, rollouts, opponents, n_games: int, seed: int = 31337, chunk: int = 20,
             workers: int | None = None, config_path=None, progress=None) -> list[Row]:
    if chunk % 2:
        raise ValueError("chunk must be even so seats alternate evenly")
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    settings = [(it, ro, opp) for it in iterations_list for ro in rollouts for opp in opponents]
    jobs, owners = [], []
    for k, (it, ro, opp) in enumerate(settings):
        for first in range(0, n_games, chunk):
            jobs.append((config_path, it, ro, opp, first, min(chunk, n_games - first), seed))
            owners.append(k)
    totals = [0.0] * len(settings)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, points in zip(owners, pool.map(_job, jobs, chunksize=1)):
            totals[k] += points
            done += 1
            if progress and done % 20 == 0:
                progress(done, len(jobs))
    return [Row(it, ro, opp, n_games, totals[k]) for k, (it, ro, opp) in enumerate(settings)]


def format_rows(rows: list[Row]) -> str:
    lines = [f"{'iterations':>10} {'rollout':<8} {'opponent':<28} {'games':>6} {'MCTS wins':>10} {'95% interval':>16}"]
    for r in rows:
        lo, hi = r.interval()
        name = os.path.basename(r.opponent) if r.opponent.startswith("ppo:") else r.opponent
        lines.append(f"{r.iterations:>10} {r.rollout:<8} {name:<28} {r.games:>6} {r.score:>10.1%} "
                     f"{f'[{lo:.1%}, {hi:.1%}]':>16}")
    return "\n".join(lines)
