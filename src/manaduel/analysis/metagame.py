"""Deck-building metagame analysis (deck mode).

Four pieces:

1. DECK MATRIX: every deck plays every other deck (same bot on both sides, so only the
   decks differ). Cell (i, j) = how often deck i beats deck j, with a 95% interval.
2. METAGAME HEALTH: is there one dominant deck? Are there rock-paper-scissors cycles?
   The equilibrium mix (the best way to choose among decks when your opponent also
   picks the best way) and its support tell how many decks a healthy metagame needs.
   In this symmetric zero-sum setting the game value is 0, so the "exploitability" of
   a deck, or of a mix, is simply how much the best counter-deck beats it.
3. CARD POWER IN DECK CONTEXT: a second natural experiment. Random decks are given to
   both players, so the difference in copies of each card between the two decks is
   random; regressing the result on it (same estimator as the starting-hand version)
   gives the value of one more copy of a card in a 20-card deck.
4. DECK BUILDER: hill climbing that swaps one card for another to beat a field of decks,
   checked afterwards on NEW games so the improvement is not just luck.

Limits: results are for the player used (a hand-written bot such as the greedy bot, or a trained
agent given as "ppo:path/to/model.zip"), which does not plan around the metagame; "equilibrium" is
among the decks supplied, not all possible decks.
"""

import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from manaduel.agents.registry import ALL_BOT_FACTORIES as BOT_FACTORIES
from manaduel.analysis.card_effects import Games
from manaduel.engine.decks import copy_limit, random_deck
from manaduel.engine.game import play_game
from manaduel.sim.tournament import play_match


def make_factory(spec: str):
    """A bot name ("greedy") or a trained model ("ppo:path/to/model.zip") -> factory(seed) -> agent."""
    if spec.startswith("ppo:"):
        import torch

        torch.set_num_threads(1)
        from manaduel.agents.rl_agent import PPOAgent, load_ppo_model

        model = load_ppo_model(spec[4:])
        return lambda seed: PPOAgent(model)
    return BOT_FACTORIES[spec]


# --------------------------------------------------------------------------
# 1. The deck matrix
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DeckMatrix:
    names: tuple
    decks: tuple
    win: np.ndarray      # (n, n): how often the ROW deck beats the COLUMN deck (draw = half)
    low: np.ndarray
    high: np.ndarray
    draws: np.ndarray
    games: int


def _pair_job(job):
    config, bot, name_i, name_j, deck_i, deck_j, n_games, seed = job
    factory = make_factory(bot)
    r = play_match(config, name_i, factory, name_j, factory, n_games, seed, decks=(deck_i, deck_j))
    return r.score_a, r.interval_a(), r.draws / r.games


def deck_matrix(config, decks: dict, bot: str = "greedy", n_games: int = 2000, seed: int = 0,
                workers: int | None = None) -> DeckMatrix:
    names = list(decks)
    n = len(names)
    jobs = [(config, bot, names[i], names[j], decks[names[i]], decks[names[j]], n_games, seed)
            for i, j in itertools.combinations(range(n), 2)]
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    if workers <= 1:
        results = [_pair_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_pair_job, jobs, chunksize=1))

    win = np.full((n, n), 0.5)
    low = np.full((n, n), 0.5)
    high = np.full((n, n), 0.5)
    draws = np.zeros((n, n))
    for (i, j), (score, (lo, hi), draw) in zip(itertools.combinations(range(n), 2), results):
        win[i, j], win[j, i] = score, 1 - score
        low[i, j], high[i, j] = lo, hi
        low[j, i], high[j, i] = 1 - hi, 1 - lo
        draws[i, j] = draws[j, i] = draw
    return DeckMatrix(tuple(names), tuple(tuple(decks[k]) for k in names), win, low, high, draws, n_games)


# --------------------------------------------------------------------------
# 2. Metagame health
# --------------------------------------------------------------------------

def payoff(win: np.ndarray) -> np.ndarray:
    """Zero-sum payoff to the row deck: win rate minus one half (so the matrix is antisymmetric)."""
    return win - 0.5


def equilibrium(win: np.ndarray, iterations: int = 20000) -> np.ndarray:
    """A mix over decks that no single deck can beat (Hedge / multiplicative weights; the average
    of the iterates converges to an equilibrium of a symmetric zero-sum game)."""
    a = payoff(win)
    n = len(a)
    weights = np.ones(n)
    average = np.zeros(n)
    for t in range(1, iterations + 1):
        x = weights / weights.sum()
        average += x
        gain = a @ x                              # how each pure deck does against the current mix
        weights *= np.exp((2.0 / np.sqrt(t)) * gain)   # shift weight toward decks that beat the mix
        weights /= weights.max()
    return average / iterations


def exploitability(win: np.ndarray, mix: np.ndarray) -> float:
    """How much the best single deck beats this mix (0 for a perfect equilibrium mix)."""
    return float(np.max(payoff(win) @ mix))


def pure_exploitability(win: np.ndarray) -> np.ndarray:
    """For each deck: how much the best counter-deck beats it (win rate minus one half)."""
    a = payoff(win)
    return np.max(a.T, axis=1)


def beats(matrix: DeckMatrix, margin: float = 0.0) -> np.ndarray:
    """edge[i, j] is True when deck i beats deck j clearly (the 95% interval is above one half + margin)."""
    return matrix.low > 0.5 + margin


def count_cycles(edges: np.ndarray) -> list[tuple[int, int, int]]:
    """Rock-paper-scissors triples: i beats j, j beats k, k beats i."""
    n = len(edges)
    found = []
    for i, j, k in itertools.combinations(range(n), 3):
        for a, b, c in ((i, j, k), (i, k, j)):
            if edges[a, b] and edges[b, c] and edges[c, a]:
                found.append((a, b, c))
    return found


def dominant_decks(matrix: DeckMatrix, threshold: float = 0.55) -> list[str]:
    """Decks that beat the average other deck by a wide margin."""
    n = len(matrix.names)
    mean_vs_others = (matrix.win.sum(axis=1) - 0.5) / (n - 1)
    return [matrix.names[i] for i in range(n) if mean_vs_others[i] >= threshold]


# --------------------------------------------------------------------------
# 3. Card power in deck context
# --------------------------------------------------------------------------

def collect_decks(config, factory, n_games: int, seed: int = 0, max_copies: int | None = None) -> Games:
    """Mirror games between two RANDOM decks; the unit of analysis is the difference in copies of
    each card between the seat-0 deck and the seat-1 deck. Feed the result to
    `card_effects.estimate_effects`."""
    rng = random.Random(seed)
    n_cards = len(config.cards)
    score0 = np.empty(n_games)
    delta = np.empty((n_games, n_cards))
    for g in range(n_games):
        d0, d1 = random_deck(config, rng, max_copies), random_deck(config, rng, max_copies)
        agents = (factory(seed + 2 * g), factory(seed + 2 * g + 1))
        result = play_game(config, agents, seed=seed + 1_000_003 * (g + 1), decks=(d0, d1))
        score0[g] = 0.5 if result.winner is None else (1.0 if result.winner == 0 else 0.0)
        delta[g] = np.array(d0) - np.array(d1)
    return Games(tuple(c.name for c in config.cards), score0, delta)


# --------------------------------------------------------------------------
# 4. Deck builder
# --------------------------------------------------------------------------

def _field_score(job) -> float:
    config, bot, deck, field_decks, n_games, seed = job
    factory = make_factory(bot)
    scores = [play_match(config, "me", factory, f"f{k}", factory, n_games, seed, decks=(deck, other)).score_a
              for k, other in enumerate(field_decks)]
    return float(np.mean(scores))


def field_scores(config, bot, decks: list, field_decks: list, n_games: int, seed: int, workers: int) -> list[float]:
    """Average win rate of each deck against every deck in the field, on the same games."""
    jobs = [(config, bot, d, tuple(field_decks), n_games, seed) for d in decks]
    if workers <= 1 or len(jobs) == 1:
        return [_field_score(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_field_score, jobs, chunksize=1))


def neighbours(deck: tuple, max_copies: int) -> list[tuple]:
    """Every deck reachable by swapping one card for a different one."""
    out = []
    for i, j in itertools.permutations(range(len(deck)), 2):
        if deck[i] > 0 and deck[j] < max_copies:
            d = list(deck)
            d[i] -= 1
            d[j] += 1
            out.append(tuple(d))
    return out


@dataclass(frozen=True)
class BuildStep:
    round: int
    swap: str
    deck: tuple
    score: float


def build_deck(config, start: tuple, field_decks: list, bot: str = "greedy", n_games: int = 300,
               seed: int = 0, max_copies: int | None = None, max_rounds: int = 8, min_gain: float = 0.01,
               workers: int | None = None, log=None) -> list[BuildStep]:
    """Hill climbing: swap one card for another while the average win rate against the field improves."""
    say = log or (lambda text: None)
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    names = [c.name for c in config.cards]
    max_copies = copy_limit(config, max_copies)
    deck = tuple(start)
    score = field_scores(config, bot, [deck], field_decks, n_games, seed, 1)[0]
    steps = [BuildStep(0, "(start)", deck, score)]
    say(f"start: {score:.1%} against the field")
    for rnd in range(1, max_rounds + 1):
        cands = neighbours(deck, max_copies)
        scores = field_scores(config, bot, cands, field_decks, n_games, seed, workers)
        best = int(np.argmax(scores))
        if scores[best] < score + min_gain:
            say(f"round {rnd}: no swap improves by {min_gain:.0%}; done")
            break
        out_i = next(i for i in range(len(deck)) if cands[best][i] < deck[i])
        in_i = next(i for i in range(len(deck)) if cands[best][i] > deck[i])
        deck, score = cands[best], scores[best]
        swap = f"-{names[out_i]} +{names[in_i]}"
        steps.append(BuildStep(rnd, swap, deck, score))
        say(f"round {rnd}: {swap}: {score:.1%} against the field")
    return steps


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def format_matrix(m: DeckMatrix) -> str:
    n = len(m.names)
    width = max(10, max(len(x) for x in m.names) + 1)
    lines = [f"Deck matrix: {m.games} games per pairing (row deck beats column deck; draw = half)",
             " " * width + "".join(f"{x:>{width}}" for x in m.names) + f"{'average':>{width}}"]
    for i, name in enumerate(m.names):
        cells = "".join(f"{'-':>{width}}" if i == j else f"{m.win[i, j]:>{width}.1%}" for j in range(n))
        avg = (m.win[i].sum() - 0.5) / (n - 1)
        lines.append(f"{name:>{width}}{cells}{avg:>{width}.1%}")
    return "\n".join(lines)


def format_health(m: DeckMatrix, config) -> str:
    from manaduel.engine.decks import describe

    n = len(m.names)
    mix = equilibrium(m.win)
    lines = ["", "Metagame health"]
    dom = dominant_decks(m)
    lines.append(f"  dominant decks (beat the average other deck by 55% or more): {dom or 'none'}")
    cycles = count_cycles(beats(m))
    lines.append(f"  clear rock-paper-scissors cycles: {len(cycles)}"
                 + (" e.g. " + " > ".join(m.names[k] for k in cycles[0]) + " > " + m.names[cycles[0][0]] if cycles else ""))
    lines.append("  best counter to each deck (how far above 50%) and equilibrium weight:")
    counters = pure_exploitability(m.win)
    for i in np.argsort(-mix):
        lines.append(f"    {m.names[i]:<14} counter wins {0.5 + counters[i]:5.1%}   equilibrium weight {mix[i]:6.1%}   "
                     f"{describe(config, m.decks[i])}")
    support = int(np.sum(mix > 0.02))
    lines.append(f"  equilibrium uses {support} of {n} decks; exploitability of the uniform mix "
                 f"{exploitability(m.win, np.full(n, 1 / n)):.1%}, of the equilibrium mix {exploitability(m.win, mix):.2%}")
    return "\n".join(lines)


def write_matrix_csv(m: DeckMatrix, path) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_deck", "column_deck", "row_wins", "low", "high", "draws"])
        for i, a in enumerate(m.names):
            for j, b in enumerate(m.names):
                if i != j:
                    w.writerow([a, b, f"{m.win[i, j]:.4f}", f"{m.low[i, j]:.4f}", f"{m.high[i, j]:.4f}",
                                f"{m.draws[i, j]:.4f}"])
