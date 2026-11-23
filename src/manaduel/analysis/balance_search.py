"""A second, stronger attempt at automatic balancing.

Why v1 (`autobalance.py`) fell short: it could only move a card's stat by whole
numbers (a big step in a 20 HP game), looked only at flagged cards, and gave up as
soon as one step failed. Here:

  * FINER STEPS: `scale_config` multiplies every HP/damage/heal/shield number, so the
    same game can be tuned in smaller steps (a scaled game plays identically: the
    bots decide the same way, which a test checks).
  * MORE LEVERS: every effect number, every cost, plus a few rules (shield cap, how
    many cards the second player gets, cards drawn per turn).
  * COARSE-TO-FINE steepest descent: try every single change at the current step
    size, take the best, and only shrink the step when nothing helps.
  * A RICHER OBJECTIVE, so fixing one problem cannot silently cause another:

        J = mean over cards of (card power in points)^2
            + w_draw * (points of draw rate above the target)^2
            + w_first * (points of first-player advantage away from 50%)^2

  * TUNED ON SEVERAL BOTS AT ONCE (J is averaged over them) and VERIFIED on new
    deals, so the result is not just an artefact of one bot or one set of seeds.

All candidate games are played with the same seeds ("common random numbers"), so
two candidates differ because of the rules, not because of luck.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from manaduel.agents.registry import BOT_FACTORIES
from manaduel.analysis.card_effects import collect, estimate_effects
from manaduel.analysis.sweep import apply_overrides
from manaduel.engine.rules import GameConfig

STAT_FIELDS = ("damage", "heal", "shield")


def scale_config(config: GameConfig, factor: int) -> GameConfig:
    """The same game with HP, damage, heal, shield and the shield cap multiplied by `factor`.
    Costs, mana and draws are unchanged, so tuning can move in finer steps."""
    cards = tuple(
        replace(c, damage=c.damage * factor, heal=c.heal * factor, shield=c.shield * factor)
        for c in config.cards
    )
    rules = replace(config.rules, max_hp=config.rules.max_hp * factor,
                    max_shield=config.rules.max_shield * factor)
    return GameConfig(cards=cards, rules=rules)


@dataclass(frozen=True)
class Lever:
    key: str          # e.g. "cards.fireball.damage" or "rules.max_shield"
    low: int
    high: int
    coarse_ok: bool   # may move in steps larger than 1 (effect sizes), unlike costs and counts


def get_value(config: GameConfig, key: str) -> int:
    parts = key.split(".")
    if parts[0] == "rules":
        return getattr(config.rules, parts[1])
    card = next(c for c in config.cards if c.name == parts[1])
    return getattr(card, parts[2])


def levers_for(config: GameConfig) -> list[Lever]:
    out = []
    for c in config.cards:
        for stat in STAT_FIELDS:
            value = getattr(c, stat)
            if value > 0:
                out.append(Lever(f"cards.{c.name}.{stat}", 1, 3 * value, True))
        out.append(Lever(f"cards.{c.name}.cost", 1, config.rules.max_mana, False))
    out.append(Lever("rules.max_shield", 1, 4 * config.rules.max_shield, True))
    out.append(Lever("rules.second_player_extra_cards", -1, 2, False))
    out.append(Lever("rules.draw_per_turn", 1, 2, False))
    return out


def candidates(config: GameConfig, levers: list[Lever], step: int) -> list[dict]:
    """Every single change of one lever by +/- step (cost-like levers always move by 1)."""
    moves = []
    for lever in levers:
        size = step if lever.coarse_ok else 1
        current = get_value(config, lever.key)
        for new in (current - size, current + size):
            if lever.low <= new <= lever.high and new != current:
                moves.append({lever.key: new})
    return moves


@dataclass(frozen=True)
class Weights:
    draw_target: float = 0.15
    w_draw: float = 0.5
    w_first: float = 0.5


def objective(effects: list[float], draw_rate: float, first_player: float, w: Weights = Weights()):
    """Returns (J, power term, draw term, first-player term); everything in 'points'."""
    power = float(np.mean([(100 * e) ** 2 for e in effects]))
    draw = (100 * max(0.0, draw_rate - w.draw_target)) ** 2
    first = (100 * (first_player - 0.5)) ** 2
    return power + w.w_draw * draw + w.w_first * first, power, draw, first


@dataclass(frozen=True)
class Eval:
    objective: float               # J, averaged over the bots
    power: float                   # the card-power part of J, from effects averaged over bots
    draw_rate: float
    first_player: float            # first player's win probability with identical hands
    effects: dict                  # card name -> effect (mean over bots)
    flagged: tuple                 # card names flagged by ANY of the bots


def _evaluate_job(job) -> Eval:
    config, bots, n_games, seed, weights = job
    effects, draws, firsts, flagged = [], [], [], set()
    for bot in bots:
        games = collect(config, BOT_FACTORIES[bot], n_games, seed)
        report = estimate_effects(games)
        effects.append([e.effect for e in report.effects])
        draws.append(float(np.mean(games.score0 == 0.5)))
        firsts.append(report.seat0_baseline)
        flagged |= {e.name for e in report.flagged()}
    mean_effects = np.mean(effects, axis=0)
    draw_rate, first_player = float(np.mean(draws)), float(np.mean(firsts))
    j = [objective(list(np.array(effects)[b]), draws[b], firsts[b], weights)[0] for b in range(len(bots))]
    power = float(np.mean([(100 * e) ** 2 for e in mean_effects]))
    names = [c.name for c in config.cards]
    return Eval(float(np.mean(j)), power, draw_rate, first_player,
                dict(zip(names, map(float, mean_effects))), tuple(sorted(flagged)))


def evaluate_many(configs: list[GameConfig], bots, n_games: int, seed: int, weights: Weights,
                  workers: int) -> list[Eval]:
    jobs = [(c, tuple(bots), n_games, seed, weights) for c in configs]
    if workers <= 1 or len(jobs) == 1:
        return [_evaluate_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_evaluate_job, jobs, chunksize=1))


@dataclass(frozen=True)
class Move:
    iteration: int
    step: int
    change: dict          # {} for the starting point
    eval: Eval


@dataclass(frozen=True)
class SearchResult:
    moves: list
    config: GameConfig
    stop_reason: str
    evaluations: int


def search(config: GameConfig, bots=("greedy", "defensive"), n_games: int = 8000, seed: int = 0,
           schedule=(4, 2, 1), max_moves: int = 25, min_gain: float = 0.02,
           weights: Weights = Weights(), workers: int | None = None, log=None) -> SearchResult:
    """Coarse-to-fine steepest descent on the objective J."""
    say = log or (lambda text: None)
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    levers = levers_for(config)
    current = evaluate_many([config], bots, n_games, seed, weights, 1)[0]
    moves = [Move(0, 0, {}, current)]
    evaluations = 1
    say(f"start: J={current.objective:.2f}, draws {current.draw_rate:.1%}, "
        f"first player {current.first_player:.1%}, flagged {list(current.flagged)}")

    stop = "move budget used"
    n_moves = 0
    for step in schedule:
        while n_moves < max_moves:
            cands = candidates(config, levers, step)
            configs = [apply_overrides(config, c) for c in cands]
            evals = evaluate_many(configs, bots, n_games, seed, weights, workers)
            evaluations += len(evals)
            best = min(range(len(evals)), key=lambda i: evals[i].objective)
            if evals[best].objective < current.objective * (1 - min_gain):
                n_moves += 1
                config, current = configs[best], evals[best]
                moves.append(Move(n_moves, step, cands[best], current))
                key, value = next(iter(cands[best].items()))
                say(f"move {n_moves} (step {step}): {key} -> {value}: J={current.objective:.2f}, "
                    f"draws {current.draw_rate:.1%}, first player {current.first_player:.1%}, "
                    f"flagged {list(current.flagged)}")
            else:
                say(f"step {step}: no change improves J by {min_gain:.0%}; going finer"
                    if step != schedule[-1] else f"step {step}: no change improves J; done")
                break
        else:
            break
    else:
        stop = "no change helps at the finest step"
    return SearchResult(moves, config, stop, evaluations)


def format_moves(result: SearchResult) -> str:
    lines = [f"{'#':>2} {'step':>4}  {'change':<38} {'J':>8} {'draws':>7} {'1st player':>10}  flagged cards"]
    for m in result.moves:
        change = ", ".join(f"{k}={v}" for k, v in m.change.items()) or "(starting game)"
        lines.append(f"{m.iteration:>2} {m.step:>4}  {change:<38} {m.eval.objective:>8.2f} "
                     f"{m.eval.draw_rate:>7.1%} {m.eval.first_player:>10.1%}  {', '.join(m.eval.flagged) or '-'}")
    lines.append(f"Stopped: {result.stop_reason} ({result.evaluations} configurations tested)")
    return "\n".join(lines)
