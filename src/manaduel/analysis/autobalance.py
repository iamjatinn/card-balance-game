"""Suggest card changes that remove flagged outliers, one small step at a time.

Each round:
  1. measure every card's power (analysis.card_effects) in mirror matches
  2. for every flagged card, try the two nearest fixes: nudge its main effect
     by 1, or its cost by 1 (too strong -> weaker effect / higher cost; too
     weak -> the opposite)
  3. discard fixes that make draws (stalled games) clearly more common
  4. keep whichever remaining fix lowers the total squared effect the most;
     stop when no card is flagged, no fix helps, or the step budget runs out

Every candidate is measured on the SAME deals (same seeds), so differences
between candidates are not just luck of the draw. Because the tuning data is
reused, always re-check the result on new seeds and a different bot (`verify`).
"""

from dataclasses import dataclass

import yaml

from manaduel.analysis.card_effects import EffectReport, collect, estimate_effects
from manaduel.analysis.sweep import apply_overrides

import numpy as np

MAIN_STATS = ("damage", "heal", "shield")


@dataclass(frozen=True)
class Change:
    card: str
    field: str
    old: int
    new: int

    def describe(self) -> str:
        return f"{self.card}.{self.field}: {self.old} -> {self.new}"


@dataclass(frozen=True)
class Step:
    iteration: int
    changes: tuple[Change, ...]   # empty for the starting point (a step can change two cards)
    report: EffectReport
    seat0_baseline: float     # first player's win rate with identical hands
    draw_rate: float
    objective: float          # sum of squared card effects (lower is more balanced)


@dataclass(frozen=True)
class AutobalanceResult:
    steps: list[Step]
    config: object            # the final GameConfig
    stop_reason: str


def main_stat(card) -> str:
    """The card's headline effect: the first of damage / heal / shield that it has."""
    for name in MAIN_STATS:
        if getattr(card, name) > 0:
            return name
    raise ValueError(f"card '{card.name}' has no effect")


def candidate_changes(config, name: str, too_strong: bool) -> list[Change]:
    """The nearest single-step fixes for one card."""
    card = next(c for c in config.cards if c.name == name)
    stat = main_stat(card)
    value, cost = getattr(card, stat), card.cost
    if too_strong:
        options = [Change(name, stat, value, value - 1), Change(name, "cost", cost, cost + 1)]
    else:
        options = [Change(name, stat, value, value + 1), Change(name, "cost", cost, cost - 1)]
    valid = []
    for c in options:
        if c.field == "cost" and not 1 <= c.new <= config.rules.max_mana:
            continue
        if c.field != "cost" and c.new < 1:
            continue
        valid.append(c)
    return valid


def _apply(config, changes):
    """Apply one change or a tuple of changes to a copy of the config."""
    if isinstance(changes, Change):
        changes = (changes,)
    return apply_overrides(config, {f"cards.{c.card}.{c.field}": c.new for c in changes})


def _moves(config, flagged):
    """Every single change for the flagged cards, plus every pair that changes two
    DIFFERENT cards (one card changed twice would just be a bigger single step)."""
    singles = []
    for target in sorted(flagged, key=lambda e: -abs(e.effect)):
        singles += candidate_changes(config, target.name, too_strong=target.effect > 0)
    moves = [(c,) for c in singles]
    for i, first in enumerate(singles):
        for second in singles[i + 1:]:
            if first.card != second.card:
                moves.append((first, second))
    return moves


def _evaluate(config, factory, n_games: int, seed: int):
    games = collect(config, factory, n_games, seed)
    report = estimate_effects(games)
    objective = float(sum(e.effect ** 2 for e in report.effects))
    draw_rate = float(np.mean(games.score0 == 0.5))
    return report, objective, draw_rate


def autobalance(config, factory, n_games: int = 10_000, seed: int = 0, max_steps: int = 15,
                min_gain: float = 0.03, max_draw_increase: float = 0.03, log=None) -> AutobalanceResult:
    """Greedy search for a config with no flagged cards. `log(text)` is called with progress.

    Fixes are tried for EVERY flagged card. A change is rejected if it makes the
    draw rate more than `max_draw_increase` (absolute, e.g. 0.03 = 3 points)
    higher than in the starting game: weaker damage tends to cause stalling, and
    card power alone would not notice.
    """
    say = log or (lambda text: None)
    report, objective, draw_rate = _evaluate(config, factory, n_games, seed)
    start_draw = draw_rate
    steps = [Step(0, (), report, report.seat0_baseline, draw_rate, objective)]
    stop = "step budget used"

    for iteration in range(1, max_steps + 1):
        flagged = report.flagged()
        if not flagged:
            stop = "no card is flagged"
            break
        say(f"step {iteration}: flagged {', '.join(f'{e.name} {e.effect:+.1%}' for e in flagged)}")

        best = None
        for move in _moves(config, flagged):
            trial = _apply(config, move)
            t_report, t_obj, t_draw = _evaluate(trial, factory, n_games, seed)
            too_many_draws = t_draw > start_draw + max_draw_increase
            note = "  (rejected: too many draws)" if too_many_draws else ""
            label = " + ".join(c.describe() for c in move)
            say(f"    {label:<52} -> squared effect {t_obj:.5f}, draws {t_draw:.1%}{note}")
            if too_many_draws:
                continue
            if best is None or t_obj < best[0]:
                best = (t_obj, move, trial, t_report, t_draw)

        if best is None or best[0] > objective * (1 - min_gain):
            stop = "no change (single or paired) improves the balance enough"
            break
        objective, move, config, report, draw_rate = best
        steps.append(Step(iteration, move, report, report.seat0_baseline, draw_rate, objective))
        say(f"    accepted: {' + '.join(c.describe() for c in move)}")

    return AutobalanceResult(steps=steps, config=config, stop_reason=stop)


def format_trajectory(result: AutobalanceResult) -> str:
    lines = [f"{'step':>4}  {'change':<52} {'flagged':>7} {'worst card':>16} {'1st player':>10} {'draws':>6}"]
    for s in result.steps:
        worst = max(s.report.effects, key=lambda e: abs(e.effect))
        change = " + ".join(c.describe() for c in s.changes) if s.changes else "(starting game)"
        lines.append(
            f"{s.iteration:>4}  {change:<52} {len(s.report.flagged()):>7} "
            f"{worst.name + ' ' + format(worst.effect, '+.1%'):>16} "
            f"{s.seat0_baseline:>10.1%} {s.draw_rate:>6.1%}"
        )
    lines.append(f"Stopped: {result.stop_reason}")
    return "\n".join(lines)


def config_to_yaml(config, path) -> None:
    """Write a config in the same format as configs/cards.yaml."""
    from dataclasses import asdict

    cards = {}
    for card in config.cards:
        stats = {k: v for k, v in asdict(card).items() if k != "name" and (v != 0)}
        if stats.get("weight") == 1:
            del stats["weight"]
        cards[card.name] = stats
    data = {"rules": asdict(config.rules), "cards": cards}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
