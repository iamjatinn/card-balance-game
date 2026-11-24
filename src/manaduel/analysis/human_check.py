"""Do the bot-based card-power verdicts hold up for human players?

`play` logs every game you (or friends) play against a bot, including the cards each side was
dealt. The deal is random, so the same natural experiment as in card_effects.py works on human games:
regress the human's result on (copies the human was dealt minus copies the bot was dealt).

What this can and cannot do:
  * It needs MANY games. With a few dozen games the intervals are wider than any plausible effect,
    and the report says so and how many games it would take to see effects of a given size.
  * Humans play against a bot, not a copy of themselves, so an effect here is "a card in a human's hand
    against the bot's hand"; it is compared with the same quantity measured for the bot (mirror games).
  * Players learn, and different people play differently: the effects describe the pooled group.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manaduel.analysis.card_effects import Games


@dataclass(frozen=True)
class HumanGames:
    games: Games
    skipped: int          # logged rows without a recorded deal (older logs) or from another card set or bot


def load_human_games(path, config, config_name: str | None = None, bot: str | None = None) -> HumanGames:
    """Read the games log (runs/human_games.csv) into the structure card_effects.estimate_effects uses."""
    log = Path(path)
    if not log.exists():
        raise FileNotFoundError(f"no games log at {path}; play some games first: python -m manaduel.cli play")
    n_cards = len(config.cards)
    scores, deltas, skipped = [], [], 0
    with open(log, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (config_name and row.get("config") != config_name) or (bot and row.get("bot") != bot):
                skipped += 1
                continue
            try:
                mine = [int(x) for x in row["human_hand"].split()]
                theirs = [int(x) for x in row["bot_hand"].split()]
                score = {"win": 1.0, "draw": 0.5, "loss": 0.0}[row["result"]]
            except (KeyError, ValueError, AttributeError):
                skipped += 1
                continue
            if len(mine) != n_cards or len(theirs) != n_cards:
                skipped += 1
                continue
            scores.append(score)
            deltas.append(np.array(mine) - np.array(theirs))
    games = Games(tuple(c.name for c in config.cards), np.array(scores, dtype=float),
                  np.array(deltas, dtype=float).reshape(len(scores), n_cards))
    return HumanGames(games, skipped)


def games_needed(se: float, n: int, target_se: float = 0.02) -> int:
    """Games needed for a card's standard error to shrink to target_se (error shrinks with 1 / sqrt(n)).
    The default, a standard error of 2 points, is about +-4 points as a 95% interval."""
    return int(np.ceil(n * (se / target_se) ** 2))


def compare(human_report, bot_report) -> dict:
    """How the human-game effects line up with the bot-game effects (per card)."""
    h = np.array([e.effect for e in human_report.effects])
    b = np.array([e.effect for e in bot_report.effects])
    inside = sum(1 for he, be in zip(human_report.effects, bot_report.effects) if he.ci_low <= be.effect <= he.ci_high)
    corr = float(np.corrcoef(h, b)[0, 1]) if len(h) > 2 and h.std() > 0 and b.std() > 0 else float("nan")
    return {"correlation": corr, "bot_effect_inside_human_interval": inside, "cards": len(h)}


def format_comparison(human_report, bot_report, bot_name: str) -> str:
    lines = [
        f"Card power from {human_report.n_games} human games (win-probability change per extra dealt copy, "
        f"vs the {human_report.reference} card) next to the same measure for {bot_name} mirror games "
        f"({bot_report.n_games} games)",
        f"{'card':>10} {'humans':>9} {'95% interval':>18} {'bot':>8}  {'bot inside human interval?':<28}{'games for SE 2 pts':>18}",
    ]
    for he, be in sorted(zip(human_report.effects, bot_report.effects), key=lambda p: -p[1].effect):
        if he.se == 0:
            lines.append(f"{he.name:>10} {0.0:>+9.1%} {'(reference card)':>18} {be.effect:>+8.1%}")
            continue
        inside = "yes" if he.ci_low <= be.effect <= he.ci_high else "NO"
        lines.append(f"{he.name:>10} {he.effect:>+9.1%} {f'[{he.ci_low:+.1%}, {he.ci_high:+.1%}]':>18} {be.effect:>+8.1%}  "
                     f"{inside:<28}{games_needed(he.se, human_report.n_games):>18,}")
    c = compare(human_report, bot_report)
    lines.append(f"correlation of the human and bot effects across cards: {c['correlation']:.2f}; "
                 f"bot effect inside the human 95% interval for {c['bot_effect_inside_human_interval']} of {c['cards']} cards")
    widest = max(e.se for e in human_report.effects) * 1.96
    if widest > 0.05:
        lines.append(f"WARNING: with {human_report.n_games} games the widest interval is about +-{widest:.0%}; "
                     "differences smaller than that cannot be seen. More games are needed before drawing conclusions.")
    return "\n".join(lines)
