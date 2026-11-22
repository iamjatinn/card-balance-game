"""Sweep: try a game setting at several values and measure what changes.

Each value gets mirror matches (the same bot in both seats), so any shift in
the numbers comes from the rules, not from bot skill. Reported per bot:
  - first-mover win rate (share of decisive games won by whoever moved first;
    50% is fair), with a 95% interval
  - draw rate
  - average game length in turns

Settings are named like  rules.max_shield  or  cards.heal.heal  (card, then field).
"""

import csv
import itertools
import math
from dataclasses import dataclass, fields, replace

from manaduel.engine.cards import Card, Rules
from manaduel.engine.rules import GameConfig, check_config
from manaduel.sim.stats import wilson_interval
from manaduel.sim.tournament import play_match

MAX_COMBINATIONS = 60


def parse_set(spec: str) -> tuple[str, list[int]]:
    """'rules.max_shield=3,4,5' -> ('rules.max_shield', [3, 4, 5])"""
    if "=" not in spec:
        raise ValueError(f"expected KEY=V1,V2,... but got '{spec}'")
    key, values = spec.split("=", 1)
    try:
        parsed = [int(v) for v in values.split(",") if v.strip() != ""]
    except ValueError:
        raise ValueError(f"values for '{key}' must be whole numbers: '{values}'") from None
    if not parsed:
        raise ValueError(f"no values given for '{key}'")
    return key.strip(), parsed


def apply_overrides(config: GameConfig, overrides: dict[str, int]) -> GameConfig:
    """Return a modified copy of the config. The original is never changed."""
    rules = config.rules
    cards = list(config.cards)
    rule_fields = {f.name for f in fields(Rules)}
    card_fields = {f.name for f in fields(Card)} - {"name"}

    for key, value in overrides.items():
        parts = key.split(".")
        if parts[0] == "rules" and len(parts) == 2:
            if parts[1] not in rule_fields:
                raise ValueError(f"unknown rule '{parts[1]}'. choices: {', '.join(sorted(rule_fields))}")
            rules = replace(rules, **{parts[1]: value})
        elif parts[0] == "cards" and len(parts) == 3:
            names = [c.name for c in cards]
            if parts[1] not in names:
                raise ValueError(f"unknown card '{parts[1]}'. choices: {', '.join(names)}")
            if parts[2] not in card_fields:
                raise ValueError(f"unknown card field '{parts[2]}'. choices: {', '.join(sorted(card_fields))}")
            i = names.index(parts[1])
            cards[i] = replace(cards[i], **{parts[2]: value})
        else:
            raise ValueError(f"'{key}' is not a valid setting; use rules.max_shield or cards.heal.heal")

    modified = GameConfig(cards=tuple(cards), rules=rules)
    check_config(modified)
    return modified


@dataclass(frozen=True)
class SweepRow:
    overrides: dict
    bot: str
    games: int
    first_mover: float       # share of decisive games won by whoever moved first
    first_mover_low: float
    first_mover_high: float
    draw_rate: float
    avg_turns: float


def run_sweep(config: GameConfig, grid: dict[str, list[int]], factories: dict,
              n_games: int, seed: int = 0) -> list[SweepRow]:
    """Run every combination of the values in `grid` (a Cartesian product)."""
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    if len(combos) > MAX_COMBINATIONS:
        raise ValueError(f"{len(combos)} combinations is too many (limit {MAX_COMBINATIONS})")

    rows = []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        modified = apply_overrides(config, overrides)
        for name, factory in factories.items():
            r = play_match(modified, name, factory, name, factory, n_games, seed)
            decisive = r.wins_a + r.wins_b
            if decisive:
                rate = r.first_player_wins / decisive
                low, high = wilson_interval(r.first_player_wins, decisive)
            else:
                rate = low = high = math.nan
            rows.append(
                SweepRow(overrides, name, r.games, rate, low, high, r.draws / r.games, r.avg_turns)
            )
    return rows


def format_table(rows: list[SweepRow]) -> str:
    lines = [
        f"{'setting':<34} {'bot':>10} {'first mover wins':>22} {'draws':>7} {'turns':>6}",
    ]
    for r in rows:
        setting = ", ".join(f"{k}={v}" for k, v in r.overrides.items())
        interval = f"{r.first_mover:5.1%} [{r.first_mover_low:5.1%}, {r.first_mover_high:5.1%}]"
        lines.append(
            f"{setting:<34} {r.bot:>10} {interval:>22} {r.draw_rate:>7.1%} {r.avg_turns:>6.1f}"
        )
    lines.append("")
    lines.append("first mover wins = share of decisive games won by whoever moved first (50% is fair)")
    return "\n".join(lines)


def write_csv(rows: list[SweepRow], path) -> None:
    keys = list(rows[0].overrides) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*keys, "bot", "games", "first_mover", "first_mover_low",
                         "first_mover_high", "draw_rate", "avg_turns"])
        for r in rows:
            writer.writerow([
                *[r.overrides[k] for k in keys], r.bot, r.games,
                f"{r.first_mover:.4f}", f"{r.first_mover_low:.4f}", f"{r.first_mover_high:.4f}",
                f"{r.draw_rate:.4f}", f"{r.avg_turns:.2f}",
            ])
