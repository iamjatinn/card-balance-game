"""Balancing with learned agents in the loop.

Why this exists: `balance_search` tunes a card set with hand-written bots. On the extended card set that
produced a game the bots find balanced and trained PPO agents find badly broken (Flurry became +20 points):
the greedy-style bots barely use the combo bonus, so the search kept raising it, while a learned agent learns to
exploit it. Balance is relative to who is playing, so the final check, and the correction, should use learned
agents.

One round: train PPO agents on the current card set, measure each card's power with them (the same
mirror-game regression as everywhere else), then nudge every card that the agents agree is off:

  * a card's effect numbers (damage, heal, shield, poison, combo) are scaled by 1 - effect / `gain`, limited
    to +/- `max_change` per round (a +20 point card loses 40%; a -4 point card gains 16%), and always
    move by at least 1;
  * a card with no effect number to scale (Insight, which only draws) gets its cost raised if too strong and
    lowered if too weak;
  * a card is only touched if its mean effect over the agents is at least `min_effect` AND all agents agree on the
    sign, so noise does not move cards.

This is a simple, damped rule of thumb, not an optimiser. It uses a few rounds because each round needs new
training runs. The first-player advantage is not part of it.
"""

from dataclasses import dataclass, replace

import numpy as np

from manaduel.engine.rules import GameConfig

EFFECT_STATS = ("damage", "heal", "shield", "poison", "combo")


@dataclass(frozen=True)
class Change:
    card: str
    effect: float            # mean measured effect (fraction of win probability per copy)
    description: str         # e.g. "combo 16 -> 10"


def _scaled(value: int, factor: float) -> int:
    new = max(1, int(round(value * factor)))
    if new == value:         # always move, otherwise a small card could never be corrected
        new = value + (1 if factor > 1 else -1)
    return max(1, new)


def correct(config: GameConfig, effects_per_agent, names, gain: float = 0.25, max_change: float = 0.4,
            min_effect: float = 0.03) -> tuple[GameConfig, list[Change]]:
    """Apply one round of corrections. effects_per_agent has shape (agents, cards) in the order of `names`."""
    effects = np.asarray(effects_per_agent, dtype=float)
    mean = effects.mean(axis=0)
    cards = list(config.cards)
    changes: list[Change] = []
    for i, card in enumerate(config.cards):
        k = names.index(card.name)
        e = float(mean[k])
        consistent = bool(np.all(np.sign(effects[:, k]) == np.sign(e)))
        if abs(e) < min_effect or not consistent:
            continue
        factor = 1.0 - float(np.clip(e / gain, -max_change, max_change))
        new, notes = card, []
        for stat in EFFECT_STATS:
            old = getattr(card, stat)
            if old > 0:
                value = _scaled(old, factor)
                new = replace(new, **{stat: value})
                notes.append(f"{stat} {old} -> {value}")
        if not notes:   # a card that only draws: change what it costs instead
            cost = card.cost + (1 if e > 0 else -1)
            cost = min(max(cost, 1), config.rules.max_mana)
            if cost != card.cost:
                new = replace(new, cost=cost)
                notes.append(f"cost {card.cost} -> {cost}")
        if notes:
            cards[i] = new
            changes.append(Change(card.name, e, ", ".join(notes)))
    rules = config.rules
    top_shield = max((c.shield for c in cards), default=0)
    if top_shield > rules.max_shield:   # a shield card bigger than the cap would be partly wasted
        rules = replace(rules, max_shield=top_shield)
    return GameConfig(cards=tuple(cards), rules=rules), changes


def summarise(effects_per_agent, names, min_effect: float = 0.03) -> dict:
    """How far from balanced a card set is, from agents' measured effects."""
    effects = np.asarray(effects_per_agent, dtype=float)
    mean = effects.mean(axis=0)
    return {
        "mean_effect": dict(zip(names, map(float, mean))),
        "worst_card": names[int(np.argmax(np.abs(mean)))],
        "worst_abs": float(np.max(np.abs(mean))),
        "rms": float(np.sqrt(np.mean(mean ** 2))),
        "off_cards": [n for n, e in zip(names, mean) if abs(e) >= min_effect],
    }
