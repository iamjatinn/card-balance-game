"""Deck building: archetype decks and random decks for deck mode.

A deck is a tuple with one count per card type (same order as the config), adding up
to `rules.deck_size`. Archetypes are built from card STATS rather than card names, so
they still make sense when the balance tools change a card.
"""

import random

from manaduel.engine.rules import GameConfig, allocate, check_deck

DEFAULT_MAX_COPIES = 6


def _aggro_score(card) -> float:
    """How much a card pushes damage per mana."""
    return (card.damage + card.combo + card.poison * (card.poison + 1) / 2) / card.cost


def _control_score(card) -> float:
    """How much a card sustains or gains resources per mana."""
    return (card.heal + card.shield + 2 * card.draw) / card.cost


def archetype_decks(config: GameConfig, max_copies: int = DEFAULT_MAX_COPIES) -> dict[str, tuple[int, ...]]:
    """Four recognisable decks:  aggro (damage), control (heal/shield/draw), burst (the most
    expensive hitters), and balanced (each card type about equally)."""
    size = config.rules.deck_size
    cards = config.cards
    floor = 0.05   # every card keeps a little weight so decks are not one-note
    aggro = [_aggro_score(c) + floor for c in cards]
    control = [_control_score(c) + floor for c in cards]
    burst = [(c.damage + c.combo + c.poison * 2) * c.cost + floor for c in cards]
    even = [1.0] * len(cards)
    decks = {
        "aggro": allocate(aggro, size, max_copies),
        "control": allocate(control, size, max_copies),
        "burst": allocate(burst, size, max_copies),
        "balanced": allocate(even, size, max_copies),
    }
    for deck in decks.values():
        check_deck(config, deck)
    return decks


def random_deck(config: GameConfig, rng: random.Random, max_copies: int = DEFAULT_MAX_COPIES) -> tuple[int, ...]:
    """A uniformly random deck: each card slot is a random card type, with at most `max_copies` of any."""
    size, n = config.rules.deck_size, len(config.cards)
    if max_copies * n < size:
        raise ValueError("max_copies is too small for this deck size")
    counts = [0] * n
    for _ in range(size):
        open_types = [i for i in range(n) if counts[i] < max_copies]
        counts[rng.choice(open_types)] += 1
    return tuple(counts)


def describe(config: GameConfig, deck) -> str:
    return " ".join(f"{c.name}x{n}" for c, n in zip(config.cards, deck) if n)
