"""Card and rule definitions, loaded from configs/cards.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

# This file is src/manaduel/engine/cards.py, so parents[3] is the project root.
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "cards.yaml"


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    damage: int = 0
    heal: int = 0
    shield: int = 0
    weight: int = 1   # how likely this card is to be drawn
    # Extra mechanics. All default to 0, which gives exactly the original five-card game.
    draw: int = 0     # draw this many cards when played
    poison: int = 0   # add this many poison counters to the opponent
    combo: int = 0    # extra damage if you already played another card this turn


@dataclass(frozen=True)
class Rules:
    max_hp: int
    max_mana: int
    max_turns: int
    max_shield: int = 6
    start_hand: int = 3
    max_hand: int = 6
    draw_per_turn: int = 1
    second_player_extra_cards: int = 0
    # Deck mode. 0 keeps the original behaviour (endless random draws). A positive
    # number gives each player a finite shuffled deck of that size, drawn without
    # replacement; a failed draw from an empty deck costs fatigue damage that grows
    # by `fatigue_damage` each time.
    deck_size: int = 0
    fatigue_damage: int = 1


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Card], Rules]:
    """Read the YAML file and return (cards by name, rules)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cards = {name: Card(name=name, **stats) for name, stats in raw["cards"].items()}
    rules = Rules(**raw["rules"])
    return cards, rules
