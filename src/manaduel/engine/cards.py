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


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Card], Rules]:
    """Read the YAML file and return (cards by name, rules)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cards = {name: Card(name=name, **stats) for name, stats in raw["cards"].items()}
    rules = Rules(**raw["rules"])
    return cards, rules
