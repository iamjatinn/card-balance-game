"""Save a finished game to a JSON file, and read it back.

The file contains the cards and rules too, so a viewer can draw the game
correctly even if configs/cards.yaml is edited later."""

import json
from dataclasses import asdict
from pathlib import Path


def save_replay(path, result, config, names, seed=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "agents": list(names),
        "seed": seed,
        "winner": result.winner,
        "turns": result.turns,
        "rules": asdict(config.rules),
        "cards": [asdict(card) for card in config.cards],
        "events": result.events,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_replay(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
