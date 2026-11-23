"""Records exactly how the v0.1 game plays, so later engine changes cannot silently alter it.

Run once, BEFORE changing the engine:   python tests/make_golden.py
It writes tests/golden_v01.json. tests/test_golden.py replays the same games and compares.
"""

import hashlib
import json
import sys
from pathlib import Path

from manaduel.agents.registry import BOT_FACTORIES
from manaduel.engine.game import play_game
from manaduel.engine.rules import load_game_config

PAIRS = [("greedy", "defensive"), ("defensive", "greedy"), ("random", "greedy"),
         ("aggressive", "defensive"), ("greedy", "greedy")]
SEEDS = range(40)


def digest(events) -> str:
    return hashlib.sha1(json.dumps(events, sort_keys=True).encode()).hexdigest()[:16]


def record(config):
    rows = []
    for a, b in PAIRS:
        for seed in SEEDS:
            result = play_game(config, (BOT_FACTORIES[a](seed), BOT_FACTORIES[b](seed + 1)), seed=seed)
            last = result.events[-1]
            rows.append({"a": a, "b": b, "seed": seed, "winner": result.winner, "turns": result.turns,
                         "events": len(result.events), "hp": last["hp"], "digest": digest(result.events)})
    return rows


if __name__ == "__main__":
    import manaduel

    print("using code from:", Path(manaduel.__file__).parent)
    rows = record(load_game_config())
    out = Path(__file__).with_name("golden_v01.json")
    out.write_text(json.dumps(rows, indent=0), encoding="utf-8")
    print(f"wrote {len(rows)} games to {out}")
