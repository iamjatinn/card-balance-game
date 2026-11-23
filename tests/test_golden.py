"""The default game must still play EXACTLY as it did in v0.1 (same cards drawn, same moves, same winners)."""

import json
from pathlib import Path

from make_golden import record

from manaduel.engine.rules import load_game_config

GOLDEN = Path(__file__).with_name("golden_v01.json")


def test_default_game_is_unchanged_since_v01():
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = record(load_game_config())
    assert len(actual) == len(expected) == 200
    mismatches = [(e["a"], e["b"], e["seed"]) for e, a in zip(expected, actual) if e != a]
    assert not mismatches, f"{len(mismatches)} games differ from v0.1, first: {mismatches[:3]}"
