import csv

import pytest

from manaduel.agents.heuristic import GreedyAgent
from manaduel.analysis.sweep import apply_overrides, format_table, parse_set, run_sweep, write_csv
from manaduel.engine.rules import load_game_config

CFG = load_game_config()


def test_parse_set():
    assert parse_set("rules.max_shield=3,4,5") == ("rules.max_shield", [3, 4, 5])
    assert parse_set("rules.second_player_extra_cards=1,0,-1")[1] == [1, 0, -1]


@pytest.mark.parametrize("bad", ["rules.max_shield", "rules.max_shield=", "rules.max_shield=a,b"])
def test_parse_set_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_set(bad)


def test_apply_overrides_changes_a_copy_only():
    modified = apply_overrides(CFG, {"rules.max_shield": 3, "cards.heal.heal": 2})
    assert modified.rules.max_shield == 3
    heal = next(c for c in modified.cards if c.name == "heal")
    assert heal.heal == 2
    assert CFG.rules.max_shield == 6                      # original untouched
    assert next(c for c in CFG.cards if c.name == "heal").heal == 4


@pytest.mark.parametrize("key,value", [
    ("rules.nonsense", 1),
    ("cards.nosuchcard.damage", 1),
    ("cards.heal.nonsense", 1),
    ("max_shield", 1),
    ("cards.heal.cost", 0),          # a free card would never end a turn
    ("rules.max_shield", 0),
])
def test_apply_overrides_rejects_invalid_settings(key, value):
    with pytest.raises(ValueError):
        apply_overrides(CFG, {key: value})


def test_grid_produces_one_row_per_combination_and_bot():
    grid = {"rules.max_shield": [3, 6], "rules.max_hand": [5, 6]}
    rows = run_sweep(CFG, grid, {"greedy": GreedyAgent}, n_games=40, seed=1)
    assert len(rows) == 4
    assert {tuple(r.overrides.items()) for r in rows} == {
        (("rules.max_shield", s), ("rules.max_hand", h)) for s in (3, 6) for h in (5, 6)
    }


def test_too_many_combinations_is_rejected():
    grid = {"rules.max_hand": list(range(1, 9)), "rules.max_shield": list(range(1, 9))}
    with pytest.raises(ValueError):
        run_sweep(CFG, grid, {"greedy": GreedyAgent}, n_games=2)


def test_sweep_reproduces_the_hand_run_compensation_result():
    """Regression check against the earlier manual experiment: giving the second
    player fewer cards must push the first mover's win rate up, clearly."""
    grid = {"rules.second_player_extra_cards": [1, 0, -1]}
    rows = run_sweep(CFG, grid, {"greedy": GreedyAgent}, n_games=1500, seed=0)
    by_extra = {r.overrides["rules.second_player_extra_cards"]: r.first_mover for r in rows}
    assert by_extra[1] + 0.03 < by_extra[0] < by_extra[-1] - 0.03


def test_table_and_csv_output(tmp_path):
    rows = run_sweep(CFG, {"rules.max_shield": [4, 6]}, {"greedy": GreedyAgent}, 40, seed=2)
    text = format_table(rows)
    assert "rules.max_shield=4" in text and "greedy" in text
    path = tmp_path / "sweep.csv"
    write_csv(rows, path)
    with open(path, encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert len(read) == 2 and read[0]["bot"] == "greedy"
