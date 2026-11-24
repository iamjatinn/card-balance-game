import csv
from pathlib import Path

import numpy as np
import pytest

from manaduel.analysis import human_check
from manaduel.analysis.card_effects import estimate_effects
from manaduel.engine.rules import load_game_config
from manaduel.play import LOG_COLUMNS

CFG = load_game_config()


def write_log(path: Path, n: int, planted: dict, seed: int = 0, bot="greedy", config_name="cards.yaml"):
    """A log in the format `play` writes, with random deals and a planted effect on the result."""
    rng = np.random.default_rng(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLUMNS)
        for g in range(n):
            mine = rng.multinomial(3, np.ones(5) / 5)
            theirs = rng.multinomial(3, np.ones(5) / 5)
            p = 0.5 + sum(planted.get(c.name, 0.0) * (mine[i] - theirs[i]) for i, c in enumerate(CFG.cards))
            outcome = "win" if rng.random() < p else "loss"
            w.writerow(["2026-01-01", config_name, bot, g % 2, g, outcome, 10,
                        " ".join(map(str, mine)), " ".join(map(str, theirs))])


def test_the_log_records_the_dealt_hands(tmp_path, monkeypatch):
    from manaduel import play
    from manaduel.agents.random_agent import RandomAgent

    monkeypatch.chdir(tmp_path)
    inputs = iter(["0"] * 200)       # always end the turn
    result = play.play_human(CFG, RandomAgent(1), 0, 1, bot_name="random", read=lambda prompt: next(inputs),
                             write=lambda text: None)
    play.record(result, CFG, 0, "random", 1, "cards.yaml", str(tmp_path / "log.csv"), str(tmp_path / "rp"))
    row = next(csv.DictReader(open(tmp_path / "log.csv", encoding="utf-8")))
    assert [int(x) for x in row["human_hand"].split()] == result.start_hands[0]
    assert [int(x) for x in row["bot_hand"].split()] == result.start_hands[1]


def test_a_planted_card_effect_is_recovered_from_human_games(tmp_path):
    write_log(tmp_path / "log.csv", 6000, {"fireball": 0.06, "shield": -0.04})
    loaded = human_check.load_human_games(tmp_path / "log.csv", CFG, "cards.yaml")
    assert loaded.skipped == 0 and len(loaded.games.score0) == 6000
    report = estimate_effects(loaded.games)
    effects = {e.name: e for e in report.effects}
    # effects are measured against the median card, so compare differences
    diff = effects["fireball"].effect - effects["shield"].effect
    assert diff == pytest.approx(0.10, abs=0.04)
    assert effects["fireball"].effect > effects["strike"].effect


def test_rows_from_another_card_set_bot_or_without_a_deal_are_skipped(tmp_path):
    write_log(tmp_path / "log.csv", 20, {}, bot="greedy")
    with open(tmp_path / "log.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["2026-01-01", "cards.yaml", "random", 0, 99, "win", 9, "", ""])   # no deal logged
    only_greedy = human_check.load_human_games(tmp_path / "log.csv", CFG, "cards.yaml", "greedy")
    assert len(only_greedy.games.score0) == 20 and only_greedy.skipped == 1
    other_set = human_check.load_human_games(tmp_path / "log.csv", CFG, "cards_extended.yaml")
    assert len(other_set.games.score0) == 0 and other_set.skipped == 21


def test_a_missing_log_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="play some games"):
        human_check.load_human_games(tmp_path / "nope.csv", CFG)


def test_few_games_produce_a_warning_and_a_games_needed_estimate(tmp_path):
    write_log(tmp_path / "log.csv", 120, {"fireball": 0.05})
    human = estimate_effects(human_check.load_human_games(tmp_path / "log.csv", CFG).games)
    bots = estimate_effects(human_check.load_human_games(tmp_path / "log.csv", CFG).games)   # any report will do
    text = human_check.format_comparison(human, bots, "greedy")
    assert "WARNING" in text and "games for SE 2 pts" in text
    assert human_check.games_needed(0.04, 100, 0.02) == 400        # half the error needs four times the games
