import dataclasses

import numpy as np

from manaduel.agents.heuristic import GreedyAgent
from manaduel.analysis.card_effects import Games, collect, estimate_effects, format_report
from manaduel.engine.rules import GameConfig, load_game_config

CFG = load_game_config()


def synthetic_games(n, true_effect, seat_advantage, seed):
    """Fake games where we KNOW how much each card is worth."""
    rng = np.random.default_rng(seed)
    k = len(true_effect)
    hands0 = rng.multinomial(3, [1 / k] * k, size=n)
    hands1 = rng.multinomial(3, [1 / k] * k, size=n)
    delta = (hands0 - hands1).astype(float)
    p = np.clip(0.5 + seat_advantage + delta @ np.asarray(true_effect), 0.02, 0.98)
    score0 = (rng.random(n) < p).astype(float)
    return Games(tuple(f"c{i}" for i in range(k)), score0, delta)


def test_recovers_a_planted_effect_and_flags_only_that_card():
    true_effect = np.array([0.0, 0.05, 0.0, 0.0, 0.0])
    games = synthetic_games(60_000, true_effect, seat_advantage=0.03, seed=1)
    report = estimate_effects(games, min_effect=0.02)

    truth = true_effect - np.median(true_effect)  # effects are relative to the median card
    for e, t in zip(report.effects, truth):
        assert abs(e.effect - t) <= 4 * e.se + 1e-9, (e.name, e.effect, t, e.se)
    assert [e.name for e in report.flagged()] == ["c1"]
    assert abs(report.seat0_baseline - 0.53) < 4 * report.seat0_baseline_se


def test_a_game_with_no_card_differences_flags_nothing():
    games = synthetic_games(60_000, np.zeros(5), seat_advantage=0.0, seed=2)
    report = estimate_effects(games)
    assert report.flagged() == []
    assert all(abs(e.effect) < 0.01 for e in report.effects)
    assert abs(report.seat0_baseline - 0.5) < 4 * report.seat0_baseline_se


def test_effects_average_to_zero_when_measured_against_the_mean():
    games = synthetic_games(5_000, np.array([0.02, -0.01, 0.03, 0.0, 0.01]), 0.0, seed=3)
    report = estimate_effects(games, reference="mean")
    assert abs(sum(e.effect for e in report.effects)) < 1e-9


def test_one_overpowered_card_does_not_make_the_others_look_weak():
    games = synthetic_games(60_000, np.array([0.0, 0.0, 0.0, 0.0, 0.15]), 0.0, seed=5)
    by_median = estimate_effects(games, reference="median")
    by_mean = estimate_effects(games, reference="mean")
    others_median = [e.effect for e in by_median.effects[:4]]
    others_mean = [e.effect for e in by_mean.effects[:4]]
    assert all(abs(x) < 0.01 for x in others_median)   # unaffected by the outlier
    assert all(x < -0.02 for x in others_mean)         # the mean is dragged up by it
    assert [e.name for e in by_median.flagged()] == ["c4"]


def test_unknown_reference_is_rejected():
    import pytest

    games = synthetic_games(100, np.zeros(5), 0.0, seed=6)
    with pytest.raises(ValueError):
        estimate_effects(games, reference="mode")


def test_collect_shapes_and_reproducibility():
    a = collect(CFG, GreedyAgent, 30, seed=5)
    b = collect(CFG, GreedyAgent, 30, seed=5)
    assert a.score0.shape == (30,) and a.delta.shape == (30, len(CFG.cards))
    assert set(np.unique(a.score0)) <= {0.0, 0.5, 1.0}
    assert np.all(a.delta.sum(axis=1) == 0)  # equal-size hands
    assert np.array_equal(a.score0, b.score0) and np.array_equal(a.delta, b.delta)


def test_finds_a_deliberately_overpowered_card_in_the_real_game():
    cards = tuple(
        dataclasses.replace(c, damage=30) if c.name == "fireball" else c for c in CFG.cards
    )
    broken = GameConfig(cards=cards, rules=CFG.rules)
    report = estimate_effects(collect(broken, GreedyAgent, 2000, seed=0))
    top = max(report.effects, key=lambda e: e.effect)
    assert top.name == "fireball"
    assert top.flagged


def test_report_text_mentions_every_card():
    games = synthetic_games(2_000, np.zeros(5), 0.0, seed=4)
    text = format_report(estimate_effects(games))
    for name in games.card_names:
        assert name in text
