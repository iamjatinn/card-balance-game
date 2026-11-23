import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest

from manaduel.agents.heuristic import GreedyAgent
from manaduel.analysis.card_effects import estimate_effects
from manaduel.analysis.metagame import (
    DeckMatrix,
    beats,
    build_deck,
    collect_decks,
    count_cycles,
    deck_matrix,
    dominant_decks,
    equilibrium,
    exploitability,
    format_health,
    format_matrix,
    neighbours,
    pure_exploitability,
    write_matrix_csv,
)
from manaduel.engine.decks import archetype_decks
from manaduel.engine.rules import GameConfig, load_game_config

ROOT = Path(__file__).resolve().parents[1]
DECKS = load_game_config(ROOT / "configs" / "cards_decks.yaml")

RPS = np.array([[0.5, 1.0, 0.0],
                [0.0, 0.5, 1.0],
                [1.0, 0.0, 0.5]])          # deck 0 beats 1, 1 beats 2, 2 beats 0


def fake_matrix(win):
    n = len(win)
    win = np.asarray(win, dtype=float)
    low, high = win - 0.05, win + 0.05
    np.fill_diagonal(low, 0.5)
    np.fill_diagonal(high, 0.5)
    return DeckMatrix(tuple(f"d{i}" for i in range(n)), tuple((20,) + (0,) * 7 for _ in range(n)),
                      win, low, high, np.zeros((n, n)), 1000)


def test_equilibrium_of_rock_paper_scissors_is_uniform():
    mix = equilibrium(RPS)
    assert np.allclose(mix, 1 / 3, atol=0.02)
    assert exploitability(RPS, mix) < 0.02
    assert exploitability(RPS, np.array([1.0, 0, 0])) == pytest.approx(0.5)   # a pure deck is fully countered


def test_equilibrium_puts_everything_on_a_dominant_deck():
    win = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
    mix = equilibrium(win)
    assert mix[0] > 0.95
    assert exploitability(win, mix) < 0.02


def test_pure_exploitability_is_the_best_counter():
    win = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
    counters = pure_exploitability(win)
    assert list(counters) == pytest.approx([0.0, 0.2, 0.3], abs=1e-9)
    # deck 0 has no counter (the best "counter" is itself, a coin flip); deck 1 is beaten 70/30 by
    # deck 0; deck 2 is beaten 80/20 by deck 0


def test_cycles_and_dominance_are_detected():
    cyc = fake_matrix(RPS)
    assert len(count_cycles(beats(cyc))) == 1
    line = fake_matrix([[0.5, 0.8, 0.8], [0.2, 0.5, 0.8], [0.2, 0.2, 0.5]])
    assert count_cycles(beats(line)) == []
    assert dominant_decks(line) == ["d0"]
    assert dominant_decks(cyc) == []


def test_neighbours_swap_exactly_one_card():
    deck = (5, 5, 5, 5, 0, 0, 0, 0)
    near = neighbours(deck, max_copies=6)
    assert all(sum(d) == 20 and max(d) <= 6 and d != deck for d in near)
    assert len(set(near)) == len(near)
    assert (4, 6, 5, 5, 0, 0, 0, 0) in near and (4, 5, 5, 5, 1, 0, 0, 0) in near
    assert all(sum(abs(a - b) for a, b in zip(d, deck)) == 2 for d in near)
    assert not any(d[4] < 0 for d in near)


def test_deck_matrix_is_symmetric_and_reproducible():
    decks = archetype_decks(DECKS)
    a = deck_matrix(DECKS, decks, "greedy", 60, seed=1, workers=1)
    b = deck_matrix(DECKS, decks, "greedy", 60, seed=1, workers=1)
    assert np.array_equal(a.win, b.win)
    assert np.allclose(a.win + a.win.T, 1.0)
    assert np.all(np.diag(a.win) == 0.5)
    assert np.all((a.low <= a.win + 1e-9) & (a.win <= a.high + 1e-9))
    assert "aggro" in format_matrix(a) and "Metagame health" in format_health(a, DECKS)


def test_deck_matrix_parallel_matches_serial():
    decks = archetype_decks(DECKS)
    serial = deck_matrix(DECKS, decks, "greedy", 40, seed=2, workers=1)
    parallel = deck_matrix(DECKS, decks, "greedy", 40, seed=2, workers=2)
    assert np.array_equal(serial.win, parallel.win)


def test_matrix_csv(tmp_path):
    m = deck_matrix(DECKS, archetype_decks(DECKS), "greedy", 20, seed=0, workers=1)
    write_matrix_csv(m, tmp_path / "m.csv")
    assert (tmp_path / "m.csv").read_text(encoding="utf-8").count("\n") == 4 * 3 + 1


def overpowered(card="fireball", damage=30):
    cards = tuple(dataclasses.replace(c, damage=damage) if c.name == card else c for c in DECKS.cards)
    return GameConfig(cards=cards, rules=DECKS.rules)


def test_random_deck_games_have_zero_sum_card_differences_and_reproduce():
    a = collect_decks(DECKS, GreedyAgent, 40, seed=3)
    b = collect_decks(DECKS, GreedyAgent, 40, seed=3)
    assert a.delta.shape == (40, 8) and np.all(a.delta.sum(axis=1) == 0)
    assert np.array_equal(a.delta, b.delta) and np.array_equal(a.score0, b.score0)


def test_deck_context_effects_find_a_planted_overpowered_card():
    report = estimate_effects(collect_decks(overpowered(), GreedyAgent, 2500, seed=0))
    top = max(report.effects, key=lambda e: e.effect)
    assert top.name == "fireball" and top.flagged


def test_the_deck_builder_adds_the_overpowered_card():
    config = overpowered()
    names = [c.name for c in config.cards]
    decks = archetype_decks(config)
    start = list(decks["control"])
    start[names.index("fireball")] = 0
    start[names.index("heal")] += decks["control"][names.index("fireball")]
    field = [decks["balanced"], decks["aggro"]]
    steps = build_deck(config, tuple(start), field, "greedy", n_games=80, seed=0, max_rounds=1, workers=1)
    assert len(steps) == 2 and "+fireball" in steps[1].swap
    assert steps[1].score > steps[0].score
    assert sum(steps[1].deck) == 20 and max(steps[1].deck) <= 6
