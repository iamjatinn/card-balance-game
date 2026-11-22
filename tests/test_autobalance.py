import dataclasses

from manaduel.agents.heuristic import GreedyAgent
from manaduel.analysis.autobalance import (
    Change,
    autobalance,
    candidate_changes,
    config_to_yaml,
    format_trajectory,
    main_stat,
)
from manaduel.engine.rules import GameConfig, load_game_config

CFG = load_game_config()


def card(name, config=CFG):
    return next(c for c in config.cards if c.name == name)


def test_main_stat_is_the_headline_effect():
    assert main_stat(card("fireball")) == "damage"
    assert main_stat(card("heal")) == "heal"
    assert main_stat(card("shield")) == "shield"
    assert main_stat(card("drain")) == "damage"


def test_fixes_for_a_too_strong_card_weaken_it():
    changes = candidate_changes(CFG, "fireball", too_strong=True)
    assert Change("fireball", "damage", 7, 6) in changes
    assert Change("fireball", "cost", 3, 4) in changes


def test_fixes_for_a_too_weak_card_strengthen_it():
    changes = candidate_changes(CFG, "shield", too_strong=False)
    assert Change("shield", "shield", 3, 4) in changes
    # shield already costs 1 mana, the minimum, so it cannot get cheaper
    assert not any(c.field == "cost" for c in changes)


def test_fixes_never_leave_the_playable_range():
    capped = candidate_changes(CFG, "fireball", too_strong=True)
    assert all(1 <= c.new <= CFG.rules.max_mana for c in capped if c.field == "cost")
    tiny = dataclasses.replace(card("strike"), damage=1)
    config = GameConfig(cards=tuple(tiny if c.name == "strike" else c for c in CFG.cards), rules=CFG.rules)
    assert not any(c.field == "damage" for c in candidate_changes(config, "strike", too_strong=True))


def test_autobalance_improves_a_game_with_a_planted_overpowered_card():
    cards = tuple(
        dataclasses.replace(c, damage=9) if c.name == "fireball" else c for c in CFG.cards
    )
    planted = GameConfig(cards=cards, rules=CFG.rules)
    result = autobalance(planted, GreedyAgent, n_games=1500, seed=0, max_steps=2)

    assert result.steps[0].changes == ()
    assert len(result.steps) >= 2                                  # at least one fix was accepted
    first = result.steps[1].changes
    assert any(c.card == "fireball" for c in first)                # it targeted the planted card
    objectives = [s.objective for s in result.steps]
    assert all(b < a for a, b in zip(objectives, objectives[1:]))  # every accepted step helped
    assert "fireball" in format_trajectory(result)


def test_moves_include_pairs_on_different_cards_only():
    from manaduel.analysis.autobalance import _moves
    from manaduel.analysis.card_effects import CardEffect

    flagged = [
        CardEffect("fireball", 0.09, 0.005, 0.08, 0.10, 18.0, True),
        CardEffect("shield", -0.06, 0.005, -0.07, -0.05, -12.0, True),
    ]
    moves = _moves(CFG, flagged)
    singles = [m for m in moves if len(m) == 1]
    pairs = [m for m in moves if len(m) == 2]
    assert singles and pairs
    assert all(a.card != b.card for a, b in pairs)
    assert len(pairs) == len(candidate_changes(CFG, "fireball", True)) * len(
        candidate_changes(CFG, "shield", False)
    )


def test_changes_that_cause_more_draws_are_rejected():
    cards = tuple(
        dataclasses.replace(c, damage=9) if c.name == "fireball" else c for c in CFG.cards
    )
    planted = GameConfig(cards=cards, rules=CFG.rules)
    # An impossible limit (draws must DROP by 100 points) rejects every candidate.
    result = autobalance(planted, GreedyAgent, n_games=300, seed=0, max_steps=3,
                         max_draw_increase=-1.0)
    assert len(result.steps) == 1
    assert result.stop_reason.startswith("no change")


def test_config_round_trips_through_yaml(tmp_path):
    from manaduel.engine.rules import load_game_config as load

    cards = tuple(
        dataclasses.replace(c, damage=5, weight=2) if c.name == "fireball" else c for c in CFG.cards
    )
    modified = GameConfig(cards=cards, rules=dataclasses.replace(CFG.rules, max_shield=4))
    path = tmp_path / "balanced.yaml"
    config_to_yaml(modified, path)
    assert load(path) == modified
