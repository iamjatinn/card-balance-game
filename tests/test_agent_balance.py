from pathlib import Path

import numpy as np

from manaduel.analysis.agent_balance import correct, summarise
from manaduel.engine.rules import load_game_config

CFG = load_game_config(Path("configs/cards_extended_balanced.yaml"))
NAMES = [c.name for c in CFG.cards]


def effects(**values):
    row = np.zeros(len(NAMES))
    for name, v in values.items():
        row[NAMES.index(name)] = v
    return np.array([row, row, row])      # three agents that agree


def card(config, name):
    return next(c for c in config.cards if c.name == name)


def test_a_card_that_agents_find_too_strong_is_weakened_by_a_bounded_amount():
    before = card(CFG, "flurry").combo
    new, changes = correct(CFG, effects(flurry=0.20), NAMES)
    after = card(new, "flurry").combo
    assert after < before and after >= round(before * 0.6) - 1       # at most about 40% in one round
    assert [c.card for c in changes] == ["flurry"]
    assert card(new, "flurry").damage < card(CFG, "flurry").damage        # every effect number of the card is scaled
    assert card(new, "flurry").cost == card(CFG, "flurry").cost            # costs are left alone for such cards
    assert card(CFG, "flurry").combo == before                           # the input is not modified


def test_a_weak_card_is_strengthened_and_always_moves_at_least_one_step():
    new, _ = correct(CFG, effects(heal=-0.04), NAMES)
    assert card(new, "heal").heal > card(CFG, "heal").heal


def test_small_or_disagreeing_effects_change_nothing():
    small, changes = correct(CFG, effects(strike=0.02, fireball=-0.025), NAMES)
    assert changes == [] and small.cards == CFG.cards
    row_a, row_b = np.zeros(len(NAMES)), np.zeros(len(NAMES))
    row_a[NAMES.index("venom")], row_b[NAMES.index("venom")] = -0.09, 0.01      # mean -4 but the agents disagree on the sign
    disagree, changes = correct(CFG, np.array([row_a, row_a, row_b]), NAMES)
    assert changes == []


def test_a_card_that_only_draws_changes_cost_in_the_right_direction():
    cost = card(CFG, "insight").cost
    too_strong, _ = correct(CFG, effects(insight=0.06), NAMES)
    too_weak, _ = correct(CFG, effects(insight=-0.06), NAMES)
    assert card(too_strong, "insight").cost == min(cost + 1, CFG.rules.max_mana)
    assert card(too_weak, "insight").cost == max(cost - 1, 1)


def test_raising_a_shield_card_above_the_cap_raises_the_cap():
    new, _ = correct(CFG, effects(shield=-0.20), NAMES)
    assert new.rules.max_shield >= card(new, "shield").shield


def test_summary_reports_the_worst_card_and_which_cards_are_off():
    s = summarise(effects(flurry=0.20, shield=-0.04, strike=0.01), NAMES)
    assert s["worst_card"] == "flurry" and abs(s["worst_abs"] - 0.20) < 1e-9
    assert set(s["off_cards"]) == {"flurry", "shield"}
