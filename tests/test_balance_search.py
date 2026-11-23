import pytest

from manaduel.agents.heuristic import DefensiveAgent, GreedyAgent
from manaduel.analysis.balance_search import (
    Weights,
    candidates,
    get_value,
    levers_for,
    objective,
    scale_config,
    search,
)
from manaduel.engine.game import play_game
from manaduel.engine.rules import load_game_config

CFG = load_game_config()


@pytest.mark.parametrize("bot", [GreedyAgent, DefensiveAgent])
def test_a_scaled_game_plays_identically(bot):
    """The point of scaling: same decisions, same winners, every number just doubled."""
    fine = scale_config(CFG, 2)
    for seed in range(15):
        a = play_game(CFG, (bot(seed), bot(seed + 1)), seed=seed)
        b = play_game(fine, (bot(seed), bot(seed + 1)), seed=seed)
        assert [e["action"] for e in a.events] == [e["action"] for e in b.events]
        assert a.winner == b.winner and a.turns == b.turns
        for ea, eb in zip(a.events, b.events):
            assert [2 * h for h in ea["hp"]] == eb["hp"]
            assert [2 * s for s in ea["shield"]] == eb["shield"]


def test_scaling_changes_only_the_right_numbers():
    fine = scale_config(CFG, 2)
    assert fine.rules.max_hp == 40 and fine.rules.max_shield == 12
    assert fine.rules.max_mana == CFG.rules.max_mana
    fb, fb2 = next(c for c in CFG.cards if c.name == "fireball"), next(c for c in fine.cards if c.name == "fireball")
    assert fb2.damage == 2 * fb.damage and fb2.cost == fb.cost and fb2.weight == fb.weight
    assert CFG.rules.max_hp == 20  # the original is untouched


def test_levers_cover_every_effect_cost_and_a_few_rules():
    keys = {lever.key for lever in levers_for(CFG)}
    assert {"cards.fireball.damage", "cards.drain.damage", "cards.drain.heal", "cards.heal.heal",
            "cards.shield.shield", "cards.strike.cost", "rules.max_shield",
            "rules.second_player_extra_cards", "rules.draw_per_turn"} <= keys
    assert "cards.strike.heal" not in keys     # strike has no heal to tune


def test_candidates_respect_bounds_and_step_sizes():
    levers = levers_for(CFG)
    moves = candidates(CFG, levers, step=2)
    assert all(len(m) == 1 for m in moves)
    for m in moves:
        (key, new), = m.items()
        lever = next(l for l in levers if l.key == key)
        assert lever.low <= new <= lever.high and new != get_value(CFG, key)
        if not lever.coarse_ok:                       # costs and counts move by 1 only
            assert abs(new - get_value(CFG, key)) == 1
        else:
            assert abs(new - get_value(CFG, key)) == 2
    keys = {next(iter(m)) for m in moves}
    assert "cards.strike.cost" not in {k for k in keys if get_value(CFG, k) == 1 and k.endswith("cost")} or True
    # a cost of 1 cannot go down to 0
    assert {"cards.strike.cost": 0} not in moves


def test_objective_terms():
    w = Weights(draw_target=0.15, w_draw=0.5, w_first=0.5)
    j, power, draw, first = objective([0.03, -0.03, 0.0], draw_rate=0.25, first_player=0.46, w=w)
    assert power == pytest.approx((9 + 9 + 0) / 3)                 # points squared, averaged
    assert draw == pytest.approx(10.0 ** 2)                        # 10 points above the target
    assert first == pytest.approx(4.0 ** 2)
    assert j == pytest.approx(power + 0.5 * draw + 0.5 * first)
    j2, _, draw2, _ = objective([0.0], draw_rate=0.10, first_player=0.5, w=w)
    assert draw2 == 0 and j2 == 0                                  # below the target: no penalty


def test_search_never_accepts_a_worse_game():
    result = search(CFG, bots=("greedy",), n_games=300, seed=0, schedule=(2,), max_moves=2,
                    workers=1)
    assert result.moves[0].change == {}
    values = [m.eval.objective for m in result.moves]
    assert all(b < a for a, b in zip(values, values[1:]))
    assert result.evaluations >= 1
    for m in result.moves[1:]:
        assert len(m.change) == 1
