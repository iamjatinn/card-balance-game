import dataclasses

import pytest

from manaduel.agents.base import Agent
from manaduel.agents.random_agent import RandomAgent
from manaduel.agents.scripted import ScriptedAgent
from manaduel.engine.game import play_game
from manaduel.engine.rules import load_game_config, new_game
from manaduel.viewer.terminal import render_game

CFG = load_game_config()


def full_hand_start():
    """A start position where both players hold plenty of every card."""
    s = new_game(CFG)
    hand = (5,) * len(CFG.cards)
    players = tuple(dataclasses.replace(p, hand=hand) for p in s.players)
    return dataclasses.replace(s, players=players)


def test_the_nine_turn_example_game():
    """The exact game from the walkthrough: player 0 wins on turn 9."""
    p0 = ScriptedAgent([
        "strike", "end_turn",
        "strike", "strike", "end_turn",
        "fireball", "end_turn",
        "fireball", "strike", "end_turn",
        "fireball",
    ])
    p1 = ScriptedAgent([
        "shield", "end_turn",
        "heal", "end_turn",
        "fireball", "end_turn",
        "drain", "shield", "end_turn",
    ])
    result = play_game(CFG, (p0, p1), state=full_hand_start())
    assert result.winner == 0
    assert result.turns == 9
    last = result.events[-1]
    assert last["hp"] == [10, 0]


def test_random_games_always_finish_with_a_valid_result():
    for seed in range(200):
        result = play_game(CFG, (RandomAgent(seed), RandomAgent(seed + 1000)), seed=seed)
        assert result.winner in (0, 1, None)
        assert 1 <= result.turns <= CFG.rules.max_turns
        assert result.events


def test_same_seeds_give_identical_games():
    a = play_game(CFG, (RandomAgent(7), RandomAgent(8)), seed=5)
    b = play_game(CFG, (RandomAgent(7), RandomAgent(8)), seed=5)
    assert a == b


def test_result_records_the_dealt_hands():
    result = play_game(CFG, (RandomAgent(1), RandomAgent(2)), seed=9)
    dealt = new_game(CFG, 9)
    assert result.start_hands == [list(p.hand) for p in dealt.players]
    assert [sum(h) for h in result.start_hands] == [
        CFG.rules.start_hand,
        CFG.rules.start_hand + CFG.rules.second_player_extra_cards,
    ]


def test_different_game_seeds_give_different_games():
    games = [
        play_game(CFG, (RandomAgent(7), RandomAgent(8)), seed=s).events for s in range(5)
    ]
    assert len({str(g) for g in games}) > 1


def test_a_cheating_agent_is_caught():
    class Cheater(Agent):
        name = "cheater"

        def act(self, state, legal_actions, config):
            return 1  # fireball, which player 0 cannot afford on turn 1

    with pytest.raises(ValueError):
        play_game(CFG, (Cheater(), RandomAgent(0)))


def test_terminal_viewer_reports_the_winner():
    result = play_game(CFG, (RandomAgent(1), RandomAgent(2)))
    text = render_game(result, CFG, ("A", "B"))
    assert ("wins on turn" in text) or ("Draw after" in text)
