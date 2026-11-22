import dataclasses

import pytest

from manaduel.engine.state import (
    GameState,
    PlayerState,
    initial_state,
    is_terminal,
    winner,
)


def test_initial_state_values():
    s = initial_state()
    assert s.players[0].hp == 20 and s.players[1].hp == 20
    assert s.players[0].shield == 0 and s.players[0].mana == 0
    assert s.current == 0
    assert s.turn == 1


def test_fresh_game_is_not_over():
    s = initial_state()
    assert not is_terminal(s)
    assert winner(s) is None


def test_player_one_wins_when_player_zero_hp_hits_zero():
    s = initial_state()
    dead = dataclasses.replace(s, players=(PlayerState(hp=0), s.players[1]))
    assert is_terminal(dead)
    assert winner(dead) == 1


def test_player_zero_wins_when_player_one_hp_negative():
    s = initial_state()
    dead = dataclasses.replace(s, players=(s.players[0], PlayerState(hp=-3)))
    assert is_terminal(dead)
    assert winner(dead) == 0


def test_going_past_max_turns_is_a_draw():
    s = dataclasses.replace(initial_state(), turn=41)
    assert is_terminal(s, max_turns=40)
    assert winner(s, max_turns=40) is None


def test_state_is_immutable():
    s = initial_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.turn = 5
