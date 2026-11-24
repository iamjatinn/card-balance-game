import dataclasses
from pathlib import Path

import pytest

from manaduel.agents.mcts import MCTSAgent
from manaduel.agents.random_agent import RandomAgent
from manaduel.engine.game import play_game
from manaduel.engine.rules import legal_actions, load_game_config, new_game, step
from manaduel.engine.state import GameState

CFG = load_game_config()
CARD = {c.name: i for i, c in enumerate(CFG.cards)}


def _state(my_hand, opp_hand, opp_hp=20, mana=5, rng=1):
    base = new_game(CFG, seed=3)
    me, opp = base.players
    me = dataclasses.replace(me, hand=tuple(my_hand), mana=mana)
    opp = dataclasses.replace(opp, hand=tuple(opp_hand), hp=opp_hp)
    return GameState(players=(me, opp), current=0, turn=3, rng=rng)


def hand(**counts):
    h = [0] * len(CFG.cards)
    for name, n in counts.items():
        h[CARD[name]] = n
    return h


def test_it_always_returns_a_legal_move_and_completes_games():
    for seed in range(3):
        a = MCTSAgent(seed=seed, iterations=20)
        r = play_game(CFG, (a, RandomAgent(seed)), seed=seed)
        assert r.turns <= CFG.rules.max_turns          # play_game raises if a move was illegal


def test_it_takes_a_win_that_is_on_the_table():
    state = _state(hand(fireball=1, heal=1), hand(strike=3), opp_hp=6, mana=3)
    legal = legal_actions(state, CFG)
    assert MCTSAgent(seed=0, iterations=150).act(state, legal, CFG) == CARD["fireball"]


def test_it_does_not_see_the_opponents_hand_or_the_next_draw():
    """The honesty test: change the opponent's hand to a different hand of the SAME size (and the game's
    hidden random-number state): the search must decide exactly the same."""
    a = _state(hand(strike=2, shield=1, fireball=1), hand(strike=3, heal=1), rng=11)
    b = _state(hand(strike=2, shield=1, fireball=1), hand(fireball=2, drain=2), rng=987654)
    assert sum(a.players[1].hand) == sum(b.players[1].hand)
    for seed in range(4):
        la, lb = legal_actions(a, CFG), legal_actions(b, CFG)
        assert MCTSAgent(seed=seed, iterations=60).act(a, la, CFG) == MCTSAgent(seed=seed, iterations=60).act(b, lb, CFG)


def test_the_same_seed_gives_the_same_game():
    r1 = play_game(CFG, (MCTSAgent(seed=5, iterations=25), RandomAgent(1)), seed=9)
    r2 = play_game(CFG, (MCTSAgent(seed=5, iterations=25), RandomAgent(1)), seed=9)
    assert r1.events == r2.events


def test_it_beats_a_random_player_most_of_the_time():
    wins = 0
    for g in range(10):
        a, b = MCTSAgent(seed=g, iterations=60), RandomAgent(g)
        agents = (a, b) if g % 2 == 0 else (b, a)
        r = play_game(CFG, agents, seed=100 + g // 2)
        wins += r.winner is not None and (r.winner == 0) == (g % 2 == 0)
    assert wins >= 7


def test_the_deck_mode_is_refused_not_silently_wrong():
    deck_cfg = load_game_config(Path("configs/cards_decks.yaml"))
    state = new_game(deck_cfg, seed=1)
    with pytest.raises(NotImplementedError):
        MCTSAgent(iterations=5).act(state, legal_actions(state, deck_cfg), deck_cfg)


def test_a_bad_rollout_name_is_rejected():
    with pytest.raises(ValueError):
        MCTSAgent(rollout="clever")


def test_the_extended_card_set_works_too():
    ext = load_game_config(Path("configs/cards_extended.yaml"))
    r = play_game(ext, (MCTSAgent(seed=1, iterations=15), RandomAgent(2)), seed=4)
    assert r.turns >= 1
    _ = step  # the engine's step is what the search uses; imported to make that dependency visible
