import dataclasses

import pytest

from manaduel.engine.rules import legal_actions, load_game_config, new_game, step

CFG = load_game_config()
IDX = {card.name: i for i, card in enumerate(CFG.cards)}
END = CFG.end_turn
N = len(CFG.cards)


def with_player(state, i, **changes):
    """Copy of state with player i's fields changed (states are frozen)."""
    players = list(state.players)
    players[i] = dataclasses.replace(players[i], **changes)
    return dataclasses.replace(state, players=tuple(players))


def fresh(copies=5):
    """A new game where both players hold plenty of every card, so a test about
    one rule does not depend on which cards happened to be drawn."""
    s = new_game(CFG)
    for i in (0, 1):
        s = with_player(s, i, hand=(copies,) * N)
    return s


# ---- setup and hands -------------------------------------------------------

def test_new_game_gives_player_zero_one_mana():
    s = new_game(CFG)
    assert s.players[0].mana == 1
    assert s.players[1].mana == 0
    assert s.current == 0 and s.turn == 1


def test_starting_hands_and_second_player_compensation():
    s = new_game(CFG)
    assert sum(s.players[0].hand) == CFG.rules.start_hand
    assert sum(s.players[1].hand) == CFG.rules.start_hand + CFG.rules.second_player_extra_cards


def test_same_seed_same_deal_and_different_seeds_differ():
    assert new_game(CFG, seed=5) == new_game(CFG, seed=5)
    hands = {new_game(CFG, seed=s).players[0].hand for s in range(10)}
    assert len(hands) > 1


def test_playing_a_card_uses_it_up():
    s = step(fresh(), IDX["strike"], CFG)
    assert s.players[0].hand[IDX["strike"]] == 4


def test_cannot_play_a_card_that_is_not_in_hand():
    hand = [5] * N
    hand[IDX["strike"]] = 0
    s = with_player(fresh(), 0, hand=tuple(hand))
    assert IDX["strike"] not in legal_actions(s, CFG)
    with pytest.raises(ValueError):
        step(s, IDX["strike"], CFG)


def test_player_draws_at_start_of_their_turn():
    s = new_game(CFG)
    before = sum(s.players[1].hand)
    s = step(s, END, CFG)  # turn 2: player 1 begins and draws
    assert sum(s.players[1].hand) == before + CFG.rules.draw_per_turn


def test_extra_cards_are_burned_at_max_hand():
    s = with_player(new_game(CFG), 1, hand=(2, 1, 1, 1, 1))  # 6 cards = max_hand
    assert sum(s.players[1].hand) == CFG.rules.max_hand
    s = step(s, END, CFG)
    assert sum(s.players[1].hand) == CFG.rules.max_hand


# ---- cards -----------------------------------------------------------------

def test_legal_actions_with_one_mana():
    assert legal_actions(fresh(), CFG) == [IDX["strike"], IDX["shield"], END]


def test_strike_damages_opponent_and_spends_mana():
    s = step(fresh(), IDX["strike"], CFG)
    assert s.players[1].hp == 17
    assert s.players[0].mana == 0


def test_shield_absorbs_damage_before_hp():
    s = with_player(fresh(), 1, shield=3)
    s = step(s, IDX["strike"], CFG)
    assert s.players[1].shield == 0
    assert s.players[1].hp == 20


def test_damage_beyond_shield_goes_to_hp():
    s = with_player(fresh(), 1, shield=3)
    s = with_player(s, 0, mana=3)
    s = step(s, IDX["fireball"], CFG)
    assert s.players[1].shield == 0
    assert s.players[1].hp == 16  # 7 damage - 3 absorbed = 4


def test_shield_is_capped():
    s = with_player(fresh(), 0, shield=5, mana=1)
    s = step(s, IDX["shield"], CFG)
    assert s.players[0].shield == CFG.rules.max_shield


def test_heal_is_capped_at_max_hp():
    s = with_player(fresh(), 0, hp=18, mana=2)
    s = step(s, IDX["heal"], CFG)
    assert s.players[0].hp == 20


def test_drain_damages_and_heals():
    s = with_player(fresh(), 0, hp=10, mana=3)
    s = step(s, IDX["drain"], CFG)
    assert s.players[1].hp == 17
    assert s.players[0].hp == 13


def test_cannot_play_a_card_you_cannot_afford():
    with pytest.raises(ValueError):
        step(fresh(), IDX["fireball"], CFG)  # costs 3, you have 1


def test_step_does_not_modify_the_old_state():
    s = fresh()
    step(s, IDX["strike"], CFG)
    assert s.players[1].hp == 20
    assert s.players[0].hand[IDX["strike"]] == 5


# ---- turns -----------------------------------------------------------------

def test_end_turn_switches_player_and_refills_mana():
    s = step(new_game(CFG), END, CFG)
    assert s.current == 1 and s.turn == 2
    assert s.players[1].mana == 1


def test_mana_grows_each_own_turn_and_is_capped():
    s = new_game(CFG)
    seen = []
    for _ in range(12):
        seen.append(s.players[s.current].mana)
        s = step(s, END, CFG)
    # own-turn numbers 1,1,2,2,3,3,4,4,5,5,6,6 -> capped at 5
    assert seen == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5]


def test_shield_expires_at_start_of_owners_next_turn():
    s = step(fresh(), END, CFG)                # turn 2: player 1 to move
    s = step(s, IDX["shield"], CFG)            # player 1 gains 3 shield
    s = step(s, END, CFG)                      # turn 3: player 0's turn
    assert s.players[1].shield == 3            # still up during the opponent's turn
    s = step(s, END, CFG)                      # turn 4: player 1's turn begins
    assert s.players[1].shield == 0            # now it has expired


def test_no_legal_actions_after_someone_dies():
    s = with_player(new_game(CFG), 1, hp=0)
    assert legal_actions(s, CFG) == []
