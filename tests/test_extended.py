"""Tests for the extra mechanics: draw, poison, combo, finite decks, played-card memory."""

import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest

from manaduel.agents.heuristic import GreedyAgent
from manaduel.agents.random_agent import RandomAgent
from manaduel.engine.decks import archetype_decks, random_deck
from manaduel.engine.game import play_game
from manaduel.engine.rules import (
    allocate,
    check_deck,
    default_deck,
    legal_actions,
    load_game_config,
    new_game,
    step,
)

ROOT = Path(__file__).resolve().parents[1]
EXT = load_game_config(ROOT / "configs" / "cards_extended.yaml")
DECKS = load_game_config(ROOT / "configs" / "cards_decks.yaml")
IDX = {c.name: i for i, c in enumerate(EXT.cards)}
END = EXT.end_turn


def with_player(state, i, **changes):
    players = list(state.players)
    players[i] = dataclasses.replace(players[i], **changes)
    return dataclasses.replace(state, players=tuple(players))


def stocked(config=EXT, copies=5, mana=5):
    """A new game where player 0 holds plenty of every card and has mana to spend."""
    s = new_game(config)
    n = len(config.cards)
    for i in (0, 1):
        s = with_player(s, i, hand=(copies,) * n)
    return with_player(s, 0, mana=mana)


# ---- draw ----------------------------------------------------------------

def test_draw_card_draws_cards():
    s = with_player(stocked(), 0, hand=(0, 0, 0, 0, 0, 1, 0, 0))   # just one Insight
    s = step(s, IDX["insight"], EXT)
    assert sum(s.players[0].hand) == 2          # played 1, drew 2
    assert s.players[0].mana == 5 - 2


def test_draw_into_a_full_hand_burns_cards():
    full = (1, 1, 1, 1, 0, 2, 0, 0)             # 6 cards = max_hand, including 2 insights
    s = with_player(stocked(), 0, hand=full)
    s = step(s, IDX["insight"], EXT)
    assert sum(s.players[0].hand) == EXT.rules.max_hand    # 5 left + room for only 1 of the 2 drawn


# ---- poison ----------------------------------------------------------------

def test_poison_ticks_at_the_start_of_the_victims_turn_and_fades():
    s = step(stocked(), IDX["venom"], EXT)
    assert s.players[1].poison == 3 and s.players[1].hp == 20     # nothing happens yet
    s = step(s, END, EXT)                                          # player 1's turn begins
    assert s.players[1].hp == 17 and s.players[1].poison == 2
    s = step(s, END, EXT)
    s = step(s, END, EXT)                                          # player 1 again
    assert s.players[1].hp == 15 and s.players[1].poison == 1
    s = step(s, END, EXT)
    s = step(s, END, EXT)
    assert s.players[1].hp == 14 and s.players[1].poison == 0      # 3 + 2 + 1 = 6 damage in total
    s = step(s, END, EXT)
    s = step(s, END, EXT)
    assert s.players[1].hp == 14                                   # it has faded


def test_poison_ignores_shield():
    s = with_player(step(stocked(), IDX["venom"], EXT), 1, shield=6)
    s = step(s, END, EXT)
    assert s.players[1].hp == 17 and s.players[1].shield == 0      # shield expired, HP still hurt


def test_poison_can_kill_and_ends_the_game():
    s = with_player(step(stocked(), IDX["venom"], EXT), 1, hp=3)
    s = step(s, END, EXT)
    assert s.players[1].hp == 0
    assert legal_actions(s, EXT) == []


def test_poison_counters_stack():
    s = step(stocked(), IDX["venom"], EXT)
    s = step(s, IDX["venom"], EXT)
    assert s.players[1].poison == 6


# ---- combo -----------------------------------------------------------------

def test_combo_needs_an_earlier_card_this_turn():
    alone = step(stocked(), IDX["flurry"], EXT)
    assert alone.players[1].hp == 20 - 2                           # no bonus when played first
    after_strike = step(step(stocked(), IDX["strike"], EXT), IDX["flurry"], EXT)
    assert after_strike.players[1].hp == 20 - 3 - (2 + 3)          # strike, then flurry with the bonus


def test_combo_counter_resets_each_turn():
    s = step(stocked(), IDX["strike"], EXT)
    s = step(s, END, EXT)
    s = step(s, END, EXT)                                          # back to player 0, new turn
    assert s.players[0].combo == 0
    s = with_player(s, 0, mana=5)
    s = step(s, IDX["flurry"], EXT)
    assert s.players[1].hp == 20 - 3 - 2        # the earlier strike (3) plus a flurry with NO bonus (2)


def test_combo_bonus_is_absorbed_by_shield_like_normal_damage():
    s = with_player(step(stocked(), IDX["strike"], EXT), 1, shield=4, hp=20)
    s = step(s, IDX["flurry"], EXT)                                # 5 damage vs 4 shield
    assert s.players[1].shield == 0 and s.players[1].hp == 19


# ---- memory (played counters) -----------------------------------------------

def test_played_counts_are_tracked():
    s = step(step(stocked(), IDX["strike"], EXT), IDX["strike"], EXT)
    assert s.players[0].played[IDX["strike"]] == 2
    assert sum(s.players[1].played) == 0


# ---- decks ------------------------------------------------------------------

def test_allocate_splits_in_proportion_and_respects_the_cap():
    assert sum(allocate([1, 1, 1], 20)) == 20
    assert allocate([3, 1], 8) == (6, 2)
    assert max(allocate([10, 1, 1], 12, max_copies=4)) == 4
    with pytest.raises(ValueError):
        allocate([1, 1], 10, max_copies=2)


def test_default_and_archetype_decks_are_valid():
    deck = default_deck(DECKS)
    assert sum(deck) == 20 and check_deck(DECKS, deck) == deck
    for name, d in archetype_decks(DECKS).items():
        assert sum(d) == 20 and max(d) <= 6, name
    assert len(set(archetype_decks(DECKS).values())) == 4          # four genuinely different decks


def test_aggro_has_more_damage_and_control_more_sustain():
    decks = archetype_decks(DECKS)
    names = [c.name for c in DECKS.cards]
    count = lambda d, *cards: sum(d[names.index(c)] for c in cards)
    assert count(decks["aggro"], "strike", "fireball", "flurry") > count(decks["control"], "strike", "fireball", "flurry")
    assert count(decks["control"], "heal", "shield", "insight") > count(decks["aggro"], "heal", "shield", "insight")


def test_random_decks_are_valid_and_reproducible():
    a = random_deck(DECKS, random.Random(1))
    assert a == random_deck(DECKS, random.Random(1))
    assert sum(a) == 20 and max(a) <= 6
    assert len({random_deck(DECKS, random.Random(i)) for i in range(10)}) > 5


def test_a_bad_deck_is_rejected():
    with pytest.raises(ValueError):
        new_game(DECKS, 0, ((5, 0, 0, 0, 0, 0, 0, 0), default_deck(DECKS)))      # wrong total
    with pytest.raises(ValueError):
        new_game(EXT, 0, (default_deck(DECKS), default_deck(DECKS)))             # not a deck game


def test_cards_come_out_of_the_deck_without_replacement():
    s = new_game(DECKS, 3)
    for p in s.players:
        assert sum(p.deck) + sum(p.hand) == 20                     # nothing created or lost
    start_cards = [tuple(d + h for d, h in zip(p.deck, p.hand)) for p in s.players]
    assert start_cards[0] == start_cards[1] == default_deck(DECKS)


def test_a_player_never_holds_more_copies_than_their_deck_has():
    greedy = GreedyAgent
    for seed in range(30):
        result = play_game(DECKS, (greedy(seed), greedy(seed + 1)), seed=seed)
        decks = result.decks
        for e in result.events:
            for p in (0, 1):
                assert all(h <= d for h, d in zip(e["hand"][p], decks[p]))
        assert result.decks and [sum(d) for d in result.decks] == [20, 20]


def test_fatigue_hurts_more_each_time_an_empty_deck_is_drawn_from():
    s = new_game(DECKS, 0)
    s = with_player(s, 1, deck=(0,) * len(DECKS.cards), hp=20)
    hp = []
    for _ in range(4):                                             # four of player 1's turns
        s = step(s, END, DECKS)
        hp.append(s.players[1].hp)
        s = step(s, END, DECKS)
    assert hp == [19, 17, 14, 10]               # 1, then 2, then 3, then 4 damage per failed draw
    assert s.players[1].fatigue == 4


def test_deck_games_finish_and_replay_exactly():
    for seed in range(40):
        a = play_game(DECKS, (RandomAgent(seed), GreedyAgent(seed)), seed=seed)
        b = play_game(DECKS, (RandomAgent(seed), GreedyAgent(seed)), seed=seed)
        assert a == b and a.winner in (0, 1, None)
        assert all("deck_left" in e for e in a.events)


def test_chosen_decks_change_the_game():
    decks = archetype_decks(DECKS)
    a = play_game(DECKS, (GreedyAgent(0), GreedyAgent(1)), seed=2, decks=(decks["aggro"], decks["control"]))
    b = play_game(DECKS, (GreedyAgent(0), GreedyAgent(1)), seed=2, decks=(decks["control"], decks["aggro"]))
    assert a.decks[0] == list(decks["aggro"]) and b.decks[0] == list(decks["control"])


# ---- the extended game plays sensibly ---------------------------------------

def test_extended_games_finish_with_every_bot():
    from manaduel.agents.registry import BOT_FACTORIES

    for first in BOT_FACTORIES.values():
        for second in BOT_FACTORIES.values():
            for seed in range(8):
                result = play_game(EXT, (first(seed), second(seed + 1)), seed=seed)
                assert result.winner in (0, 1, None)


def test_greedy_actually_uses_the_new_cards():
    used = set()
    for seed in range(30):
        result = play_game(EXT, (GreedyAgent(seed), GreedyAgent(seed + 1)), seed=seed)
        used |= {e["action"] for e in result.events}
    assert {"insight", "venom", "flurry"} <= used


# ---- observation ----------------------------------------------------------------

def test_observation_sizes():
    from manaduel.rl.obs import observation, observation_size

    base = load_game_config()
    assert observation_size(base) == 13                           # unchanged from v0.1
    assert observation_size(base, memory=True) == 13 + 5
    assert observation_size(EXT) == 3 + 8 + 3 + 1 + 1 + 2 + 1     # + poison pair + combo flag
    assert observation_size(DECKS) == observation_size(EXT) + 8 + 2
    for config in (base, EXT, DECKS):
        for memory in (False, True):
            s = new_game(config, 1)
            obs = observation(s, 0, config, memory)
            assert obs.shape == (observation_size(config, memory),)
            assert obs.min() >= 0 and obs.max() <= 1


def test_memory_features_reflect_what_the_opponent_played():
    from manaduel.rl.obs import observation

    s = step(stocked(), IDX["strike"], EXT)                       # player 0 plays a strike
    with_mem = observation(s, 1, EXT, memory=True)                # as seen by player 1
    without = observation(s, 1, EXT, memory=False)
    assert len(with_mem) == len(without) + len(EXT.cards)
    assert with_mem[-len(EXT.cards):][IDX["strike"]] == pytest.approx(0.1)
    assert np.all(with_mem[-len(EXT.cards):][np.arange(len(EXT.cards)) != IDX["strike"]] == 0)
    assert np.array_equal(with_mem[:len(without)], without)


def test_the_environment_works_for_the_extended_and_deck_games():
    pytest.importorskip("gymnasium")
    from manaduel.rl.env import ManaDuelEnv

    for config, kwargs in ((EXT, {}), (EXT, {"memory": True}),
                           (DECKS, {"memory": True, "decks": list(archetype_decks(DECKS).values())})):
        env = ManaDuelEnv(config, **kwargs)
        rng = np.random.default_rng(0)
        for episode in range(10):
            obs, _ = env.reset(seed=episode)
            assert obs.shape == env.observation_space.shape
            done = False
            while not done:
                legal = np.flatnonzero(env.action_masks())
                obs, reward, done, _, _ = env.step(int(rng.choice(legal)))
            assert reward in (-1.0, 0.0, 1.0)


def test_a_model_for_a_different_card_set_is_refused():
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    from manaduel.agents.rl_agent import PPOAgent
    from manaduel.rl.env import ManaDuelEnv

    base = load_game_config()
    env = ActionMasker(ManaDuelEnv(base), lambda e: e.unwrapped.action_masks())
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu", seed=0)
    agent = PPOAgent(model)
    s = new_game(EXT, 0)
    with pytest.raises(ValueError, match="different card set"):
        agent.act(s, legal_actions(s, EXT), EXT)
