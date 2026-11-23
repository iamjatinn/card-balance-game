"""Turn a game state into the list of numbers a neural network can read.

The network never sees the opponent's actual cards, only how many they hold:
a real player cannot see them either.

The base observation is the same as in v0.1 (so old trained models still load for the
original game). Extra numbers are added only when the game needs them:
  * poison counters (when any card uses poison)
  * whether a card was already played this turn (when any card has a combo)
  * the player's own remaining deck, and the opponent's deck size (deck mode)
  * with `memory=True`: how many copies of each card the OPPONENT has played so far.
    That is public information in a real game; this is the "opponent-card memory" feature.
"""

import numpy as np

MEMORY_SCALE = 10.0   # played counts are divided by this and clipped to 1


def _extra_size(config, memory: bool) -> int:
    n = len(config.cards)
    size = 0
    if config.uses_poison:
        size += 2
    if any(c.combo > 0 for c in config.cards):
        size += 1
    if config.uses_decks:
        size += n + 2
    if memory:
        size += n
    return size


def observation_size(config, memory: bool = False) -> int:
    # my hp, shield, mana | my hand (one number per card type)
    # | opponent hp, shield, hand size | turn | am I the first player
    return 3 + len(config.cards) + 3 + 1 + 1 + _extra_size(config, memory)


def observation(state, player: int, config, memory: bool = False) -> np.ndarray:
    """The game as seen by `player`, scaled so every number is between 0 and 1."""
    r = config.rules
    n = len(config.cards)
    me = state.players[player]
    opp = state.players[1 - player]
    hand = list(me.hand) if me.hand else [0] * n
    values = [
        me.hp / r.max_hp,
        me.shield / r.max_shield,
        me.mana / r.max_mana,
        *[count / r.max_hand for count in hand],
        opp.hp / r.max_hp,
        opp.shield / r.max_shield,
        sum(opp.hand) / r.max_hand,
        state.turn / r.max_turns,
        1.0 if player == 0 else 0.0,
    ]
    if config.uses_poison:
        values += [me.poison / 10.0, opp.poison / 10.0]
    if any(c.combo > 0 for c in config.cards):
        values.append(1.0 if me.combo > 0 else 0.0)
    if config.uses_decks:
        deck = list(me.deck) if me.deck else [0] * n
        values += [count / r.deck_size for count in deck]
        values += [sum(me.deck) / r.deck_size, sum(opp.deck) / r.deck_size]
    if memory:
        played = list(opp.played) if opp.played else [0] * n
        values += [count / MEMORY_SCALE for count in played]
    return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)
