"""Turn a game state into the list of numbers a neural network can read.

The network never sees the opponent's actual cards, only how many they hold:
a real player cannot see them either.
"""

import numpy as np


def observation_size(config) -> int:
    # my hp, shield, mana | my hand (one number per card type)
    # | opponent hp, shield, hand size | turn | am I the first player
    return 3 + len(config.cards) + 3 + 1 + 1


def observation(state, player: int, config) -> np.ndarray:
    """The game as seen by `player`, scaled so every number is between 0 and 1."""
    r = config.rules
    me = state.players[player]
    opp = state.players[1 - player]
    hand = list(me.hand) if me.hand else [0] * len(config.cards)
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
    return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)
