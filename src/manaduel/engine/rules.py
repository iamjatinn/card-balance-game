"""Game rules: which actions are legal, and step(), which applies an action.

Actions are plain integers so they are easy to log and, later, to feed to RL:
    0 .. len(cards)-1  -> play that card (same order as configs/cards.yaml)
    len(cards)         -> end the turn

Card draws are random, but the game stays a pure function: the random number
generator's state lives inside GameState (the `rng` field), so the same state
and action always give the same next state, and a game is replayable from its seed.
"""

from dataclasses import dataclass, replace
from pathlib import Path

from manaduel.engine.cards import CONFIG_PATH, Card, Rules, load_config
from manaduel.engine.state import GameState, PlayerState, initial_state, is_terminal


@dataclass(frozen=True)
class GameConfig:
    """Everything that defines one version of the game: the cards and the rules."""

    cards: tuple[Card, ...]
    rules: Rules

    @property
    def end_turn(self) -> int:
        """The action number that means 'end my turn'."""
        return len(self.cards)

    @property
    def n_actions(self) -> int:
        return len(self.cards) + 1

    def action_name(self, action: int) -> str:
        return "end_turn" if action == self.end_turn else self.cards[action].name


def load_game_config(path: Path = CONFIG_PATH) -> GameConfig:
    cards, rules = load_config(path)
    for card in cards.values():
        # A free card could be played forever and a turn would never end.
        if card.cost < 1:
            raise ValueError(f"card '{card.name}' must cost at least 1 mana")
        if card.weight < 1:
            raise ValueError(f"card '{card.name}' must have weight of at least 1")
    return GameConfig(cards=tuple(cards.values()), rules=rules)


# --------------------------------------------------------------------------
# Random card draws
# --------------------------------------------------------------------------

def _next_rng(rng: int) -> int:
    """One step of a simple linear congruential generator (same constants as glibc)."""
    return (rng * 1103515245 + 12345) & 0x7FFFFFFF


def _draw_one(rng: int, cards: tuple[Card, ...]) -> tuple[int, int]:
    """Pick a card type, weighted by each card's `weight`. Returns (card index, new rng)."""
    rng = _next_rng(rng)
    pick = (rng >> 8) % sum(card.weight for card in cards)
    for i, card in enumerate(cards):
        if pick < card.weight:
            return i, rng
        pick -= card.weight
    raise AssertionError("unreachable")


def _draw(state: GameState, player: int, count: int, config: GameConfig) -> GameState:
    """Give `player` `count` random cards. Cards beyond max_hand are burned."""
    rng = state.rng
    players = list(state.players)
    me = players[player]
    hand = list(me.hand) if me.hand else [0] * len(config.cards)
    for _ in range(count):
        card, rng = _draw_one(rng, config.cards)
        if sum(hand) < config.rules.max_hand:
            hand[card] += 1
    players[player] = replace(me, hand=tuple(hand))
    return replace(state, players=tuple(players), rng=rng)


# --------------------------------------------------------------------------
# Turn structure
# --------------------------------------------------------------------------

def begin_turn(state: GameState, config: GameConfig) -> GameState:
    """Start-of-turn effects for the player whose turn it now is:
    refill mana, their old shield expires, and they draw cards
    (except on turn 1: the very first player does not draw)."""
    own_turn = (state.turn + 1) // 2  # turns 1,2 -> 1; turns 3,4 -> 2; ...
    players = list(state.players)
    me = players[state.current]
    players[state.current] = replace(
        me, mana=min(own_turn, config.rules.max_mana), shield=0
    )
    state = replace(state, players=tuple(players))
    if state.turn > 1:
        state = _draw(state, state.current, config.rules.draw_per_turn, config)
    return state


def new_game(config: GameConfig, seed: int = 0) -> GameState:
    """A fresh game: shuffle-by-seed, deal starting hands, begin player 0's turn.
    The player who moves second gets extra cards as compensation."""
    state = initial_state(config.rules.max_hp)
    state = replace(state, rng=(seed * 2654435761 + 12345) & 0x7FFFFFFF)
    state = _draw(state, 0, config.rules.start_hand, config)
    state = _draw(
        state, 1, config.rules.start_hand + config.rules.second_player_extra_cards, config
    )
    return begin_turn(state, config)


def _holds(player: PlayerState, card_index: int) -> bool:
    return card_index < len(player.hand) and player.hand[card_index] > 0


def legal_actions(state: GameState, config: GameConfig) -> list[int]:
    """Actions the current player may take. Empty once the game is over.
    A card is playable if it is in hand and affordable."""
    if is_terminal(state, config.rules.max_turns):
        return []
    me = state.players[state.current]
    actions = [
        i for i, card in enumerate(config.cards)
        if card.cost <= me.mana and _holds(me, i)
    ]
    actions.append(config.end_turn)  # ending the turn is always allowed
    return actions


def _play_card(state: GameState, action: int, config: GameConfig) -> GameState:
    card = config.cards[action]
    rules = config.rules
    me, opp = state.current, 1 - state.current
    players = list(state.players)
    mine, theirs = players[me], players[opp]

    # Damage hits the opponent's shield first; the rest goes to HP.
    absorbed = min(theirs.shield, card.damage)
    theirs = replace(
        theirs,
        shield=theirs.shield - absorbed,
        hp=theirs.hp - (card.damage - absorbed),
    )
    # Use up the card, pay the cost, heal (capped at max HP), gain shield (capped).
    hand = list(mine.hand)
    hand[action] -= 1
    mine = replace(
        mine,
        mana=mine.mana - card.cost,
        hp=min(rules.max_hp, mine.hp + card.heal),
        shield=min(rules.max_shield, mine.shield + card.shield),
        hand=tuple(hand),
    )

    players[me], players[opp] = mine, theirs
    return replace(state, players=tuple(players))


def step(state: GameState, action: int, config: GameConfig) -> GameState:
    """Apply one action and return the NEW state. The old state is untouched."""
    if action not in legal_actions(state, config):
        name = config.action_name(action) if 0 <= action < config.n_actions else "?"
        me = state.players[state.current]
        raise ValueError(
            f"illegal action {action} ({name}) for player {state.current} "
            f"with mana {me.mana} and hand {me.hand}"
        )
    if action == config.end_turn:
        passed = replace(state, current=1 - state.current, turn=state.turn + 1)
        return begin_turn(passed, config)
    return _play_card(state, action, config)
