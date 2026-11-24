"""Game rules: which actions are legal, and step(), which applies an action.

Actions are plain integers so they are easy to log and, later, to feed to RL:
    0 .. len(cards)-1  -> play that card (same order as configs/cards.yaml)
    len(cards)         -> end the turn

Card draws are random, but the game stays a pure function: the random number
generator's state lives inside GameState (the `rng` field), so the same state
and action always give the same next state, and a game is replayable from its seed.

The extra mechanics (draw, poison, combo, finite decks, played-card memory) are all
off by default: with a card set that does not use them, the game plays exactly as
it did in v0.1 (a test replays 200 recorded games to prove it).
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

    @property
    def uses_poison(self) -> bool:
        return any(c.poison > 0 for c in self.cards)

    @property
    def uses_decks(self) -> bool:
        return self.rules.deck_size > 0


def check_config(config: GameConfig) -> None:
    """Reject settings that would break the game (for example, a free card)."""
    for card in config.cards:
        # A free card could be played forever and a turn would never end.
        if card.cost < 1:
            raise ValueError(f"card '{card.name}' must cost at least 1 mana")
        if card.weight < 1:
            raise ValueError(f"card '{card.name}' must have weight of at least 1")
        if card.draw < 0 or card.poison < 0 or card.combo < 0:
            raise ValueError(f"card '{card.name}' cannot have a negative draw, poison or combo")
    rules = config.rules
    if rules.start_hand + rules.second_player_extra_cards < 0:
        raise ValueError("the second player would start with a negative number of cards")
    if rules.max_hp < 1 or rules.max_mana < 1 or rules.max_hand < 1 or rules.max_turns < 1:
        raise ValueError("max_hp, max_mana, max_hand and max_turns must all be at least 1")
    if rules.max_shield < 1:
        raise ValueError("max_shield must be at least 1")
    if not 0 <= rules.shield_keep_percent <= 100:
        raise ValueError("shield_keep_percent must be between 0 and 100")
    if rules.deck_size < 0 or rules.fatigue_damage < 1:
        raise ValueError("deck_size cannot be negative and fatigue_damage must be at least 1")
    if rules.deck_size and rules.deck_size < rules.start_hand + max(0, rules.second_player_extra_cards):
        raise ValueError("the deck is too small to deal the starting hands")


def load_game_config(path: Path = CONFIG_PATH) -> GameConfig:
    cards, rules = load_config(path)
    config = GameConfig(cards=tuple(cards.values()), rules=rules)
    check_config(config)
    return config


# --------------------------------------------------------------------------
# Decks
# --------------------------------------------------------------------------

def allocate(weights, size: int, max_copies: int | None = None) -> tuple[int, ...]:
    """Split `size` cards between card types in proportion to `weights` (D'Hondt method),
    optionally with at most `max_copies` of each type. Deterministic."""
    counts = [0] * len(weights)
    for _ in range(size):
        open_types = [i for i in range(len(weights)) if max_copies is None or counts[i] < max_copies]
        if not open_types:
            raise ValueError("max_copies is too small for this deck size")
        best = max(open_types, key=lambda i: (weights[i] / (counts[i] + 1), -i))
        counts[best] += 1
    return tuple(counts)


def default_deck(config: GameConfig) -> tuple[int, ...]:
    """The deck both players get when none is chosen: card types in proportion to their draw weights."""
    return allocate([c.weight for c in config.cards], config.rules.deck_size)


def check_deck(config: GameConfig, deck) -> tuple[int, ...]:
    if len(deck) != len(config.cards) or any(n < 0 for n in deck) or sum(deck) != config.rules.deck_size:
        raise ValueError(
            f"a deck needs {len(config.cards)} counts that are not negative and add up to "
            f"{config.rules.deck_size}, got {tuple(deck)}"
        )
    return tuple(int(n) for n in deck)


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


def _draw_from_deck(rng: int, deck: list[int]) -> tuple[int, int]:
    """Pick a card type in proportion to how many copies are left in the deck."""
    rng = _next_rng(rng)
    pick = (rng >> 8) % sum(deck)
    for i, count in enumerate(deck):
        if pick < count:
            return i, rng
        pick -= count
    raise AssertionError("unreachable")


def _draw(state: GameState, player: int, count: int, config: GameConfig) -> GameState:
    """Give `player` `count` cards. Cards beyond max_hand are burned. With a finite deck the
    cards come out of it; drawing from an empty deck costs fatigue damage instead."""
    rng = state.rng
    players = list(state.players)
    me = players[player]
    hand = list(me.hand) if me.hand else [0] * len(config.cards)
    deck = list(me.deck) if me.deck else None      # None = endless random draws
    hp, fatigue = me.hp, me.fatigue
    for _ in range(count):
        if deck is None:
            card, rng = _draw_one(rng, config.cards)
        elif sum(deck) > 0:
            card, rng = _draw_from_deck(rng, deck)
            deck[card] -= 1
        else:
            fatigue += 1
            hp -= fatigue * config.rules.fatigue_damage
            continue
        if sum(hand) < config.rules.max_hand:
            hand[card] += 1
    players[player] = replace(
        me, hand=tuple(hand), hp=hp, fatigue=fatigue, deck=tuple(deck) if deck is not None else me.deck
    )
    return replace(state, players=tuple(players), rng=rng)


# --------------------------------------------------------------------------
# Turn structure
# --------------------------------------------------------------------------

def begin_turn(state: GameState, config: GameConfig) -> GameState:
    """Start-of-turn effects for the player whose turn it now is: refill mana, their old
    shield expires (or part of it stays, see rules.shield_keep_percent), poison hurts them (ignoring shield) and fades, and they draw cards
    (except on turn 1: the very first player does not draw)."""
    own_turn = (state.turn + 1) // 2  # turns 1,2 -> 1; turns 3,4 -> 2; ...
    players = list(state.players)
    me = players[state.current]
    players[state.current] = replace(
        me,
        mana=min(own_turn, config.rules.max_mana),
        shield=me.shield * config.rules.shield_keep_percent // 100,   # 0: expires, 100: stays until used
        combo=0,
        hp=me.hp - me.poison,
        poison=max(0, me.poison - 1),
    )
    state = replace(state, players=tuple(players))
    if state.turn > 1 and players[state.current].hp > 0:
        state = _draw(state, state.current, config.rules.draw_per_turn, config)
    return state


def new_game(config: GameConfig, seed: int = 0, decks=None) -> GameState:
    """A fresh game: shuffle-by-seed, deal starting hands, begin player 0's turn.
    The player who moves second gets extra cards as compensation. In deck mode `decks`
    (one count-per-card tuple per player) says what each player brought; the default is
    the same balanced deck for both."""
    state = initial_state(config.rules.max_hp)
    if config.uses_decks:
        d0, d1 = decks if decks is not None else (default_deck(config), default_deck(config))
        p0, p1 = state.players
        state = replace(state, players=(replace(p0, deck=check_deck(config, d0)),
                                        replace(p1, deck=check_deck(config, d1))))
    elif decks is not None:
        raise ValueError("decks were given but this game has no deck_size")
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

    # A combo card hits harder if another card was already played this turn.
    damage = card.damage + (card.combo if mine.combo > 0 else 0)
    # Damage hits the opponent's shield first; the rest goes to HP.
    absorbed = min(theirs.shield, damage)
    theirs = replace(
        theirs,
        shield=theirs.shield - absorbed,
        hp=theirs.hp - (damage - absorbed),
        poison=theirs.poison + card.poison,
    )
    # Use up the card, pay the cost, heal (capped at max HP), gain shield (capped).
    hand = list(mine.hand)
    hand[action] -= 1
    played = list(mine.played) if mine.played else [0] * len(config.cards)
    played[action] += 1
    mine = replace(
        mine,
        mana=mine.mana - card.cost,
        hp=min(rules.max_hp, mine.hp + card.heal),
        shield=min(rules.max_shield, mine.shield + card.shield),
        hand=tuple(hand),
        combo=mine.combo + 1,
        played=tuple(played),
    )

    players[me], players[opp] = mine, theirs
    state = replace(state, players=tuple(players))
    if card.draw:
        state = _draw(state, me, card.draw, config)
    return state


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
