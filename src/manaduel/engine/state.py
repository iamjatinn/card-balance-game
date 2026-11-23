"""Game state: a snapshot of everything needed to describe the game right now."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerState:
    hp: int = 20
    shield: int = 0
    mana: int = 0
    # How many copies of each card type this player holds, in cards.yaml order.
    # For example (2, 0, 1, 0, 0) = two strikes and one heal.
    hand: tuple[int, ...] = ()
    # --- used only by the extra mechanics; all stay at their defaults in the original game ---
    poison: int = 0          # counters: at the start of this player's turn they lose this much HP, then it drops by 1
    combo: int = 0           # cards this player has played so far this turn
    fatigue: int = 0         # failed draws from an empty deck so far
    # Cumulative copies played of each card type (known to the opponent: this is the "memory" signal).
    played: tuple[int, ...] = ()
    # Cards left in this player's deck, per card type. Empty means endless random draws.
    deck: tuple[int, ...] = ()


@dataclass(frozen=True)
class GameState:
    players: tuple[PlayerState, PlayerState]
    current: int = 0   # whose turn it is: 0 or 1
    turn: int = 1      # total turns taken so far by both players, starting at 1
    rng: int = 0       # state of the game's own random number generator (for card draws)


def initial_state(max_hp: int = 20) -> GameState:
    """Return the starting state: both players at max_hp, 0 shield, 0 mana,
    player 0 to move, turn 1."""
    return GameState(
        players=(PlayerState(hp=max_hp), PlayerState(hp=max_hp)),
        current=0,
        turn=1,
    )


def is_terminal(state: GameState, max_turns: int = 40) -> bool:
    """True if the game is over: someone's hp is 0 or below,
    or the turn counter has gone past max_turns."""
    someone_dead = any(p.hp <= 0 for p in state.players)
    out_of_turns = state.turn > max_turns
    return someone_dead or out_of_turns


def winner(state: GameState, max_turns: int = 40) -> int | None:
    """Return 0 or 1 for the winning player, or None for a draw / unfinished game.
    A player wins if the other player's hp is 0 or below."""
    p0, p1 = state.players
    if p0.hp <= 0 and p1.hp > 0:
        return 1
    if p1.hp <= 0 and p0.hp > 0:
        return 0
    return None
