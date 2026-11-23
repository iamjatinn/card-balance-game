"""The game loop: let two agents play one full game and record a replay."""

from dataclasses import dataclass, field

from manaduel.engine.rules import GameConfig, legal_actions, new_game, step
from manaduel.engine.state import GameState, is_terminal, winner


@dataclass(frozen=True)
class GameResult:
    winner: int | None   # 0, 1, or None for a draw
    turns: int           # turn number the game ended on
    events: list[dict]   # one JSON-friendly record per action: the replay
    # The cards each player was dealt before the game began (copies of each card
    # type). The deal is random and independent of how anyone plays, which is
    # what lets the balance tool measure card power without bias.
    start_hands: list[list[int]] = field(default_factory=list)
    # In deck mode: the deck each player brought (copies of each card type).
    decks: list[list[int]] = field(default_factory=list)


def _event(before: GameState, action: int, after: GameState, config: GameConfig) -> dict:
    event = {
        "turn": before.turn,
        "player": before.current,
        "action": config.action_name(action),
        "hp": [p.hp for p in after.players],
        "shield": [p.shield for p in after.players],
        "mana": [p.mana for p in after.players],
        "hand": [list(p.hand) for p in after.players],
    }
    # Extra keys appear only when the game uses those mechanics, so the original
    # game's replays are unchanged.
    if config.uses_poison:
        event["poison"] = [p.poison for p in after.players]
    if config.uses_decks:
        event["deck_left"] = [sum(p.deck) for p in after.players]
    return event


def play_game(config: GameConfig, agents, seed: int = 0, state: GameState | None = None,
              decks=None) -> GameResult:
    """agents is a pair (agent for player 0, agent for player 1).
    `seed` decides the card draws. `state` lets a test start from a chosen position.
    `decks` (deck mode only) is a pair of decks, one per seat."""
    state = new_game(config, seed, decks) if state is None else state
    start_hands = [list(p.hand) for p in state.players]
    # The deck a player brought = what is left of it after the starting hand was dealt + that hand.
    start_decks = ([[d + h for d, h in zip(p.deck, p.hand)] for p in state.players]
                   if config.uses_decks else [])
    events: list[dict] = []
    max_turns = config.rules.max_turns

    while not is_terminal(state, max_turns):
        legal = legal_actions(state, config)
        action = agents[state.current].act(state, legal, config)
        if action not in legal:
            raise ValueError(
                f"agent {agents[state.current].name!r} chose illegal action {action}"
            )
        after = step(state, action, config)
        events.append(_event(state, action, after, config))
        state = after

    return GameResult(
        winner=winner(state, max_turns),
        turns=min(state.turn, max_turns),
        events=events,
        start_hands=start_hands,
        decks=start_decks,
    )
