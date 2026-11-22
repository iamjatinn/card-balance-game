"""The game loop: let two agents play one full game and record a replay."""

from dataclasses import dataclass

from manaduel.engine.rules import GameConfig, legal_actions, new_game, step
from manaduel.engine.state import GameState, is_terminal, winner


@dataclass(frozen=True)
class GameResult:
    winner: int | None   # 0, 1, or None for a draw
    turns: int           # turn number the game ended on
    events: list[dict]   # one JSON-friendly record per action: the replay


def _event(before: GameState, action: int, after: GameState, config: GameConfig) -> dict:
    return {
        "turn": before.turn,
        "player": before.current,
        "action": config.action_name(action),
        "hp": [p.hp for p in after.players],
        "shield": [p.shield for p in after.players],
        "mana": [p.mana for p in after.players],
        "hand": [list(p.hand) for p in after.players],
    }


def play_game(config: GameConfig, agents, seed: int = 0, state: GameState | None = None) -> GameResult:
    """agents is a pair (agent for player 0, agent for player 1).
    `seed` decides the card draws. `state` lets a test start from a chosen position."""
    state = new_game(config, seed) if state is None else state
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
    )
