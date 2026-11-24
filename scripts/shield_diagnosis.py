"""Where does a Shield card's value go?  For many agent-vs-agent games, count the shield points gained from Shield
cards, how many of them absorbed damage, and how many expired unused. Usage:

    python scripts/shield_diagnosis.py CONFIG.yaml MODEL.zip [games]
"""
import sys
from pathlib import Path

from manaduel.agents.rl_agent import PPOAgent, load_ppo_model
from manaduel.engine.rules import legal_actions, load_game_config, new_game, step
from manaduel.engine.state import is_terminal


def diagnose(config, model, games: int, seed: int = 5):
    agents = (PPOAgent(model), PPOAgent(model))
    idx = next(i for i, c in enumerate(config.cards) if c.shield > 0 and c.damage == 0)
    played = gained = absorbed = expired = 0
    dmg_cards = turns_with_shield_up = enemy_turns = 0
    for g in range(games):
        state = new_game(config, seed + 1_000_003 * g)
        while not is_terminal(state, config.rules.max_turns):
            me = state.current
            action = agents[me].act(state, legal_actions(state, config), config)
            before = state
            state = step(state, action, config)
            if action == config.end_turn:
                # the player who just became active loses their old shield
                expired += state.players[state.current].shield == 0 and before.players[state.current].shield or 0
                enemy_turns += 1
                turns_with_shield_up += before.players[state.current].shield > 0
                continue
            card = config.cards[action]
            if action == idx:
                played += 1
                gained += state.players[me].shield - before.players[me].shield
            if card.damage:
                absorbed += before.players[1 - me].shield - state.players[1 - me].shield
    return dict(games=games, shield_cards_played=played, points_gained=gained, points_absorbed=absorbed,
                points_expired_unused=expired,
                share_absorbed=absorbed / gained if gained else float("nan"),
                played_per_game=played / games,
                absorbed_per_card_played=absorbed / played if played else float("nan"))


if __name__ == "__main__":
    cfg = load_game_config(Path(sys.argv[1]))
    result = diagnose(cfg, load_ppo_model(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 300)
    for k, v in result.items():
        print(f"{k:>26}: {v:.2f}" if isinstance(v, float) else f"{k:>26}: {v}")
