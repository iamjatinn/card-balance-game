import dataclasses

from manaduel.agents.heuristic import AggressiveAgent, DefensiveAgent, GreedyAgent
from manaduel.agents.random_agent import RandomAgent
from manaduel.engine.game import play_game
from manaduel.engine.rules import legal_actions, load_game_config

CFG = load_game_config()
IDX = {card.name: i for i, card in enumerate(CFG.cards)}
END = CFG.end_turn


def with_player(state, i, **changes):
    players = list(state.players)
    players[i] = dataclasses.replace(players[i], **changes)
    return dataclasses.replace(state, players=tuple(players))


def new_game(config):
    """A new game where both players hold plenty of every card, so these tests
    check the bot's choice, not what it happened to draw."""
    from manaduel.engine.rules import new_game as real_new_game

    s = real_new_game(config)
    for i in (0, 1):
        s = with_player(s, i, hand=(5,) * len(config.cards))
    return s


def choose(agent, state):
    return agent.act(state, legal_actions(state, CFG), CFG)


def test_aggressive_picks_biggest_affordable_hit():
    s = with_player(new_game(CFG), 0, mana=3)
    assert choose(AggressiveAgent(), s) == IDX["fireball"]
    s = with_player(new_game(CFG), 0, mana=1)
    assert choose(AggressiveAgent(), s) == IDX["strike"]
    s = with_player(new_game(CFG), 0, mana=0)
    assert choose(AggressiveAgent(), s) == END


def test_defensive_heals_when_hurt():
    s = with_player(new_game(CFG), 0, hp=8, mana=2)
    assert choose(DefensiveAgent(), s) == IDX["heal"]


def test_defensive_shields_when_healthy_and_unshielded():
    s = with_player(new_game(CFG), 0, mana=1)
    assert choose(DefensiveAgent(), s) == IDX["shield"]


def test_defensive_attacks_when_already_shielded():
    s = with_player(new_game(CFG), 0, mana=1, shield=3)
    assert choose(DefensiveAgent(), s) == IDX["strike"]


def test_greedy_takes_lethal():
    s = with_player(new_game(CFG), 0, mana=3)
    s = with_player(s, 1, hp=5)
    assert choose(GreedyAgent(), s) == IDX["fireball"]  # 7 damage kills at 5 hp


def test_greedy_ends_turn_with_no_mana():
    s = with_player(new_game(CFG), 0, mana=0)
    assert choose(GreedyAgent(), s) == END


def test_every_bot_plays_legal_complete_games():
    bots = [AggressiveAgent, DefensiveAgent, GreedyAgent, RandomAgent]
    for first in bots:
        for second in bots:
            for seed in range(20):
                result = play_game(CFG, (first(seed), second(seed + 1)))
                assert result.winner in (0, 1, None)
