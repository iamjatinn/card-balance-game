from pathlib import Path

from manaduel.agents.heuristic import BoldAgent, CautiousAgent, DefensiveAgent, GreedyAgent
from manaduel.agents.registry import ALL_BOT_FACTORIES, BOT_FACTORIES, EXTRA_BOT_FACTORIES
from manaduel.engine.game import play_game
from manaduel.engine.rules import load_game_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_game_config()
EXT = load_game_config(ROOT / "configs" / "cards_extended.yaml")


def cards_used(config, bot_factory, games=40):
    used = set()
    for seed in range(games):
        result = play_game(config, (bot_factory(seed), bot_factory(seed + 1)), seed=seed)
        used |= {e["action"] for e in result.events if e["action"] != "end_turn"}
    return used


def test_registries_keep_the_original_four_bots_unchanged():
    assert list(BOT_FACTORIES) == ["random", "aggressive", "defensive", "greedy"]
    assert list(EXTRA_BOT_FACTORIES) == ["cautious", "bold"]
    assert list(ALL_BOT_FACTORIES) == ["random", "aggressive", "defensive", "greedy", "cautious", "bold"]


def test_the_new_bots_play_complete_legal_games_in_every_card_set():
    decks = load_game_config(ROOT / "configs" / "cards_decks.yaml")
    for config in (CFG, EXT, decks):
        for first in (CautiousAgent, BoldAgent):
            for second in (CautiousAgent, BoldAgent, GreedyAgent):
                for seed in range(6):
                    result = play_game(config, (first(seed), second(seed + 1)), seed=seed)
                    assert result.winner in (0, 1, None)


def test_the_new_bots_use_every_extended_card_but_the_defensive_bot_does_not():
    new_cards = {"insight", "venom", "flurry"}
    for bot in (GreedyAgent, CautiousAgent, BoldAgent):
        assert new_cards <= cards_used(EXT, bot), bot
    # why the defensive bot cannot be used to measure these two cards: it never plays them
    assert not ({"insight", "venom"} & cards_used(EXT, DefensiveAgent))


def test_the_bots_really_value_things_differently():
    from manaduel.engine.rules import legal_actions, new_game

    differ = 0
    for seed in range(60):
        s = new_game(EXT, seed)
        legal = legal_actions(s, EXT)
        picks = {cls(0).act(s, legal, EXT) for cls in (GreedyAgent, CautiousAgent, BoldAgent)}
        differ += len(picks) > 1
    assert differ >= 3          # not every opening position separates them, but several do
    games = [play_game(EXT, (CautiousAgent(0), BoldAgent(0)), seed=s).events for s in range(8)]
    assert len({str(g) for g in games}) > 1


def test_greedy_is_still_exactly_the_v01_greedy():
    # the weights refactor must not change the original bot (the golden regression also guards this)
    assert GreedyAgent.weights == {"damage": 1.0, "heal": 1.0, "shield": 0.7, "draw": 1.5, "poison": 0.8}
