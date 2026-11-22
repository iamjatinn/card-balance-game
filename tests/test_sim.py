from manaduel.agents.heuristic import AggressiveAgent, DefensiveAgent, GreedyAgent
from manaduel.agents.random_agent import RandomAgent
from manaduel.engine.game import play_game
from manaduel.engine.rules import load_game_config
from manaduel.sim.elo import fit_elo
from manaduel.sim.replay import load_replay, save_replay
from manaduel.sim.stats import wilson_interval
from manaduel.sim.tournament import play_match, round_robin

CFG = load_game_config()


def test_wilson_interval_basics():
    lo, hi = wilson_interval(50, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert abs((0.5 - lo) - (hi - 0.5)) < 0.01  # symmetric around 50%


def test_wilson_interval_stays_in_bounds_at_extremes():
    lo, hi = wilson_interval(100, 100)
    assert 0.9 < lo <= hi <= 1.0
    lo, hi = wilson_interval(0, 100)
    assert 0.0 <= lo <= hi < 0.1


def test_more_games_give_a_narrower_interval():
    lo1, hi1 = wilson_interval(5, 10)
    lo2, hi2 = wilson_interval(500, 1000)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_match_counts_add_up_and_are_reproducible():
    a = play_match(CFG, "aggr", AggressiveAgent, "rand", RandomAgent, 40, seed=1)
    b = play_match(CFG, "aggr", AggressiveAgent, "rand", RandomAgent, 40, seed=1)
    assert a.wins_a + a.wins_b + a.draws == 40
    assert a == b


def test_greedy_beats_random():
    r = play_match(CFG, "greedy", GreedyAgent, "rand", RandomAgent, 300, seed=0)
    assert r.score_a > 0.7


def test_elo_ranks_the_stronger_bot_higher():
    factories = {"random": RandomAgent, "defensive": DefensiveAgent, "greedy": GreedyAgent}
    results = round_robin(CFG, factories, 100, seed=0)
    pairs = [r for r in results if r.name_a != r.name_b]
    ratings = fit_elo(list(factories), pairs)
    assert ratings["random"] < ratings["defensive"]
    assert ratings["random"] < ratings["greedy"]
    assert abs(sum(ratings.values()) / 3 - 1000) < 1e-6


def test_replay_round_trip(tmp_path):
    result = play_game(CFG, (RandomAgent(3), RandomAgent(4)))
    path = save_replay(tmp_path / "game.json", result, CFG, ("A", "B"), seed=3)
    data = load_replay(path)
    assert data["winner"] == result.winner
    assert data["turns"] == result.turns
    assert data["events"] == result.events
    assert data["agents"] == ["A", "B"]
    assert len(data["cards"]) == 5
    assert data["start_hands"] == result.start_hands
    assert [sum(h) for h in data["start_hands"]] == [3, 3]


def test_match_tracks_average_game_length():
    r = play_match(CFG, "greedy", GreedyAgent, "rand", RandomAgent, 40, seed=2)
    assert r.turns_total > 0
    assert 1 <= r.avg_turns <= CFG.rules.max_turns
