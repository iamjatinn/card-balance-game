import pytest

pytest.importorskip("pygame")

import os  # noqa: E402

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no real window in tests

import pygame  # noqa: E402

from manaduel.agents.heuristic import DefensiveAgent, GreedyAgent  # noqa: E402
from manaduel.engine.game import play_game  # noqa: E402
from manaduel.engine.rules import load_game_config  # noqa: E402
from manaduel.sim.replay import load_replay, save_replay  # noqa: E402
from manaduel.viewer import pygame_viewer as pv  # noqa: E402

CFG = load_game_config()


@pytest.fixture(scope="module")
def replay_path(tmp_path_factory):
    result = play_game(CFG, (GreedyAgent(8), DefensiveAgent(9)), seed=8)
    path = tmp_path_factory.mktemp("replays") / "game.json"
    save_replay(path, result, CFG, ("Greedy", "Defensive"), seed=8)
    return path


def test_beats_are_the_card_plays_and_chain_together(replay_path):
    replay = load_replay(replay_path)
    view = pv.build_view(replay)
    plays = [e for e in replay["events"] if e["action"] != "end_turn"]
    assert len(view.beats) == len(plays) > 0
    for earlier, later in zip(view.beats, view.beats[1:]):
        assert earlier.after.hp == later.before.hp          # HP is continuous between beats
    first = view.beats[0]
    assert first.before.hp == (CFG.rules.max_hp,) * 2
    assert first.before.mana[0] == 1                        # player 0 starts turn 1 with 1 mana
    assert [sum(h) for h in view.start.hand] == [3, 3]


def test_absorbed_damage_matches_the_numbers(replay_path):
    view = pv.build_view(load_replay(replay_path))
    for beat in view.beats:
        if beat.card["damage"]:
            opp = 1 - beat.player
            took = beat.before.hp[opp] - beat.after.hp[opp]
            assert took == beat.card["damage"] - beat.absorbed


def test_every_frame_renders_and_is_not_blank(replay_path):
    view = pv.build_view(load_replay(replay_path))
    fonts = pv.Fonts()
    for idx in (0, len(view.beats) // 2, len(view.beats) - 1, len(view.beats)):
        for p in (0.0, 0.3, 0.6, 0.95):
            surface = pv.render_to_surface(view, idx, p, fonts)
            assert surface.get_size() == (pv.W, pv.H)
            colors = {surface.get_at((x, y))[:3] for x in range(0, pv.W, 40) for y in range(0, pv.H, 40)}
            assert len(colors) > 4


def test_a_drawn_game_and_a_win_both_render(tmp_path):
    result = play_game(CFG, (GreedyAgent(1), GreedyAgent(2)), seed=3)
    path = save_replay(tmp_path / "g.json", result, CFG, ("A", "B"), seed=3)
    view = pv.build_view(load_replay(path))
    assert pv.render_to_surface(view, len(view.beats), 1.0).get_size() == (pv.W, pv.H)


def test_player_controller():
    view = pv.build_view(load_replay_stub())
    player = pv.Player(view, speed=1.0)
    assert player.idx == 0 and player.playing
    player.update(pv.BEAT_SECONDS * 1.5)
    assert player.idx == 1
    player.step(1)
    assert player.idx == 2 and not player.playing
    player.step(-5)
    assert player.idx == 0
    player.step(99)
    assert player.finished
    player.restart()
    assert player.idx == 0 and player.playing
    player.change_speed(100)
    assert player.speed == 8.0
    player.change_speed(1e-6)
    assert player.speed == 0.25


def test_png_and_gif_export(replay_path, tmp_path):
    png = tmp_path / "frame.png"
    pv.save_png(png, replay_path)
    assert png.stat().st_size > 5_000

    gif = tmp_path / "game.gif"
    frames = pv.save_gif(gif, replay_path, fps=8, speed=3.0, max_beats=3)
    assert gif.stat().st_size > 5_000 and frames >= 6
    from PIL import Image

    with Image.open(gif) as img:
        assert img.n_frames >= 6 and img.size == (640, 360)


def load_replay_stub():
    result = play_game(CFG, (GreedyAgent(4), DefensiveAgent(5)), seed=4)
    return {
        "agents": ["A", "B"],
        "winner": result.winner,
        "turns": result.turns,
        "rules": {"max_hp": 20, "max_mana": 5, "max_shield": 6},
        "cards": [{"name": c.name, "cost": c.cost, "damage": c.damage, "heal": c.heal,
                   "shield": c.shield, "weight": c.weight} for c in CFG.cards],
        "start_hands": result.start_hands,
        "events": result.events,
    }
