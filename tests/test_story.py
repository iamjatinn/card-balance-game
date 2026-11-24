import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from manaduel.story.render import (  # noqa: E402
    animate_learning,
    caption_text,
    duration_text,
    load_story,
    make_card,
    pick_checkpoints,
    plot_behaviour,
    plot_learning_curve,
    rolling_mean,
    window_for,
)

BOTS = ["random", "aggressive", "defensive", "greedy"]
CARDS = ["strike", "fireball", "heal", "shield", "drain"]


def make_fake_recording(folder: Path, n_matches=600, checkpoints=5, seconds=480.0, turns=(28, 20),
                        hp=(-2, 7), plays=None):
    """A hand-made recording in the same format run_story writes, with a clear upward trend."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with open(folder / "matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["match", "step", "result", "opponent", "seat", "turns"])
        for m in range(1, n_matches + 1):
            p = 0.3 + 0.5 * m / n_matches
            w.writerow([m, m * 20, float(rng.random() < p), BOTS[m % 4], m % 2, 20])
    cols = (["matches", "steps", "seconds", "eval_games", "mean"] + BOTS + ["turns", "hp_margin"]
            + [f"plays_{c}" for c in CARDS])
    with open(folder / "checkpoints.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(checkpoints):
            t = k / (checkpoints - 1)
            wins = [0.3 + 0.55 * t + 0.02 * i for i in range(4)]
            p = plays or [(2.0, 2.0 + (i + 1) * 0.3) for i in range(5)]       # (start, end) per card
            w.writerow([int(n_matches * t), int(n_matches * 20 * t), seconds * t, 100, sum(wins) / 4, *wins,
                        turns[0] + (turns[1] - turns[0]) * t, hp[0] + (hp[1] - hp[0]) * t,
                        *[a + (b - a) * t for a, b in p]])
    (folder / "meta.json").write_text(json.dumps({"config": "cards.yaml", "opponents": BOTS, "seconds": seconds,
                                                  "matches": n_matches, "finished": True}), encoding="utf-8")
    return folder


# ---- pure helpers -----------------------------------------------------------------

def test_rolling_mean_uses_a_shorter_window_at_the_start():
    assert list(rolling_mean([1, 0, 1, 1, 0], 3)) == pytest.approx([1.0, 0.5, 2 / 3, 2 / 3, 2 / 3])
    assert rolling_mean(np.ones(10), 50) == pytest.approx(np.ones(10))
    assert len(rolling_mean([], 5)) == 0


def test_window_and_duration_text():
    assert duration_text(45) == "45 s" and duration_text(89) == "89 s"
    assert duration_text(90) == "2 min" and duration_text(480) == "8 min"


def test_loading_a_recording_and_its_errors(tmp_path):
    story = load_story(make_fake_recording(tmp_path / "ok"))
    assert story.n_matches == 600 and len(story.match) == 600
    assert story.bots == BOTS and story.cards == CARDS
    assert window_for(story) >= 100
    assert pick_checkpoints(story, 4) == [0, 1, 3, 4] or len(pick_checkpoints(story, 4)) == 4
    with pytest.raises(FileNotFoundError):
        load_story(tmp_path / "nothing")
    bad = make_fake_recording(tmp_path / "bad", checkpoints=2)
    (bad / "checkpoints.csv").write_text(
        (bad / "checkpoints.csv").read_text(encoding="utf-8").split("\n")[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_story(bad)


# ---- the pictures, from a hand-made recording (no training needed) ------------------

def test_the_figures_are_written(tmp_path):
    story = load_story(make_fake_recording(tmp_path / "rec"))
    plot_learning_curve(story, tmp_path / "curve.png")
    plot_behaviour(story, tmp_path / "behaviour.png")
    make_card(story, tmp_path / "card.png", footer="example footer")
    from PIL import Image

    for name in ("curve.png", "behaviour.png", "card.png"):
        assert (tmp_path / name).stat().st_size > 10_000
    assert Image.open(tmp_path / "card.png").size == (1200, 627)      # the LinkedIn feed size


def test_the_animation_runs_from_match_one_to_the_last_match(tmp_path):
    story = load_story(make_fake_recording(tmp_path / "rec"))
    n = animate_learning(story, tmp_path / "anim.gif", frames=8, hold=3, size=(600, 338))
    from PIL import Image

    gif = Image.open(tmp_path / "anim.gif")
    assert n == 11 and gif.size == (600, 338)
    assert gif.n_frames <= 11                    # identical hold frames may be merged, never more
    assert gif.n_frames >= 8


def test_the_caption_uses_the_real_numbers_and_states_the_limits(tmp_path):
    story = load_story(make_fake_recording(tmp_path / "rec"))
    text = caption_text(story)
    first, last = story.ck["mean"][0] * 100, story.ck["mean"][-1] * 100
    assert f"{first:.0f}%" in text and f"{last:.0f}%" in text
    assert "600 matches" in text and "8 min" in text
    assert "Honest limits" in text and "DRAFT" in text
    assert "ends games sooner" not in text      # a claim the old template made regardless of the data
    assert "#ReinforcementLearning" in text


def test_the_caption_only_claims_changes_the_numbers_support(tmp_path):
    from manaduel.story.render import learned_facts

    # nothing changed: no 'what changed' claims at all (this is exactly the case that once produced a false line)
    flat = load_story(make_fake_recording(tmp_path / "flat", turns=(21, 21), hp=(0, 0.5),
                                          plays=[(2.0, 2.1)] * 5))
    assert learned_facts(flat) == []
    flat_text = caption_text(flat)
    assert "What changed in its play" not in flat_text and "sooner" not in flat_text

    # a real change in each respect is reported, with the right numbers
    changed = load_story(make_fake_recording(tmp_path / "changed", turns=(26, 21), hp=(-1.5, 8.8),
                                             plays=[(2.4, 2.3), (2.0, 2.4), (2.0, 2.1), (0.9, 2.4), (2.5, 2.4)]))
    facts = " | ".join(learned_facts(changed))
    assert "HP lead of +8.8 (untrained: -1.5)" in facts
    assert "Shield 2.7x as often (0.9 to 2.4 times per game)" in facts
    assert "games last 21 turns on average instead of 26" in facts
    text = caption_text(changed)
    assert "What changed in its play" in text and "- it plays Shield" in text


def test_a_card_that_was_barely_played_is_not_called_a_big_increase(tmp_path):
    from manaduel.story.render import learned_facts

    story = load_story(make_fake_recording(tmp_path / "rare", turns=(21, 21), hp=(0, 0),
                                           plays=[(0.1, 0.6), (2, 2), (2, 2), (2, 2), (2, 2)]))
    assert learned_facts(story) == []          # 0.1 -> 0.6 is a 6x ratio on a tiny base; not a claim worth making


# ---- recording and rendering with a real (tiny) training run -----------------------------

@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    pytest.importorskip("gymnasium")
    pytest.importorskip("sb3_contrib")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from manaduel.engine.rules import load_game_config
    from manaduel.story.record import run_story

    out = tmp_path_factory.mktemp("story")
    config = load_game_config()
    run_story(config, out, steps=2048, chunk_steps=1024, eval_games=4, final_games=6, envs=2, seed=0,
              log=lambda t: None)
    return out, config


def test_the_recording_has_every_file_and_a_match_per_finished_game(tiny_run):
    out, _ = tiny_run
    for name in ("matches.csv", "checkpoints.csv", "meta.json"):
        assert (out / name).exists()
    assert (out / "models" / "ckpt_00000000.zip").exists() and (out / "models" / "final.zip").exists()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(out / "matches.csv", encoding="utf-8")))
    assert meta["finished"] and meta["matches"] == len(rows) > 20
    assert [int(r["match"]) for r in rows] == list(range(1, len(rows) + 1))      # numbered 1..N in order
    assert {r["result"] for r in rows} <= {"0.0", "0.5", "1.0"}
    assert {r["opponent"] for r in rows} <= {"random", "aggressive", "defensive", "greedy"}
    assert {r["seat"] for r in rows} == {"0", "1"}
    assert all(1 <= int(r["turns"]) <= 40 for r in rows)
    assert meta["seconds"] > 0 and meta["wall_seconds"] >= meta["seconds"]


def test_checkpoint_zero_is_the_untrained_agent_and_the_last_row_is_the_end(tiny_run):
    out, _ = tiny_run
    story = load_story(out)
    assert story.ck["matches"][0] == 0 and story.ck["steps"][0] == 0
    assert list(story.ck["steps"]) == [0, 1024, 2048]
    assert story.ck["eval_games"][-1] == 6 and story.ck["eval_games"][0] == 4
    assert story.n_matches == json.loads((out / "meta.json").read_text(encoding="utf-8"))["matches"]
    assert np.all((story.ck["mean"] >= 0) & (story.ck["mean"] <= 1))


def test_evaluation_is_repeatable_so_checkpoints_are_comparable(tiny_run):
    from manaduel.agents.rl_agent import load_ppo_model
    from manaduel.story.record import evaluate_model

    out, config = tiny_run
    model = load_ppo_model(out / "models" / "final.zip")
    a = evaluate_model(model, config, 6)
    b = evaluate_model(model, config, 6)
    assert a == b
    assert set(a["win"]) == {"random", "aggressive", "defensive", "greedy"}
    assert a["turns"] > 0 and set(a["plays_per_game"]) == set(CARDS)


def test_the_environment_reports_who_it_played_when_a_match_ends():
    from manaduel.rl.env import ManaDuelEnv

    env = ManaDuelEnv()
    rng = np.random.default_rng(0)
    env.reset(seed=3)
    done, info = False, {}
    while not done:
        legal = np.flatnonzero(env.action_masks())
        _, _, done, _, info = env.step(int(rng.choice(legal)))
    assert info["opponent"] in {"random", "aggressive", "defensive", "greedy"}
    assert info["learner_side"] in (0, 1) and 1 <= info["turns"] <= 40


def test_render_story_end_to_end_and_the_command(tiny_run, tmp_path, monkeypatch):
    pytest.importorskip("pygame")
    out, config = tiny_run
    from manaduel.cli import main

    folder = tmp_path / "copy"
    import shutil

    shutil.copytree(out, folder)
    main(["story", "--render-only", "--out", str(folder), "--bot", "greedy", "--footer", "test footer"])
    for name in ("learning_curve.png", "behaviour.png", "linkedin_card.png", "learning_animation.gif",
                 "before_after.gif"):
        assert (folder / name).stat().st_size > 5_000, name
    post = (folder / "linkedin_post.md").read_text(encoding="utf-8")
    assert "DRAFT" in post and "Honest limits" in post
    assert (folder / "replays" / "before.json").exists() and (folder / "replays" / "after.json").exists()
    from PIL import Image

    ba = Image.open(folder / "before_after.gif")
    assert ba.size[0] > ba.size[1] * 2 and ba.n_frames > 10        # two games side by side


# ---- the match-by-match pictures (story/matches.py) ---------------------------------------

from manaduel.story import matches as mt  # noqa: E402


def test_grid_holds_every_match_and_is_roughly_the_wanted_shape():
    for n in (1, 7, 600, 28_692):
        rows, cols = mt.grid_shape(n)
        assert rows * cols >= n and (rows - 1) * cols < n          # fits all, no wasted row
    rows, cols = mt.grid_shape(28_692, 1.0)
    assert abs(rows - cols) <= 2


def test_colours_follow_the_result_win_loss_draw():
    c = mt.result_colors([1.0, 0.0, 0.5, 1.0])
    assert [tuple(x) for x in c] == [mt.WIN_RGB, mt.LOSS_RGB, mt.DRAW_RGB, mt.WIN_RGB]


def test_first_and_last_segment_win_rates_use_real_results(tmp_path):
    story = load_story(make_fake_recording(tmp_path / "rec"))
    first, last = mt.segment_win_rates(story, k=100)
    assert first == pytest.approx(story.result[:100].mean())
    assert last == pytest.approx(story.result[-100:].mean())
    assert last > first                                           # the fake run improves


def test_frame_counts_are_increasing_start_early_and_end_at_the_last_match():
    counts = mt.match_frame_counts(28_692, 100)
    assert counts[0] >= 1 and counts[-1] == 28_692
    assert counts == sorted(set(counts))                          # strictly increasing, no repeats
    assert counts[10] < 28_692 * 0.02                             # the early failing phase gets many frames


def test_the_matches_animation_runs_from_match_one_to_the_last(tmp_path):
    from PIL import Image

    story = load_story(make_fake_recording(tmp_path / "rec"))
    n = mt.animate_matches(story, tmp_path / "m.gif", frames=10, hold=3, size=(600, 338))
    gif = Image.open(tmp_path / "m.gif")
    assert gif.size == (600, 338) and gif.n_frames >= 8 and n >= gif.n_frames


def test_stage_checkpoints_start_untrained_end_at_the_end_and_never_repeat(tmp_path):
    story = load_story(make_fake_recording(tmp_path / "rec", checkpoints=9))
    picks = mt.pick_stage_checkpoints(story, 5)
    assert picks[0] == 0 and picks[-1] == len(story.ck["matches"]) - 1
    assert picks == sorted(set(picks))


def test_journey_game_needs_a_loss_first_then_a_win_and_prefers_the_honest_stage(monkeypatch):
    """With scripted outcomes: seed 2 never loses first, seed 3 loses then wins at stage 1 (expected), seed 4
    wins only at the last stage, seed 5 is a win-then-loss path."""
    import types

    import manaduel.agents.rl_agent as rl
    import manaduel.engine.game as game

    table = {1: [1, 1, 1], 2: [0, 1, 1], 3: [0, 0, 1], 5: [0, 1, 0]}     # winner==0 means the agent won
    plan = {s: [types.SimpleNamespace(winner=0 if w else 1, turns=9) for w in ws] for s, ws in table.items()}
    calls = {}

    class FakeAgent:
        def __init__(self, m):
            self.m = m

    def fake_play(config, agents, seed):
        stage = calls.setdefault(seed, [])
        idx = len(stage) % len(plan[seed])        # stage number, cycling per pick
        stage.append(1)
        return plan[seed][idx]

    monkeypatch.setattr(rl, "PPOAgent", FakeAgent)
    monkeypatch.setattr(game, "play_game", fake_play)
    seed, results = mt.pick_journey_game(None, ["a", "b", "c"], "greedy", seeds=[1, 2, 3, 5], expected_first_win=1)
    assert seed == 2 and [r.winner == 0 for r in results] == [False, True, True]
    seed, _ = mt.pick_journey_game(None, ["a", "b", "c"], "greedy", seeds=[1, 3, 5], expected_first_win=2)
    assert seed == 3
    with pytest.raises(RuntimeError):
        mt.pick_journey_game(None, ["a", "b", "c"], "greedy", seeds=[1, 5])
