from pathlib import Path

import pytest

from manaduel.agents.heuristic import GreedyAgent
from manaduel.engine.rules import legal_actions, load_game_config, new_game
from manaduel.play import describe_card, play_human, record, summarize
from manaduel.sim.replay import load_replay

CFG = load_game_config()
EXT = load_game_config(Path(__file__).resolve().parents[1] / "configs" / "cards_extended.yaml")


class Script:
    """Stands in for the keyboard: answers each prompt from a list, then keeps ending the turn."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.asked = 0

    def __call__(self, prompt):
        self.asked += 1
        return self.answers.pop(0) if self.answers else "0"


def run(config, answers, seat=0, seed=3):
    out = []
    script = Script(answers)
    result = play_human(config, GreedyAgent(0), seat, seed, "greedy", read=script, write=out.append)
    return result, "\n".join(out), script


def test_a_game_that_only_ends_its_turns_finishes_and_loses_or_draws():
    result, text, script = run(CFG, [])
    assert result.winner in (1, None)             # never playing a card cannot win
    assert "You are player 1 and move first" in text
    assert script.asked >= 1


def test_playing_cards_by_number_uses_them():
    result, text, _ = run(CFG, ["1", "1", "0"], seat=0, seed=0)
    first = [e for e in result.events if e["player"] == 0 and e["action"] != "end_turn"]
    assert first                                   # the typed numbers played real cards
    assert "Your hand:" in text and "cost" in text


def test_invalid_input_is_rejected_and_asked_again():
    result, text, script = run(CFG, ["banana", "99", "0"], seat=0, seed=1)
    assert text.count("Please type one of the numbers above.") == 2
    assert result.events[0]["action"] == "end_turn"


def test_playing_second_lets_the_bot_move_first():
    result, text, _ = run(CFG, [], seat=1, seed=2)
    assert result.events[0]["player"] == 0
    assert "move second" in text


def test_a_human_can_win_by_playing_well():
    """A scripted 'human' that plays the biggest affordable damage card every time beats a random-ish line."""
    wins = 0
    for seed in range(12):
        shown = []

        def read(prompt):
            # play the first listed card when there is one, otherwise end the turn
            menu = []
            for line in reversed(shown):
                menu.append(line)
                if "Play a card" in line:
                    break
            return "1" if any(line.strip().startswith("1)") for line in menu) else "0"

        result = play_human(CFG, GreedyAgent(0), 0, seed, "greedy", read=read, write=shown.append)
        wins += result.winner == 0
        assert result.winner in (0, 1, None)          # every game completes
        assert any(e["player"] == 0 and e["action"] != "end_turn" for e in result.events)
    assert wins >= 1                                    # always playing a card is enough to beat the bot sometimes


def test_describe_card_covers_every_mechanic():
    by = {c.name: describe_card(c) for c in EXT.cards}
    assert by["fireball"] == "deal 7"
    assert by["drain"] == "deal 3, heal 3"
    assert by["venom"] == "poison 3"
    assert by["insight"] == "draw 2"
    assert "if you already played a card this turn" in by["flurry"]


def test_extended_game_shows_poison_in_the_state():
    _, text, _ = run(EXT, [], seat=0, seed=4)
    assert "poison" in text


def test_record_saves_a_replay_and_a_log_row(tmp_path):
    result, _, _ = run(CFG, [], seat=0, seed=5)
    log = tmp_path / "log.csv"
    replay = record(result, CFG, 0, "greedy", 5, "cards.yaml", log, tmp_path / "replays")
    data = load_replay(replay)
    assert data["agents"] == ["You", "greedy"] and data["winner"] == result.winner
    record(result, CFG, 0, "greedy", 6, "cards.yaml", log, tmp_path / "replays")
    assert log.read_text(encoding="utf-8").count("\n") == 3        # header + two games
    text = summarize(log)
    assert "2 games logged" in text and "greedy" in text and "95% interval" in text


def test_summary_without_a_log():
    assert "no games logged" in summarize("does/not/exist.csv")


def test_the_play_command_end_to_end(tmp_path, monkeypatch, capsys):
    import builtins

    from manaduel.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "0")
    main(["play", "--bot", "greedy", "--seat", "1", "--seed", "7"])
    out = capsys.readouterr().out
    assert "Saved the replay" in out
    assert (tmp_path / "runs" / "human_games.csv").exists()
    main(["play", "--summary"])
    assert "1 games logged" in capsys.readouterr().out
