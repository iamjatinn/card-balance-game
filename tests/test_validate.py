from manaduel.agents.heuristic import GreedyAgent
from manaduel.analysis.validate import (
    NullTrial,
    Trial,
    ValidationResult,
    format_validation,
    run_validation,
    summarize_by_size,
    write_csv,
)
from manaduel.engine.rules import load_game_config

CFG = load_game_config()


def test_a_large_planted_change_is_detected_and_attributed():
    result = run_validation(CFG, GreedyAgent, deltas=(6,), n_games=1200, baseline_games=2000,
                            reps=1, null_reps=1, seed=0, workers=1)
    fireball = next(t for t in result.trials if t.card == "fireball")
    assert fireball.new == 13 and fireball.change > 0.03
    assert fireball.detected and fireball.attributed


def test_nerfs_are_tested_and_cannot_go_below_one():
    result = run_validation(CFG, GreedyAgent, deltas=(-2,), n_games=300, baseline_games=300,
                            reps=1, null_reps=1, seed=0, workers=1)
    assert {t.card for t in result.trials} == {"strike", "fireball", "heal", "shield", "drain"}
    assert all(t.new >= 1 for t in result.trials)
    assert all(t.new < t.old for t in result.trials)


def test_trials_run_in_parallel_and_match_the_serial_result():
    kwargs = dict(deltas=(2,), n_games=200, baseline_games=200, reps=1, null_reps=1, seed=3)
    serial = run_validation(CFG, GreedyAgent, workers=1, **kwargs)
    parallel = run_validation(CFG, GreedyAgent, workers=2, **kwargs)
    assert serial == parallel


def test_summary_groups_and_report_text(tmp_path):
    def trial(delta_pct_new, detected):
        return Trial("fireball", "damage", 10, delta_pct_new, 0, 0.05, 4.0, detected, detected)

    result = ValidationResult(
        # old is 10, so these are +10%, +20%, +40% and +100%
        trials=[trial(11, False), trial(12, True), trial(14, True), trial(20, True)],
        null_trials=[NullTrial(0, 1.2, False), NullTrial(1, 3.1, True)],
        n_games=1000,
        alpha=0.05,
    )
    summary = {label: (n, det) for label, n, det, _ in summarize_by_size(result)}
    assert summary == {"under 25%": (2, 1), "25% to 50%": (1, 1), "50% or more": (1, 1)}
    text = format_validation(result)
    assert "False alarms with NO change planted: 1/2" in text
    write_csv(result, tmp_path / "v.csv")
    assert (tmp_path / "v.csv").read_text(encoding="utf-8").count("\n") == 5
