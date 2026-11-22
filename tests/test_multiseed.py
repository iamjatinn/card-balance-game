import numpy as np
import pytest

pytest.importorskip("matplotlib")

from manaduel.analysis.card_effects import Games, estimate_effects  # noqa: E402
from manaduel.analysis.multiseed import (  # noqa: E402
    aggregate,
    format_final,
    load_curves,
    write_final_csv,
)
from manaduel.analysis.plots import (  # noqa: E402
    load_validation_csv,
    plot_card_effects,
    plot_detection,
    plot_learning_curves,
)

BOTS = ["random", "aggressive", "defensive", "greedy"]


def write_run(path, rows):
    lines = ["steps," + ",".join(BOTS) + ",mean"]
    for steps, vals in rows:
        lines.append(f"{steps}," + ",".join(f"{v:.3f}" for v in vals) + f",{sum(vals) / 4:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def run_files(tmp_path):
    write_run(tmp_path / "progress_seed0.csv", [(50000, [0.8, 0.9, 0.6, 0.6]), (100000, [0.9, 0.9, 0.7, 0.7])])
    write_run(tmp_path / "progress_seed1.csv", [(50000, [0.6, 0.7, 0.4, 0.4]), (100000, [0.7, 0.9, 0.5, 0.5]),
                                               (150000, [0.9, 0.9, 0.9, 0.9])])
    return str(tmp_path / "progress_seed*.csv")


def test_load_curves_compares_only_the_shared_part(run_files):
    curves = load_curves(run_files)
    assert curves.scores.shape == (2, 2, 4)              # the longer run is cut to 2 rows
    assert list(curves.steps) == [50000, 100000]
    assert curves.bots == tuple(BOTS)


def test_aggregate_mean_and_sample_std(run_files):
    agg = aggregate(load_curves(run_files))
    assert agg.runs == 2
    assert agg.mean[0, 0] == pytest.approx(0.7)           # (0.8 + 0.6) / 2
    assert agg.std[0, 0] == pytest.approx(np.std([0.8, 0.6], ddof=1))
    assert agg.overall_mean[0] == pytest.approx(np.mean([0.725, 0.525]))


def test_single_run_has_zero_spread(tmp_path):
    write_run(tmp_path / "progress_seed0.csv", [(50000, [0.8, 0.9, 0.6, 0.6])])
    agg = aggregate(load_curves(str(tmp_path / "progress_seed0.csv")))
    assert np.all(agg.std == 0)


def test_missing_or_empty_files_give_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_curves(str(tmp_path / "nothing*.csv"))
    (tmp_path / "progress_seed0.csv").write_text("steps,random,aggressive,defensive,greedy,mean\n")
    with pytest.raises(ValueError):
        load_curves(str(tmp_path / "progress_seed0.csv"))


def test_final_table_text_and_csv(tmp_path):
    scores = np.array([[0.9, 0.9, 0.7, 0.7], [0.8, 0.9, 0.6, 0.6]])
    paths = ["a/ppo_seed0.zip", "a/ppo_seed1.zip"]
    text = format_final(paths, BOTS, scores, 500)
    assert "ppo_seed0.zip" in text and "std across models" in text and "worst model" in text
    write_final_csv(paths, BOTS, scores, tmp_path / "f.csv")
    assert (tmp_path / "f.csv").read_text(encoding="utf-8").count("\n") == 3


def test_figures_are_written(run_files, tmp_path):
    agg = aggregate(load_curves(run_files))
    plot_learning_curves(agg, tmp_path / "curves.png")

    rng = np.random.default_rng(0)
    delta = rng.integers(-2, 3, size=(4000, 5)).astype(float)
    games = Games(("strike", "fireball", "heal", "shield", "drain"),
                  (rng.random(4000) < 0.5 + 0.05 * delta[:, 1]).astype(float), delta)
    plot_card_effects(estimate_effects(games), tmp_path / "effects.png")

    csv_path = tmp_path / "v.csv"
    csv_path.write_text(
        "card,field,old,new,relative_change,rep,power_change,z,detected,attributed\n"
        "fireball,damage,7,9,0.286,0,0.033,7.7,1,1\nshield,shield,3,4,0.333,0,0.007,1.8,0,0\n",
        encoding="utf-8")
    plot_detection(load_validation_csv(csv_path), tmp_path / "detect.png")

    for name in ("curves.png", "effects.png", "detect.png"):
        assert (tmp_path / name).stat().st_size > 8_000


def test_final_eval_runs_on_a_saved_model(tmp_path):
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    from manaduel.analysis.multiseed import final_eval
    from manaduel.rl.env import ManaDuelEnv

    env = ActionMasker(ManaDuelEnv(), lambda e: e.unwrapped.action_masks())
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu", seed=0)
    model.learn(64)
    model.save(tmp_path / "tiny")
    scores = final_eval([str(tmp_path / "tiny.zip")], n_games=4, seed=1, workers=1)
    assert scores.shape == (1, 4)
    assert np.all((scores >= 0) & (scores <= 1))
