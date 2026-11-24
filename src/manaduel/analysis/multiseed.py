"""Combine several training runs (different random seeds) into one honest summary.

A single RL training run can be lucky or unlucky, so we report the mean and the
spread (sample standard deviation) across runs, both for the learning curves
logged during training and for a final evaluation of every saved model on the
same fresh games.
"""

import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Curves:
    bots: tuple
    steps: np.ndarray     # shape (T,)
    scores: np.ndarray    # shape (runs, T, bots): win rate against each bot
    labels: tuple         # one per run (the file names)


@dataclass(frozen=True)
class Aggregate:
    bots: tuple
    steps: np.ndarray
    mean: np.ndarray      # (T, bots)
    std: np.ndarray       # (T, bots) sample standard deviation across runs (0 with a single run)
    overall_mean: np.ndarray   # (T,) mean over bots and runs
    overall_std: np.ndarray    # (T,) std across runs of the per-run mean over bots
    runs: int


def load_curves(pattern: str) -> Curves:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern!r}")
    tables = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            tables.append(list(csv.DictReader(f)))
    if any(len(t) == 0 for t in tables):
        raise ValueError("a learning-curve file has no rows yet (training still starting?)")
    bots = tuple(k for k in tables[0][0] if k not in ("steps", "mean"))
    length = min(len(t) for t in tables)  # runs may be at different points: compare the shared part
    steps = np.array([int(r["steps"]) for r in tables[0][:length]])
    scores = np.array([[[float(r[b]) for b in bots] for r in t[:length]] for t in tables])
    return Curves(bots, steps, scores, tuple(os.path.basename(p) for p in paths))


def aggregate(curves: Curves) -> Aggregate:
    runs = curves.scores.shape[0]
    ddof = 1 if runs > 1 else 0
    per_run_overall = curves.scores.mean(axis=2)  # (runs, T)
    return Aggregate(
        bots=curves.bots,
        steps=curves.steps,
        mean=curves.scores.mean(axis=0),
        std=curves.scores.std(axis=0, ddof=ddof),
        overall_mean=per_run_overall.mean(axis=0),
        overall_std=per_run_overall.std(axis=0, ddof=ddof),
        runs=runs,
    )


def _eval_checkpoint(job) -> list[float]:
    """Worker: win rate of one saved model against every bot, on a fixed set of fresh games."""
    path, n_games, seed, config_path = job
    import torch

    torch.set_num_threads(1)
    from manaduel.agents.registry import BOT_FACTORIES
    from manaduel.agents.rl_agent import PPOAgent, load_ppo_model
    from manaduel.sim.tournament import play_match, play_match_over_decks

    config = _load_config(config_path)
    model = load_ppo_model(path)
    agent = lambda s: PPOAgent(model)  # noqa: E731
    play = play_match_over_decks if config.uses_decks else play_match   # deck mode: over all deck pairings
    return [
        play(config, "ppo", agent, name, factory, n_games, seed=seed).score_a
        for name, factory in BOT_FACTORIES.items()
    ]


def _load_config(config_path):
    from pathlib import Path

    from manaduel.engine.rules import load_game_config

    return load_game_config(Path(config_path)) if config_path else load_game_config()


def _run(func, jobs, workers):
    if workers is None:
        workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    if workers <= 1:
        return [func(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, jobs, chunksize=1))


def final_eval(paths: list[str], n_games: int = 500, seed: int = 777,
               workers: int | None = None, config_path=None) -> np.ndarray:
    """Evaluate each checkpoint; returns an array of shape (checkpoints, bots)."""
    return np.array(_run(_eval_checkpoint, [(p, n_games, seed, config_path) for p in paths], workers))


def _duel_job(job) -> float:
    """Worker: how often model A beats model B (draw = half), seats swapped every game."""
    path_a, path_b, n_games, seed, config_path = job
    import torch

    torch.set_num_threads(1)
    from manaduel.agents.rl_agent import PPOAgent, load_ppo_model
    from manaduel.sim.tournament import play_match, play_match_over_decks

    config = _load_config(config_path)
    model_a, model_b = load_ppo_model(path_a), load_ppo_model(path_b)
    play = play_match_over_decks if config.uses_decks else play_match
    return play(config, "a", lambda s: PPOAgent(model_a), "b", lambda s: PPOAgent(model_b),
                n_games, seed=seed).score_a


def head_to_head(paths_a: list[str], paths_b: list[str], n_games: int = 400, seed: int = 4242,
                 workers: int | None = None, config_path=None) -> np.ndarray:
    """Every model in group A plays every model in group B. Returns shape (len(A), len(B)):
    how often the A model wins. Averaging over many model pairs is what makes a
    group-against-group claim more than one lucky pairing."""
    jobs = [(a, b, n_games, seed, config_path) for a in paths_a for b in paths_b]
    return np.array(_run(_duel_job, jobs, workers)).reshape(len(paths_a), len(paths_b))


def format_head_to_head(label_a: str, label_b: str, scores: np.ndarray, n_games: int) -> str:
    """Summary with the spread across model pairs (not across games)."""
    flat = scores.ravel()
    per_a = scores.mean(axis=1)
    lines = [
        f"{label_a} (rows) against {label_b} (columns): {n_games} games per pairing of two models",
        "win rate of the row model:",
    ]
    for i, row in enumerate(scores):
        lines.append("  model A%d " % i + "".join(f"{v:>8.1%}" for v in row) + f"   mean {row.mean():.1%}")
    ddof = 1 if len(flat) > 1 else 0
    lines.append(f"mean over all {len(flat)} model pairs: {flat.mean():.1%} "
                 f"(std across pairs {flat.std(ddof=ddof):.1%}; best A model {per_a.max():.1%}, worst {per_a.min():.1%})")
    wins = int(np.sum(flat > 0.5))
    lines.append(f"{label_a} wins the pairing in {wins} of {len(flat)} model pairs")
    return "\n".join(lines)


def format_final(paths: list[str], bots, scores: np.ndarray, n_games: int) -> str:
    runs = scores.shape[0]
    ddof = 1 if runs > 1 else 0
    lines = [
        f"Final evaluation of {runs} trained model(s): {n_games} games per bot, same fresh games for every model",
        f"{'model':<22}" + "".join(f"{b:>12}" for b in bots) + f"{'mean':>10}",
    ]
    for path, row in zip(paths, scores):
        lines.append(f"{os.path.basename(path):<22}" + "".join(f"{v:>12.1%}" for v in row) + f"{row.mean():>10.1%}")
    mean, std = scores.mean(axis=0), scores.std(axis=0, ddof=ddof)
    lines.append(f"{'mean':<22}" + "".join(f"{v:>12.1%}" for v in mean) + f"{scores.mean():>10.1%}")
    lines.append(f"{'std across models':<22}" + "".join(f"{v:>12.1%}" for v in std)
                 + f"{scores.mean(axis=1).std(ddof=ddof):>10.1%}")
    lines.append(f"{'worst model':<22}" + "".join(f"{v:>12.1%}" for v in scores.min(axis=0))
                 + f"{scores.mean(axis=1).min():>10.1%}")
    lines.append(f"{'best model':<22}" + "".join(f"{v:>12.1%}" for v in scores.max(axis=0))
                 + f"{scores.mean(axis=1).max():>10.1%}")
    return "\n".join(lines)


def write_final_csv(paths, bots, scores: np.ndarray, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", *bots, "mean"])
        for p, row in zip(paths, scores):
            w.writerow([os.path.basename(p), *[f"{v:.4f}" for v in row], f"{row.mean():.4f}"])
