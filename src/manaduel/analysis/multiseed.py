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
    path, n_games, seed = job
    import torch

    torch.set_num_threads(1)
    from manaduel.agents.registry import BOT_FACTORIES
    from manaduel.agents.rl_agent import PPOAgent, load_ppo_model
    from manaduel.engine.rules import load_game_config
    from manaduel.sim.tournament import play_match

    config = load_game_config()
    model = load_ppo_model(path)
    agent = lambda s: PPOAgent(model)  # noqa: E731
    return [
        play_match(config, "ppo", agent, name, factory, n_games, seed=seed).score_a
        for name, factory in BOT_FACTORIES.items()
    ]


def final_eval(paths: list[str], n_games: int = 500, seed: int = 777,
               workers: int | None = None) -> np.ndarray:
    """Evaluate each checkpoint; returns an array of shape (checkpoints, bots)."""
    jobs = [(p, n_games, seed) for p in paths]
    if workers is None:
        workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    if workers <= 1:
        rows = [_eval_checkpoint(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_eval_checkpoint, jobs, chunksize=1))
    return np.array(rows)


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
