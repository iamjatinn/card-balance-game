"""Driver for balancing with learned agents: train, measure, correct, repeat.

    python -m manaduel.analysis.agent_rounds --start configs/cards_extended_balanced.yaml --rounds 4 --out runs/agent_balance

Each round trains `--seeds` PPO agents on the current card set (in parallel), measures card power with every
agent in mirror games, writes the round's table, and applies analysis.agent_balance.correct for the next round.
Round 0 can reuse agents already trained on the start config (--reuse "checkpoints/extbal_seed{seed}.zip").
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from manaduel.analysis import agent_balance
from manaduel.analysis.autobalance import config_to_yaml
from manaduel.engine.rules import load_game_config


def _effects_job(job):
    model_path, config_path, n_games, seed = job
    import torch

    torch.set_num_threads(1)
    from manaduel.agents.rl_agent import PPOAgent, load_ppo_model
    from manaduel.analysis.card_effects import collect, estimate_effects

    config = load_game_config(Path(config_path))
    model = load_ppo_model(model_path)
    report = estimate_effects(collect(config, lambda s: PPOAgent(model), n_games, seed))
    return [e.effect for e in report.effects], report.seat0_baseline


def train_agents(config_path: Path, out_dir: Path, seeds, steps: int) -> list[Path]:
    env = dict(os.environ, OMP_NUM_THREADS="1")
    procs, paths = [], []
    for s in seeds:
        path = out_dir / f"agent_seed{s}"
        cmd = [sys.executable, "-m", "manaduel.rl.train", "--config", str(config_path), "--steps", str(steps),
               "--seed", str(s), "--out", str(path), "--log", str(out_dir / f"curve_seed{s}.csv")]
        log = open(out_dir / f"train_seed{s}.log", "w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        paths.append(path.with_suffix(".zip"))
    for p, log in procs:
        if p.wait() != 0:
            raise RuntimeError("a training run failed; see the train_seed*.log files")
        log.close()
    return paths


def measure(config_path: Path, models, n_games: int, seed: int):
    with ProcessPoolExecutor(max_workers=len(models)) as pool:
        out = list(pool.map(_effects_job, [(str(m), str(config_path), n_games, seed) for m in models]))
    return np.array([o[0] for o in out]), [o[1] for o in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="the card set to start from (YAML)")
    ap.add_argument("--rounds", type=int, default=4, help="correction rounds after round 0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--n", type=int, default=8000, help="mirror games per agent for the card-power measurement")
    ap.add_argument("--measure-seed", type=int, default=99991)
    ap.add_argument("--reuse", help='round-0 agents already trained on --start, e.g. "checkpoints/extbal_seed{seed}.zip"')
    ap.add_argument("--out", default="runs/agent_balance")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_game_config(Path(args.start))
    names = [c.name for c in config.cards]
    history = []
    for k in range(args.rounds + 1):
        round_dir = out / f"round{k}"
        round_dir.mkdir(exist_ok=True)
        config_path = round_dir / "cards.yaml"
        config_to_yaml(config, config_path)
        if k == 0 and args.reuse:
            models = [Path(args.reuse.format(seed=s)) for s in args.seeds]
        else:
            models = train_agents(config_path, round_dir, args.seeds, args.steps)
        effects, first = measure(config_path, models, args.n, args.measure_seed)
        summary = agent_balance.summarise(effects, names)
        summary.update(round=k, first_player=[float(x) for x in first],
                       effects_per_agent=effects.tolist(), config=str(config_path))
        history.append(summary)
        (round_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"round {k}: worst {summary['worst_card']} {summary['mean_effect'][summary['worst_card']]:+.1%}, "
              f"rms {summary['rms']:.1%}, cards off by 3+ points: {summary['off_cards']}, "
              f"first player {np.mean(first):.1%}", flush=True)
        if k == args.rounds or not summary["off_cards"]:
            break
        config, changes = agent_balance.correct(config, effects, names)
        for c in changes:
            print(f"   {c.card} ({c.effect:+.1%}): {c.description}", flush=True)
    best = min(history, key=lambda h: h["rms"])
    print(f"\nbest round by RMS card power: {best['round']} (rms {best['rms']:.1%}, worst {best['worst_card']} "
          f"{best['mean_effect'][best['worst_card']]:+.1%}); config {best['config']}", flush=True)
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
