"""Driver for balancing with learned agents: train, measure, correct, repeat.

    python -m manaduel.analysis.agent_rounds --start configs/cards_extended_balanced.yaml --rounds 4 --out runs/agent_balance
    python -m manaduel.analysis.agent_rounds --start configs/cards_decks_final.yaml --rounds 5 --out runs/deck_balance \
        --min-effect 0.015 --gain 0.10 --metagame-n 500

Each round trains `--seeds` PPO agents on the current card set (in parallel), measures card power with every
agent in mirror games, writes the round's table, and applies analysis.agent_balance.correct for the next round.
Round 0 can reuse agents already trained on the start config (--reuse "checkpoints/extbal_seed{seed}.zip").

DECK MODE (the start config has deck_size): agents are trained on the four archetype decks plus `--random-decks`
random decks; card power is the "one more copy in the deck" effect (both players get random decks, see
analysis/metagame.py); and, with --metagame-n, each round also plays an eight-deck round robin with the first
agent to see how dominant the best deck is. Those effects per copy are much smaller than the opening-hand effects,
so use a smaller --min-effect and --gain there.
"""

import argparse
import json
import os
import random
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
    factory = lambda s: PPOAgent(model)  # noqa: E731
    if config.uses_decks:                       # deck mode: effect of one more copy in the deck, random decks
        from manaduel.analysis.metagame import collect_decks

        games = collect_decks(config, factory, n_games, seed)
    else:
        games = collect(config, factory, n_games, seed)
    report = estimate_effects(games)
    return [e.effect for e in report.effects], report.seat0_baseline


def train_agents(config_path: Path, out_dir: Path, seeds, steps: int, random_decks: int = 0) -> list[Path]:
    env = dict(os.environ, OMP_NUM_THREADS="1")
    procs, paths = [], []
    for s in seeds:
        path = out_dir / f"agent_seed{s}"
        cmd = [sys.executable, "-m", "manaduel.rl.train", "--config", str(config_path), "--steps", str(steps),
               "--seed", str(s), "--out", str(path), "--log", str(out_dir / f"curve_seed{s}.csv")]
        if random_decks:
            cmd += ["--random-decks", str(random_decks)]
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


def metagame_check(config, model, n_games: int, seed: int, workers: int, report_path: Path) -> dict:
    """Deck round robin (four archetypes + four fixed random decks) played by one trained agent:
    how dominant is the best deck, how many decks does the equilibrium need, how many cycles are there."""
    from manaduel.analysis import metagame as mg
    from manaduel.engine.decks import archetype_decks, random_deck

    rng = random.Random(0)
    decks = dict(archetype_decks(config))
    for k in range(4):
        decks[f"random{k + 1}"] = random_deck(config, rng)
    matrix = mg.deck_matrix(config, decks, f"ppo:{model}", n_games, seed, workers)
    n = len(matrix.names)
    average = (matrix.win.sum(axis=1) - 0.5) / (n - 1)            # each deck's mean win rate against the others
    mix = mg.equilibrium(matrix.win)
    report_path.write_text(mg.format_matrix(matrix) + "\n" + mg.format_health(matrix, config) + "\n", encoding="utf-8")
    return dict(best_deck=matrix.names[int(np.argmax(average))], best_average=float(average.max()),
                dominant=mg.dominant_decks(matrix), cycles=len(mg.count_cycles(mg.beats(matrix))),
                equilibrium_decks=int(np.sum(mix > 0.02)), n_decks=n,
                equilibrium_exploitability=float(mg.exploitability(matrix.win, mix)))


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
    ap.add_argument("--gain", type=float, default=0.25, help="effect that would scale a card by 100%% (bigger = gentler)")
    ap.add_argument("--max-change", type=float, default=0.4, help="largest relative change per round")
    ap.add_argument("--min-effect", type=float, default=0.03, help="smallest effect that moves a card")
    ap.add_argument("--random-decks", type=int, default=12, help="deck mode: random decks added to the training pool")
    ap.add_argument("--metagame-n", type=int, default=0, help="deck mode: games per deck pairing in the per-round check (0 = skip)")
    ap.add_argument("--workers", type=int, default=10, help="processes for the metagame check")
    ap.add_argument("--select", choices=["rms", "metagame"], help="how to pick the best round (default: rms, or "
                    "metagame in deck mode when --metagame-n is set)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_game_config(Path(args.start))
    names = [c.name for c in config.cards]
    deck_mode = config.uses_decks
    select = args.select or ("metagame" if deck_mode and args.metagame_n else "rms")
    history = []
    for k in range(args.rounds + 1):
        round_dir = out / f"round{k}"
        round_dir.mkdir(exist_ok=True)
        config_path = round_dir / "cards.yaml"
        config_to_yaml(config, config_path)
        if k == 0 and args.reuse:
            models = [Path(args.reuse.format(seed=s)) for s in args.seeds]
        else:
            models = train_agents(config_path, round_dir, args.seeds, args.steps,
                                  args.random_decks if deck_mode else 0)
        effects, first = measure(config_path, models, args.n, args.measure_seed)
        summary = agent_balance.summarise(effects, names, args.min_effect)
        summary.update(round=k, first_player=[float(x) for x in first],
                       effects_per_agent=effects.tolist(), config=str(config_path))
        line = (f"round {k}: worst {summary['worst_card']} {summary['mean_effect'][summary['worst_card']]:+.1%}, "
                f"rms {summary['rms']:.1%}, cards off by {args.min_effect * 100:.1f}+ points: {summary['off_cards']}, "
                f"first player {np.mean(first):.1%}")
        if deck_mode and args.metagame_n:
            meta = metagame_check(config, models[0], args.metagame_n, args.measure_seed, args.workers,
                                  round_dir / "metagame.txt")
            summary["metagame"] = meta
            line += (f"; metagame (agent 0): best deck {meta['best_deck']} {meta['best_average']:.1%} against the others, "
                     f"equilibrium uses {meta['equilibrium_decks']} of {meta['n_decks']} decks, {meta['cycles']} cycles")
        history.append(summary)
        (round_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(line, flush=True)
        if k == args.rounds or not summary["off_cards"]:
            break
        config, changes = agent_balance.correct(config, effects, names, args.gain, args.max_change, args.min_effect)
        for c in changes:
            print(f"   {c.card} ({c.effect:+.1%}): {c.description}", flush=True)
    if select == "metagame":
        best = min(history, key=lambda h: (h["metagame"]["best_average"], h["rms"]))
        label = "the lowest best-deck win rate"
    else:
        best = min(history, key=lambda h: h["rms"])
        label = "RMS card power"
    print(f"\nbest round by {label}: {best['round']} (rms {best['rms']:.1%}, worst {best['worst_card']} "
          f"{best['mean_effect'][best['worst_card']]:+.1%}); config {best['config']}", flush=True)
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
