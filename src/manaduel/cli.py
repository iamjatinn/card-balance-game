"""Command line entry point.   Examples:
    python -m manaduel.cli watch --a aggressive --b random --seed 1 --save
    python -m manaduel.cli tournament --n 1000
    python -m manaduel.cli tournament --n 500 --ppo checkpoints/ppo_v1.zip
"""

import argparse
import sys
from pathlib import Path

from manaduel.agents.registry import BOT_FACTORIES
from manaduel.analysis.autobalance import autobalance, config_to_yaml, format_trajectory
from manaduel.analysis.card_effects import collect, estimate_effects
from manaduel.analysis.card_effects import format_report as format_effects
from manaduel.analysis.sweep import format_table, parse_set, run_sweep, write_csv
from manaduel.analysis.validate import format_validation, run_validation
from manaduel.analysis.validate import write_csv as write_validation_csv
from manaduel.engine.game import play_game
from manaduel.engine.rules import load_game_config
from manaduel.sim.replay import save_replay
from manaduel.sim.tournament import format_report, round_robin
from manaduel.viewer.terminal import render_game


def _factories(ppo_path):
    """The hand-written bots, plus 'ppo' if a trained model file is given."""
    factories = dict(BOT_FACTORIES)
    if ppo_path:
        from manaduel.agents.rl_agent import PPOAgent, load_ppo_model

        model = load_ppo_model(ppo_path)
        factories["ppo"] = lambda seed: PPOAgent(model)
    return factories


def _pick(factories, name):
    if name not in factories:
        raise SystemExit(f"unknown bot '{name}'. choices: {', '.join(factories)}")
    return factories[name]


def _watch(args, config, factories) -> None:
    agents = (_pick(factories, args.a)(args.seed), _pick(factories, args.b)(args.seed + 1))
    names = (f"A({args.a})", f"B({args.b})")
    result = play_game(config, agents, seed=args.seed)
    print(render_game(result, config, names))
    if args.save:
        path = Path("replays") / f"{args.a}_vs_{args.b}_seed{args.seed}.json"
        save_replay(path, result, config, names, seed=args.seed)
        print(f"\nSaved replay to {path}")


def _tournament(args, config, factories) -> None:
    names = args.agents or list(factories)
    chosen = {n: _pick(factories, n) for n in names}
    print(f"Round robin: {len(names)} bots, {args.n} games per pairing, seed {args.seed}\n")
    results = round_robin(config, chosen, args.n, args.seed)
    print(format_report(results, names))


def _effects(args, config, factories) -> None:
    factory = _pick(factories, args.bot)
    print(f"Mirror matches of '{args.bot}': {args.n} games, seed {args.seed}\n")
    games = collect(config, factory, args.n, args.seed)
    print(format_effects(estimate_effects(games)))


def _validate(args, config, factories) -> None:
    factory = _pick(factories, args.bot)
    print(f"Validation with '{args.bot}': deltas {args.deltas}, {args.n} games per measurement, "
          f"{args.reps} rep(s), {args.null_reps} no-change repeats, seed {args.seed}\n", flush=True)
    result = run_validation(config, factory, tuple(args.deltas), args.n, args.baseline_n,
                            args.reps, args.null_reps, args.seed, workers=args.workers)
    print(format_validation(result))
    if args.csv:
        write_validation_csv(result, args.csv)
        print(f"\nSaved {args.csv}")


def _autobalance(args, config, factories) -> None:
    factory = _pick(factories, args.bot)
    print(f"Autobalance with '{args.bot}': {args.n} games per measurement, "
          f"up to {args.max_steps} steps, seed {args.seed}\n", flush=True)
    result = autobalance(config, factory, args.n, args.seed, args.max_steps,
                         log=lambda text: print(text, flush=True))
    print("\n" + format_trajectory(result))
    print("\nFinal card power on the tuning games:")
    print(format_effects(result.steps[-1].report))
    if args.verify_bot:
        verify = _pick(factories, args.verify_bot)
        print(f"\nCheck on NEW games with a different bot ('{args.verify_bot}', seed {args.verify_seed}):")
        games = collect(result.config, verify, args.n, args.verify_seed)
        print(format_effects(estimate_effects(games)))
    if args.out:
        config_to_yaml(result.config, args.out)
        print(f"\nSaved the balanced config to {args.out}")


def _sweep(args, config, factories) -> None:
    try:
        grid = dict(parse_set(spec) for spec in args.set)
    except ValueError as error:
        raise SystemExit(str(error))
    chosen = {n: _pick(factories, n) for n in args.bots}
    print(f"Sweep: {args.n} mirror games per setting per bot, seed {args.seed}\n")
    try:
        rows = run_sweep(config, grid, chosen, args.n, args.seed)
    except ValueError as error:
        raise SystemExit(str(error))
    print(format_table(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nSaved {args.csv}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="manaduel")
    parser.add_argument("--config", help="card/rule file to use instead of configs/cards.yaml "
                                         "(goes before the command)")
    sub = parser.add_subparsers(dest="command", required=True)

    watch = sub.add_parser("watch", help="play one game and print it")
    watch.add_argument("--a", default="random", help="bot for player A")
    watch.add_argument("--b", default="random", help="bot for player B")
    watch.add_argument("--seed", type=int, default=0)
    watch.add_argument("--save", action="store_true", help="save a replay JSON to replays/")
    watch.add_argument("--ppo", help="path to a trained model; makes the bot 'ppo' available")

    tour = sub.add_parser("tournament", help="round robin between bots")
    tour.add_argument("--n", type=int, default=1000, help="games per pairing")
    tour.add_argument("--seed", type=int, default=0)
    tour.add_argument("--agents", nargs="+", help="which bots to include (default: all)")
    tour.add_argument("--ppo", help="path to a trained model; adds a 'ppo' bot")

    eff = sub.add_parser("effects", help="estimate how strong each card is (mirror matches)")
    eff.add_argument("--bot", default="greedy", help="bot that plays both seats")
    eff.add_argument("--n", type=int, default=20000, help="number of games")
    eff.add_argument("--seed", type=int, default=0)
    eff.add_argument("--ppo", help="path to a trained model; makes 'ppo' available")

    sw = sub.add_parser("sweep", help="try a setting at several values and measure the effect")
    sw.add_argument("--set", action="append", required=True, metavar="KEY=V1,V2,...",
                    help="e.g. rules.max_shield=3,4,5,6 or cards.heal.heal=3,4 (repeat for a grid)")
    sw.add_argument("--bots", nargs="+", default=["greedy", "defensive"])
    sw.add_argument("--n", type=int, default=2000, help="mirror games per setting per bot")
    sw.add_argument("--seed", type=int, default=0)
    sw.add_argument("--csv", help="also write the results to this CSV file")

    ab = sub.add_parser("autobalance", help="suggest card changes that remove over/underpowered cards")
    ab.add_argument("--bot", default="greedy", help="bot used for tuning")
    ab.add_argument("--n", type=int, default=10000, help="games per measurement")
    ab.add_argument("--seed", type=int, default=0)
    ab.add_argument("--max-steps", type=int, default=15)
    ab.add_argument("--verify-bot", default="defensive", help="different bot for the final check")
    ab.add_argument("--verify-seed", type=int, default=99991, help="new seed for the final check")
    ab.add_argument("--out", help="write the balanced config to this YAML file")

    va = sub.add_parser("validate", help="plant known card changes and measure whether the tool detects them")
    va.add_argument("--bot", default="greedy")
    va.add_argument("--deltas", nargs="+", type=int, default=[1, 2, 4, -1, -2],
                    help="changes to each card's main stat (default: +1 +2 +4 -1 -2)")
    va.add_argument("--n", type=int, default=10000, help="games per measurement")
    va.add_argument("--baseline-n", type=int, default=20000, help="games for the baseline measurement")
    va.add_argument("--reps", type=int, default=2, help="independent repeats per planted change")
    va.add_argument("--null-reps", type=int, default=20, help="no-change repeats for the false-alarm rate")
    va.add_argument("--seed", type=int, default=0)
    va.add_argument("--workers", type=int, default=None, help="parallel processes (default: most cores)")
    va.add_argument("--csv", help="also write the trials to this CSV file")

    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_game_config(Path(args.config)) if args.config else load_game_config()
    factories = _factories(getattr(args, "ppo", None))
    handlers = {"watch": _watch, "tournament": _tournament, "effects": _effects,
                "sweep": _sweep, "autobalance": _autobalance, "validate": _validate}
    handlers[args.command](args, config, factories)


if __name__ == "__main__":
    main()
