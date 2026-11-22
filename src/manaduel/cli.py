"""Command line entry point.   Examples:
    python -m manaduel.cli watch --a aggressive --b random --seed 1 --save
    python -m manaduel.cli tournament --n 1000
    python -m manaduel.cli tournament --n 500 --ppo checkpoints/ppo_v1.zip
"""

import argparse
import sys
from pathlib import Path

from manaduel.agents.registry import BOT_FACTORIES
from manaduel.analysis.card_effects import collect, estimate_effects
from manaduel.analysis.card_effects import format_report as format_effects
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="manaduel")
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

    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_game_config()
    factories = _factories(args.ppo)
    handlers = {"watch": _watch, "tournament": _tournament, "effects": _effects}
    handlers[args.command](args, config, factories)


if __name__ == "__main__":
    main()
