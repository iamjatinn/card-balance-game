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


def _metagame(args, config, factories) -> None:
    import random

    import numpy as np

    from manaduel.analysis import metagame as mg
    from manaduel.analysis import plots
    from manaduel.engine.decks import archetype_decks, describe, random_deck

    if not config.uses_decks:
        raise SystemExit("this game has no decks; use --config configs/cards_decks.yaml")
    rng = random.Random(args.seed)
    decks = dict(archetype_decks(config))
    for k in range(args.random_decks):
        decks[f"random{k + 1}"] = random_deck(config, rng)
    print(f"Metagame with bot '{args.bot}': {len(decks)} decks, {args.n} games per pairing, seed {args.seed}\n", flush=True)
    for name, deck in decks.items():
        print(f"  {name:<10} {describe(config, deck)}")
    print()

    matrix = mg.deck_matrix(config, decks, args.bot, args.n, args.seed, args.workers)
    print(mg.format_matrix(matrix))
    print(mg.format_health(matrix, config), flush=True)

    if args.effects_n:
        from manaduel.analysis.card_effects import collect, estimate_effects

        factory = _pick(factories, args.bot)
        print(f"\nCard power, two natural experiments ({args.effects_n} games each, bot '{args.bot}'):")
        by_deck = estimate_effects(mg.collect_decks(config, factory, args.effects_n, args.seed))
        by_hand = estimate_effects(collect(config, factory, args.effects_n, args.seed))
        hand = {e.name: e for e in by_hand.effects}
        print(f"  {'card':<10}{'one more copy in the DECK':>28}{'one more copy in the OPENING HAND':>38}")
        for e in sorted(by_deck.effects, key=lambda e: -e.effect):
            h = hand[e.name]
            print(f"  {e.name:<10}{e.effect:>+10.1%} [{e.ci_low:>+6.1%},{e.ci_high:>+6.1%}]{'*' if e.flagged else ' '}"
                  f"{h.effect:>+14.1%} [{h.ci_low:>+6.1%},{h.ci_high:>+6.1%}]{'*' if h.flagged else ' '}")
        corr = float(np.corrcoef([e.effect for e in by_deck.effects], [hand[e.name].effect for e in by_deck.effects])[0, 1])
        print(f"  (* = flagged)  correlation between the two: {corr:.2f}")

    if args.build:
        names = [c.name for c in config.cards]
        field = list(decks.values())
        start = decks["balanced"]
        print(f"\nDeck builder: hill climbing from 'balanced' against all {len(field)} decks "
              f"({args.build_n} games per matchup)", flush=True)
        steps = mg.build_deck(config, start, field, args.bot, args.build_n, args.seed + 1,
                              workers=args.workers, log=lambda text: print("  " + text, flush=True))
        built = steps[-1].deck
        print(f"  built deck: {describe(config, built)}")
        check_seed = args.seed + 777
        before, after = mg.field_scores(config, args.bot, [start, built], field, args.n, check_seed,
                                        workers=args.workers or 1)
        print(f"  check on NEW games (seed {check_seed}, {args.n} per matchup): "
              f"balanced {before:.1%} -> built {after:.1%} average win rate against the field")
        with_built = dict(decks)
        with_built["built"] = built
        bigger = mg.deck_matrix(config, with_built, args.bot, args.n, check_seed, args.workers)
        print("\nWith the built deck added to the metagame:")
        print(mg.format_matrix(bigger))
        print(mg.format_health(bigger, config))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        mg.write_matrix_csv(matrix, out.with_suffix(".csv"))
        plots.plot_deck_matrix(matrix, out.with_suffix(".png"))
        print(f"\nSaved {out.with_suffix('.csv')} and {out.with_suffix('.png')}")


def _tune(args, config, factories) -> None:
    import numpy as np

    from manaduel.analysis import balance_search as bs

    print(f"Tuning with bots {args.bots}: {args.n} games per bot per configuration, steps {args.schedule}, "
          f"seed {args.seed}, up to {args.max_moves} moves\n", flush=True)
    result = bs.search(config, tuple(args.bots), args.n, args.seed, tuple(args.schedule), args.max_moves,
                       workers=args.workers, log=lambda text: print(text, flush=True))
    print("\n" + bs.format_moves(result))

    print(f"\nVerification on NEW deals (seed {args.verify_seed}, {args.verify_n} games), before vs after:")
    for bot in args.verify_bots:
        for label, cfg in (("before", config), ("after ", result.config)):
            games = collect(cfg, _pick(factories, bot), args.verify_n, args.verify_seed)
            report = estimate_effects(games)
            flagged = ", ".join(f"{e.name} {e.effect:+.1%}" for e in report.flagged()) or "none"
            print(f"  [{label}] {bot:<10} draws {np.mean(games.score0 == 0.5):5.1%}  "
                  f"first player {report.seat0_baseline:5.1%}  flagged: {flagged}")
            if label == "after ":
                print(format_effects(report))
    if args.out:
        config_to_yaml(result.config, args.out)
        print(f"\nSaved the tuned config to {args.out}")


def _realdata(args, config, factories) -> None:
    import time

    from manaduel.analysis import plots, real_data

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    started = time.time()

    def progress(read, used):
        print(f"  read {read:>9,} games ({used:,} usable)  {time.time() - started:5.0f}s", flush=True)

    print(f"Reading {path.name} (this streams the file in chunks)\n", flush=True)
    sums = real_data.accumulate(path, groups=args.groups, max_rows=args.max_rows, progress=progress)
    report = real_data.estimate(sums, min_effect=args.min_effect)
    print()
    print(real_data.format_report(report))

    split = real_data.split_half(sums, seed=args.seed)
    comp = real_data.comparison(report)
    print("\nDoes the estimate replicate? Two random halves of the drafts:")
    print(f"  correlation of the two halves' effects: {split['correlation']:.2f}")
    print(f"  same sign in both halves (all cards): {split['sign_agreement_all']:.0%}")
    print(f"  cards clearly non-zero in BOTH halves: {split['cards_clear_in_both_halves']}, "
          f"of which same sign: {split['sign_agreement_clear']:.0%}")
    print("\nAgainst the usual biased 'games in hand' statistic:")
    print(f"  correlation {comp['pearson']:.2f} (rank {comp['spearman']:.2f}); its spread is "
          f"{comp['naive_scale']:.1f}x larger; same sign for {comp['sign_agreement']:.0%} of cards")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        real_data.write_csv(report, out.with_suffix(".csv"))
        plots.plot_real_vs_naive(report, out.with_suffix(".png"))
        print(f"\nSaved {out.with_suffix('.csv')} and {out.with_suffix('.png')}")


def _headtohead(args, config, factories) -> None:
    import glob

    from manaduel.analysis.multiseed import format_head_to_head, head_to_head

    a, b = sorted(glob.glob(args.a)), sorted(glob.glob(args.b))
    if not a or not b:
        raise SystemExit(f"no models match {args.a!r} or {args.b!r}")
    print(f"{len(a)} models vs {len(b)} models, {args.n} games per pairing, seed {args.seed}\n", flush=True)
    scores = head_to_head(a, b, args.n, args.seed, args.workers, config_path=args.config)
    print(format_head_to_head(args.label_a, args.label_b, scores, args.n))


def _multiseed(args, config, factories) -> None:
    import glob

    paths = sorted(glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"no models match {args.checkpoints!r}")
    from manaduel.analysis.multiseed import final_eval, format_final, write_final_csv

    print(f"Evaluating {len(paths)} models, {args.n} games per bot, seed {args.seed}\n", flush=True)
    scores = final_eval(paths, args.n, args.seed, args.workers, config_path=args.config)
    bots = list(BOT_FACTORIES)
    print(format_final(paths, bots, scores, args.n))
    if args.csv:
        write_final_csv(paths, bots, scores, args.csv)
        print(f"\nSaved {args.csv}")


def _figures(args, config, factories) -> None:
    from manaduel.analysis import plots
    from manaduel.analysis.multiseed import aggregate, load_curves

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    if args.curves:
        plots.plot_learning_curves(aggregate(load_curves(args.curves)), out / "learning_curves.png")
        made.append("learning_curves.png")
    if args.validation:
        plots.plot_detection(plots.load_validation_csv(args.validation), out / "detection.png")
        made.append("detection.png")
    if args.effects_bot:
        games = collect(config, _pick(factories, args.effects_bot), args.effects_n, 0)
        plots.plot_card_effects(estimate_effects(games), out / "card_effects.png")
        made.append("card_effects.png")
    print("Wrote " + ", ".join(str(out / m) for m in made) if made else "Nothing to draw: give --curves, --validation or --effects-bot")


def _view(args, config, factories) -> None:
    # Imported here so the rest of the project works without pygame installed.
    from manaduel.viewer import pygame_viewer

    if not Path(args.replay).exists():
        raise SystemExit(f"no such replay file: {args.replay}  (create one with: watch --save)")
    if args.png:
        pygame_viewer.save_png(args.png, args.replay, args.beat, args.progress)
        print(f"Saved {args.png}")
    elif args.gif:
        frames = pygame_viewer.save_gif(args.gif, args.replay, speed=args.speed)
        print(f"Saved {args.gif} ({frames} frames)")
    else:
        pygame_viewer.run_window(args.replay, args.speed)


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

    mgp = sub.add_parser("metagame", help="deck-vs-deck analysis in deck mode (use --config configs/cards_decks.yaml)")
    mgp.add_argument("--bot", default="greedy")
    mgp.add_argument("--n", type=int, default=2000, help="games per deck pairing")
    mgp.add_argument("--seed", type=int, default=0)
    mgp.add_argument("--random-decks", type=int, default=4, help="extra random decks in the field")
    mgp.add_argument("--effects-n", type=int, default=20000, help="games for the card-power comparison (0 to skip)")
    mgp.add_argument("--build", action="store_true", help="also run the deck builder")
    mgp.add_argument("--build-n", type=int, default=300, help="games per matchup while building")
    mgp.add_argument("--workers", type=int, default=None)
    mgp.add_argument("--out", help="write OUT.csv and OUT.png")

    tn = sub.add_parser("tune", help="coarse-to-fine balance search over many card and rule settings")
    tn.add_argument("--bots", nargs="+", default=["greedy", "defensive"], help="bots the objective averages over")
    tn.add_argument("--n", type=int, default=8000, help="mirror games per bot per configuration")
    tn.add_argument("--seed", type=int, default=0)
    tn.add_argument("--schedule", nargs="+", type=int, default=[4, 2, 1], help="step sizes, coarse to fine")
    tn.add_argument("--max-moves", type=int, default=25)
    tn.add_argument("--workers", type=int, default=None)
    tn.add_argument("--verify-bots", nargs="+", default=["greedy", "defensive"])
    tn.add_argument("--verify-n", type=int, default=20000)
    tn.add_argument("--verify-seed", type=int, default=99991)
    tn.add_argument("--out", help="write the tuned config to this YAML file")

    rd = sub.add_parser("realdata", help="card power from a real 17Lands game-data file")
    rd.add_argument("file", help="game_data_public.<SET>.PremierDraft.csv.gz")
    rd.add_argument("--groups", type=int, default=200, help="draft groups for the jackknife")
    rd.add_argument("--max-rows", type=int, default=None, help="only read this many rows (for a quick test)")
    rd.add_argument("--min-effect", type=float, default=0.01, help="smallest effect worth flagging")
    rd.add_argument("--seed", type=int, default=0)
    rd.add_argument("--out", help="write OUT.csv and OUT.png")

    hh = sub.add_parser("headtohead", help="every model in group A plays every model in group B")
    hh.add_argument("a", help='glob for group A models, e.g. "checkpoints/selfplay_seed*.zip"')
    hh.add_argument("b", help="glob for group B models")
    hh.add_argument("--label-a", default="A")
    hh.add_argument("--label-b", default="B")
    hh.add_argument("--n", type=int, default=400, help="games per pairing of two models")
    hh.add_argument("--seed", type=int, default=4242)
    hh.add_argument("--workers", type=int, default=None)

    ms = sub.add_parser("multiseed", help="evaluate several trained models on the same fresh games")
    ms.add_argument("checkpoints", help='glob such as "checkpoints/ppo_seed*.zip"')
    ms.add_argument("--n", type=int, default=500, help="games per bot per model")
    ms.add_argument("--seed", type=int, default=777)
    ms.add_argument("--workers", type=int, default=None)
    ms.add_argument("--csv", help="also write the table to this CSV file")

    fg = sub.add_parser("figures", help="draw README figures from saved data")
    fg.add_argument("--out", default="docs", help="folder for the PNG files")
    fg.add_argument("--curves", help='learning-curve CSVs, e.g. "runs/progress_seed*.csv"')
    fg.add_argument("--validation", help="validation CSV from the validate command")
    fg.add_argument("--effects-bot", help="measure card power with this bot and plot it")
    fg.add_argument("--effects-n", type=int, default=20000)

    vw = sub.add_parser("view", help="watch a saved replay in a window (needs pygame)")
    vw.add_argument("replay", help="a replay JSON from: watch --save")
    vw.add_argument("--speed", type=float, default=1.0)
    vw.add_argument("--png", help="instead of a window, save one frame to this PNG")
    vw.add_argument("--beat", type=int, help="which card play to render with --png (default: middle)")
    vw.add_argument("--progress", type=float, default=0.85, help="how far into the beat, 0 to 1 (with --png)")
    vw.add_argument("--gif", help="instead of a window, export the whole game as an animated GIF")

    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_game_config(Path(args.config)) if args.config else load_game_config()
    factories = _factories(getattr(args, "ppo", None))
    handlers = {"watch": _watch, "tournament": _tournament, "effects": _effects,
                "sweep": _sweep, "autobalance": _autobalance, "validate": _validate, "view": _view,
                "multiseed": _multiseed, "headtohead": _headtohead, "figures": _figures, "realdata": _realdata,
                "tune": _tune, "metagame": _metagame}
    handlers[args.command](args, config, factories)


if __name__ == "__main__":
    main()
