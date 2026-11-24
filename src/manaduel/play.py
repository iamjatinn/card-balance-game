"""Play the game yourself in the terminal, against any bot.

    python -m manaduel.cli play --bot greedy
    python -m manaduel.cli play --bot greedy --config configs/cards_extended.yaml

Every finished game is saved as a replay (watch it with the `view` command) and one line is
added to runs/human_games.csv, so a few people playing a few games each gives real
evidence about how the cards feel, which the bot-based tools cannot give.
"""

import csv
import datetime
from pathlib import Path

from manaduel.engine.game import GameResult, _event
from manaduel.engine.rules import legal_actions, new_game, step
from manaduel.engine.state import is_terminal, winner
from manaduel.sim.replay import save_replay
from manaduel.sim.stats import wilson_interval

LOG_COLUMNS = ["date", "config", "bot", "human_seat", "seed", "result", "turns"]


def describe_card(card) -> str:
    """One short line saying what a card does, built from its stats."""
    parts = []
    if card.damage:
        parts.append(f"deal {card.damage}")
    if card.combo:
        parts.append(f"+{card.combo} if you already played a card this turn")
    if card.heal:
        parts.append(f"heal {card.heal}")
    if card.shield:
        parts.append(f"gain {card.shield} shield")
    if card.poison:
        parts.append(f"poison {card.poison}")
    if card.draw:
        parts.append(f"draw {card.draw}")
    return ", ".join(parts) or "no effect"


def show_state(state, me: int, config, names, write) -> None:
    mine, theirs = state.players[me], state.players[1 - me]
    extra = lambda p: f"  poison {p.poison}" if config.uses_poison else ""
    write("")
    write(f"--- turn {state.turn} ---")
    write(f"  You:      {mine.hp:>3} HP  shield {mine.shield}  mana {mine.mana}{extra(mine)}")
    write(f"  {names[1 - me]}: {theirs.hp:>3} HP  shield {theirs.shield}  holds {sum(theirs.hand)} cards{extra(theirs)}")
    hand = [f"{c.name} x{n}" for c, n in zip(config.cards, mine.hand) if n]
    write(f"  Your hand: {', '.join(hand) if hand else '(empty)'}")


def choose_action(state, config, read, write, me: int) -> int:
    legal = legal_actions(state, config)
    options = [a for a in legal if a != config.end_turn]
    write("  Play a card (number) or 0 to end your turn:")
    for number, action in enumerate(options, start=1):
        card = config.cards[action]
        write(f"    {number}) {card.name} (cost {card.cost}): {describe_card(card)}")
    write("    0) end turn")
    while True:
        text = read("  > ").strip().lower()
        if text in ("0", "end", "e", ""):
            return config.end_turn
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1]
        write("  Please type one of the numbers above.")


def play_human(config, bot, human_seat: int = 0, seed: int = 0, bot_name: str = "bot",
               read=None, write=print, decks=None) -> GameResult:
    """Play one game. `read`/`write` can be replaced (that is how the tests play it)."""
    if read is None:
        read = input     # looked up now, not when the function was defined
    names = ("You", bot_name) if human_seat == 0 else (bot_name, "You")
    state = new_game(config, seed, decks)
    start_hands = [list(p.hand) for p in state.players]
    events = []
    write(f"You are player {human_seat + 1} and move {'first' if human_seat == 0 else 'second'}. "
          f"Opponent: {bot_name}. Reduce its HP to 0 to win.")
    announced = None
    while not is_terminal(state, config.rules.max_turns):
        if state.current == human_seat:
            if announced != state.turn:
                show_state(state, human_seat, config, names, write)
                announced = state.turn
            action = choose_action(state, config, read, write, human_seat)
        else:
            action = bot.act(state, legal_actions(state, config), config)
            write(f"  {bot_name} {'plays ' + config.cards[action].name if action != config.end_turn else 'ends its turn'}")
        after = step(state, action, config)
        events.append(_event(state, action, after, config))
        if state.current == human_seat and action != config.end_turn:
            show_state(after, human_seat, config, names, write)   # show the result of each play
            announced = after.turn
        state = after
    w = winner(state, config.rules.max_turns)
    write("")
    write("You win!" if w == human_seat else "A draw." if w is None else f"{bot_name} wins.")
    return GameResult(w, min(state.turn, config.rules.max_turns), events, start_hands,
                      [list(p.deck) for p in state.players] if config.uses_decks else [])


def record(result, config, human_seat, bot_name, seed, config_name, log_path="runs/human_games.csv",
           replay_dir="replays") -> Path:
    """Save the replay and append one row to the games log; returns the replay path."""
    names = ("You", bot_name) if human_seat == 0 else (bot_name, "You")
    replay = Path(replay_dir) / f"human_vs_{bot_name}_seed{seed}.json"
    save_replay(replay, result, config, names, seed=seed)
    outcome = "draw" if result.winner is None else ("win" if result.winner == human_seat else "loss")
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(LOG_COLUMNS)
        w.writerow([datetime.date.today().isoformat(), config_name, bot_name, human_seat, seed, outcome, result.turns])
    return replay


def summarize(log_path="runs/human_games.csv") -> str:
    """Win rate of the human players against each bot, with a 95% interval."""
    log = Path(log_path)
    if not log.exists():
        return f"no games logged yet in {log_path}"
    rows = list(csv.DictReader(open(log, encoding="utf-8")))
    lines = [f"{len(rows)} games logged in {log_path}",
             f"{'config':<28}{'bot':<12}{'games':>6}{'you win':>10}{'95% interval':>18}"]
    groups = {}
    for r in rows:
        groups.setdefault((r["config"], r["bot"]), []).append(r["result"])
    for (config_name, bot), results in sorted(groups.items()):
        points = results.count("win") + 0.5 * results.count("draw")
        lo, hi = wilson_interval(points, len(results))
        lines.append(f"{config_name:<28}{bot:<12}{len(results):>6}{points / len(results):>10.0%}"
                     f"{f'[{lo:.0%}, {hi:.0%}]':>18}")
    return "\n".join(lines)
