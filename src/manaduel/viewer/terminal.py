"""Plain-text viewer: print a finished game turn by turn."""

from manaduel.engine.game import GameResult
from manaduel.engine.rules import GameConfig


def bar(value: int, maximum: int, width: int = 20) -> str:
    filled = max(0, min(width, round(width * value / maximum)))
    return "#" * filled + "-" * (width - filled)


def render_game(result: GameResult, config: GameConfig, names: tuple[str, str]) -> str:
    lines: list[str] = []
    last_turn = None
    for e in result.events:
        if e["action"] == "end_turn":
            continue  # turn changes are shown by the blank line between turns
        if last_turn is not None and e["turn"] != last_turn:
            lines.append("")
        last_turn = e["turn"]
        status = " | ".join(
            f"{names[i]} {e['hp'][i]:>3}hp {e['shield'][i]}sh {e['mana'][i]}m"
            for i in range(2)
        )
        lines.append(f"T{e['turn']:02d} {names[e['player']]:<12} plays {e['action']:<9} -> {status}")

    final = result.events[-1] if result.events else None
    lines.append("")
    if final:
        for i in range(2):
            lines.append(
                f"{names[i]:<12} [{bar(final['hp'][i], config.rules.max_hp)}] "
                f"{max(final['hp'][i], 0)}/{config.rules.max_hp}"
            )
    if result.winner is None:
        lines.append(f"Draw after {result.turns} turns.")
    else:
        lines.append(f"{names[result.winner]} wins on turn {result.turns}.")
    return "\n".join(lines)
