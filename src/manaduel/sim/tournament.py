"""Play many games between bots and summarise who wins."""

from dataclasses import dataclass

from manaduel.engine.game import play_game
from manaduel.sim.elo import fit_elo
from manaduel.sim.stats import wilson_interval


@dataclass(frozen=True)
class MatchResult:
    name_a: str
    name_b: str
    wins_a: int
    wins_b: int
    draws: int
    first_player_wins: int  # games won by whoever moved first (either bot)
    turns_total: int = 0    # sum of game lengths, for the average

    @property
    def games(self) -> int:
        return self.wins_a + self.wins_b + self.draws

    @property
    def avg_turns(self) -> float:
        return self.turns_total / self.games

    @property
    def points_a(self) -> float:
        return self.wins_a + 0.5 * self.draws

    @property
    def score_a(self) -> float:
        """A's win rate, counting a draw as half a win."""
        return self.points_a / self.games

    def interval_a(self) -> tuple[float, float]:
        return wilson_interval(self.points_a, self.games)


def play_match(config, name_a, factory_a, name_b, factory_b, n_games, seed=0, decks=None) -> MatchResult:
    """Play n_games between two bots, swapping who moves first every game.

    factory_x(seed) must build a fresh bot. Fresh bots per game mean the games
    are independent and each one is reproducible from its seed.

    decks (deck mode only): (deck for bot A, deck for bot B); each bot keeps its
    deck when the seats swap.
    """
    wins_a = wins_b = draws = first_wins = turns_total = 0
    for g in range(n_games):
        a_moves_first = g % 2 == 0
        bot_a = factory_a(seed + 2 * g)
        bot_b = factory_b(seed + 2 * g + 1)
        agents = (bot_a, bot_b) if a_moves_first else (bot_b, bot_a)
        # Games come in pairs that share one card-draw seed (seats swapped),
        # which cancels some of the luck of the draw between the two bots.
        game_seed = seed + 1_000_003 * (g // 2 + 1)
        seat_decks = None
        if decks is not None:
            seat_decks = decks if a_moves_first else (decks[1], decks[0])
        result = play_game(config, agents, seed=game_seed, decks=seat_decks)
        turns_total += result.turns

        if result.winner is None:
            draws += 1
            continue
        if result.winner == 0:
            first_wins += 1
        a_won = (result.winner == 0) == a_moves_first
        if a_won:
            wins_a += 1
        else:
            wins_b += 1
    return MatchResult(name_a, name_b, wins_a, wins_b, draws, first_wins, turns_total)


def round_robin(config, factories: dict, n_games: int, seed: int = 0) -> list[MatchResult]:
    """Every bot plays every other bot, and also a copy of itself (the 'mirror'
    match, which measures first-player advantage)."""
    names = list(factories)
    results = []
    for i, a in enumerate(names):
        for b in names[i:]:
            results.append(
                play_match(config, a, factories[a], b, factories[b], n_games, seed)
            )
    return results


def format_report(results: list[MatchResult], names: list[str]) -> str:
    lines = []
    pairs = [r for r in results if r.name_a != r.name_b]
    mirrors = [r for r in results if r.name_a == r.name_b]

    lines.append("Head to head (win rate of the first bot, draw = half a win, 95% interval)")
    for r in pairs:
        lo, hi = r.interval_a()
        lines.append(
            f"  {r.name_a:>10} vs {r.name_b:<10} {r.score_a:6.1%}  [{lo:5.1%}, {hi:5.1%}]"
            f"   W{r.wins_a} L{r.wins_b} D{r.draws}"
        )

    # Score matrix: row bot's win rate against column bot.
    scores = {}
    for r in pairs:
        scores[(r.name_a, r.name_b)] = r.score_a
        scores[(r.name_b, r.name_a)] = 1 - r.score_a
    lines.append("")
    lines.append("Win-rate matrix (row beats column this often)")
    lines.append(" " * 11 + "".join(f"{n:>11}" for n in names))
    for row in names:
        cells = "".join(
            f"{'-':>11}" if row == col else f"{scores[(row, col)]:>11.1%}"
            for col in names
        )
        lines.append(f"{row:>10} {cells}")

    if mirrors:
        lines.append("")
        lines.append("First-player advantage (bot vs its own copy; 50% = fair)")
        for r in mirrors:
            decisive = r.wins_a + r.wins_b
            rate = r.first_player_wins / decisive if decisive else float("nan")
            lines.append(f"  {r.name_a:>10}: first mover wins {rate:6.1%} of decisive games ({r.draws} draws)")

    ratings = fit_elo(names, pairs)
    lines.append("")
    lines.append("Elo ratings (relative strength; 400 points = 10x odds)")
    for n in sorted(names, key=lambda n: -ratings[n]):
        lines.append(f"  {n:>10}: {ratings[n]:6.0f}")
    return "\n".join(lines)
