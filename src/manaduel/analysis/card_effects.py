"""Estimate how strong each card is, using the random deal as a natural experiment.

The naive statistic, "how often does the player who PLAYED this card win?", is
biased: strong players play strong cards when they are already winning.

The deal is different. Each player's starting hand is random and does not
depend on anyone's skill or choices. So if seat 0 happened to be dealt more
Fireballs than seat 1, any resulting change in win rate is caused by the cards.

For each game we regress the outcome on the difference in starting copies:

    score0 = baseline + sum over cards c of  effect_c * (copies0_c - copies1_c)

where score0 is 1 for a seat-0 win, 0.5 for a draw and 0 for a loss. Both seats
are played by the SAME bot (a mirror match), so skill is equal and only the
cards differ. `effect_c` is the change in win probability from one extra
starting copy of card c.

Because both hands have equal size, the differences sum to a constant, so only
differences BETWEEN cards are identified. Effects are therefore reported
relative to the average card (they average to zero). `seat0_baseline` is the
first player's win probability when both players are dealt the same cards
(valid when both hands are the same size, as in the default config).

Limits: the estimate measures the value of holding a card at the start of the
game (and the bot's use of it), for the bots tested. A hand-size cap that burns
cards makes the deal slightly less than perfectly exogenous later in a game,
but starting hands are well below the cap.
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from manaduel.engine.game import play_game


@dataclass(frozen=True)
class Games:
    card_names: tuple[str, ...]
    score0: np.ndarray   # shape (n,): 1 = seat 0 won, 0.5 = draw, 0 = seat 1 won
    delta: np.ndarray    # shape (n, n_cards): copies dealt to seat 0 minus seat 1


def collect(config, factory, n_games: int, seed: int = 0) -> Games:
    """Play `n_games` mirror matches (the same bot in both seats) and record the outcomes
    and the dealt hands. Each game gets its own deal seed."""
    n_cards = len(config.cards)
    score0 = np.empty(n_games)
    delta = np.empty((n_games, n_cards))
    for g in range(n_games):
        deal_seed = seed + 1_000_003 * (g + 1)
        agents = (factory(seed + 2 * g), factory(seed + 2 * g + 1))
        result = play_game(config, agents, seed=deal_seed)
        if result.winner is None:
            score0[g] = 0.5
        else:
            score0[g] = 1.0 if result.winner == 0 else 0.0
        hand0, hand1 = result.start_hands
        delta[g] = np.array(hand0) - np.array(hand1)
    return Games(tuple(card.name for card in config.cards), score0, delta)


@dataclass(frozen=True)
class CardEffect:
    name: str
    effect: float    # win-probability change per extra starting copy, vs the average card
    se: float
    ci_low: float    # 95% confidence interval (not adjusted for testing several cards)
    ci_high: float
    z: float
    flagged: bool    # clearly stronger or weaker than the average card


@dataclass(frozen=True)
class EffectReport:
    effects: tuple[CardEffect, ...]
    seat0_baseline: float      # first player's win probability with identical hands
    seat0_baseline_se: float
    n_games: int

    def flagged(self) -> list[CardEffect]:
        return [e for e in self.effects if e.flagged]


def estimate_effects(games: Games, alpha: float = 0.05, min_effect: float = 0.02) -> EffectReport:
    """Fit the regression and flag outlier cards.

    A card is flagged when BOTH hold:
      - its effect is statistically distinguishable from zero, with the
        threshold tightened (Bonferroni) for testing every card at once, and
      - its size is at least `min_effect` (2 percentage points of win
        probability per extra copy by default), so tiny but "significant"
        differences from a very large sample are not reported.
    """
    n, k = games.delta.shape
    # Drop the last card's column so the design matrix is full rank.
    X = np.column_stack([np.ones(n), games.delta[:, : k - 1]])
    y = games.score0

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    # Heteroskedasticity-robust covariance (HC1): outcomes are 0 / 0.5 / 1, so the
    # usual equal-variance formula would be wrong.
    meat = (X * resid[:, None] ** 2).T @ X
    cov = xtx_inv @ meat @ xtx_inv * n / (n - X.shape[1])

    # beta[1:] are effects relative to the last card. Re-express each card
    # relative to the average card: e_c = b_c - mean(b), where b_last = 0.
    center = np.eye(k)[:, : k - 1] - 1.0 / k
    effect = center @ beta[1:]
    cov_effect = center @ cov[1:, 1:] @ center.T
    se = np.sqrt(np.clip(np.diag(cov_effect), 0.0, None))

    z95 = NormalDist().inv_cdf(0.975)
    z_crit = NormalDist().inv_cdf(1 - alpha / (2 * k))  # Bonferroni across the k cards

    effects = []
    for c in range(k):
        z = effect[c] / se[c] if se[c] > 0 else 0.0
        effects.append(
            CardEffect(
                name=games.card_names[c],
                effect=float(effect[c]),
                se=float(se[c]),
                ci_low=float(effect[c] - z95 * se[c]),
                ci_high=float(effect[c] + z95 * se[c]),
                z=float(z),
                flagged=bool(abs(z) > z_crit and abs(effect[c]) >= min_effect),
            )
        )
    return EffectReport(
        effects=tuple(effects),
        seat0_baseline=float(beta[0]),
        seat0_baseline_se=float(np.sqrt(max(cov[0, 0], 0.0))),
        n_games=n,
    )


def format_report(report: EffectReport) -> str:
    lines = [
        f"Card power from {report.n_games} mirror games "
        "(win-probability change per extra starting copy, vs the average card)",
        f"{'card':>10} {'effect':>8} {'95% interval':>18} {'z':>7}  verdict",
    ]
    for e in sorted(report.effects, key=lambda e: -e.effect):
        verdict = ("TOO STRONG" if e.effect > 0 else "TOO WEAK") if e.flagged else "ok"
        lines.append(
            f"{e.name:>10} {e.effect:>+8.1%} [{e.ci_low:>+7.1%}, {e.ci_high:>+7.1%}] {e.z:>7.1f}  {verdict}"
        )
    lines.append(
        f"First player's win rate with identical hands: {report.seat0_baseline:.1%} "
        f"(+/- {1.96 * report.seat0_baseline_se:.1%})"
    )
    return "\n".join(lines)
