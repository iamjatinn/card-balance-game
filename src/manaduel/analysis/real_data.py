"""Card power from REAL game logs (17Lands Magic: The Gathering draft data).

This is the same natural-experiment idea as `card_effects.py`, adapted to data
where players CHOOSE their deck. The deck is not random, but the opening hand is
a random draw from that deck, so what is exogenous is how many MORE copies of a
card landed in the opening hand than the deck's composition predicts:

    d_c = copies of c in the opening hand - copies of c in the deck * 7 / deck size

We regress "won" on d_c for every card (plus a constant and "on the play").
effect_c = change in win probability from one extra copy of c in the opening
hand, taking the place of an average basic land (basic lands are left out of the
regression, so they are the reference). Only games with no mulligan are used,
because choosing which cards to keep after a mulligan is not random.

Standard errors come from a delete-one-group jackknife, where a group is a set of
drafts (all games of one draft always sit in the same group), because games from
the same draft share a player and a deck.

For comparison we also compute the usual "games in hand" statistic (17Lands'
GIH WR): win rate of games where the card was in the opening hand OR drawn later,
minus win rate of games where it was in the deck but never seen. That one is
biased, because a longer game draws more cards and game length is linked to the
result. Seeing where the two disagree is part of the point.

Limits: the effect is conditional on the card being in the deck of the kind of
player who chose it, there is no ground truth about how good a card "really" is,
and the file is one set and one event type.
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
HAND_SIZE = 7


@dataclass
class Sums:
    """Everything needed from one pass over the file, kept per group of drafts."""
    names: list
    xtx: np.ndarray        # (groups, k, k)  X'X, X = [1, on_play, d_1 .. d_c]
    xty: np.ndarray        # (groups, k)
    n: np.ndarray          # (groups,)       games
    # counts for the naive statistic, totals over all games (no groups needed)
    deck_games: np.ndarray     # games where the card is in the deck
    hand_games: np.ndarray     # in the opening hand
    seen_games: np.ndarray     # in the opening hand or drawn later
    seen_wins: np.ndarray
    unseen_games: np.ndarray   # in the deck but never seen
    unseen_wins: np.ndarray
    hand_wins: np.ndarray
    games_used: int
    games_read: int


def card_columns(header: list[str]) -> list[str]:
    """Card names that have an opening-hand column, basic lands removed."""
    names = [c[len("opening_hand_"):] for c in header if c.startswith("opening_hand_")]
    return [n for n in names if n not in BASIC_LANDS]


def accumulate(path, groups: int = 200, chunk_rows: int = 40_000, max_rows: int | None = None,
               min_deck_fraction: float = 0.003, progress=None) -> Sums:
    """One streaming pass: read in chunks, keep only what the analysis needs.

    Cards that appear in fewer than `min_deck_fraction` of games' decks are dropped
    (their coefficients would be unreliable and the matrix nearly singular)."""
    import pandas as pd

    header = list(pd.read_csv(path, nrows=0).columns)
    all_cards = card_columns(header)
    deck_cols_all = [c for c in header if c.startswith("deck_")]

    probe_n = min(max_rows or 10**9, 150_000)
    probe = pd.read_csv(path, usecols=["num_mulligans"] + [f"deck_{c}" for c in all_cards], nrows=probe_n)
    probe = probe[probe["num_mulligans"] == 0]
    in_deck = (probe[[f"deck_{c}" for c in all_cards]].to_numpy() > 0).sum(axis=0)
    names = [c for c, k in zip(all_cards, in_deck) if k >= min_deck_fraction * max(len(probe), 1)]
    k = 2 + len(names)

    hand_cols = [f"opening_hand_{c}" for c in names]
    drawn_cols = [f"drawn_{c}" for c in names]
    deck_cols = [f"deck_{c}" for c in names]
    usecols = ["draft_id", "won", "on_play", "num_mulligans"] + hand_cols + drawn_cols + deck_cols_all
    dtypes = {c: np.uint8 for c in hand_cols + drawn_cols + deck_cols_all}
    dtypes["num_mulligans"] = np.int8

    xtx = np.zeros((groups, k, k))
    xty = np.zeros((groups, k))
    n_g = np.zeros(groups, dtype=np.int64)
    c = len(names)
    deck_games = np.zeros(c, dtype=np.int64)
    hand_games = np.zeros(c, dtype=np.int64)
    hand_wins = np.zeros(c, dtype=np.int64)
    seen_games = np.zeros(c, dtype=np.int64)
    seen_wins = np.zeros(c, dtype=np.int64)
    unseen_games = np.zeros(c, dtype=np.int64)
    unseen_wins = np.zeros(c, dtype=np.int64)
    used = read = 0

    reader = pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunk_rows, nrows=max_rows)
    for chunk in reader:
        read += len(chunk)
        chunk = chunk[(chunk["num_mulligans"] == 0)]
        if chunk.empty:
            continue
        deck_all = chunk[deck_cols_all].to_numpy(dtype=np.float64)
        deck_size = deck_all.sum(axis=1)
        ok = deck_size > 0
        if not ok.all():
            chunk, deck_all, deck_size = chunk[ok], deck_all[ok], deck_size[ok]
        hand = chunk[hand_cols].to_numpy(dtype=np.float64)
        drawn = chunk[drawn_cols].to_numpy(dtype=np.float64)
        deck = chunk[deck_cols].to_numpy(dtype=np.float64)
        y = chunk["won"].to_numpy(dtype=np.float64)
        on_play = chunk["on_play"].to_numpy(dtype=np.float64)

        d = hand - deck * (HAND_SIZE / deck_size)[:, None]
        X = np.column_stack([np.ones(len(chunk)), on_play, d])
        gid = (pd.util.hash_pandas_object(chunk["draft_id"], index=False).to_numpy() % groups).astype(np.int64)
        for g in np.unique(gid):
            m = gid == g
            xg = X[m]
            xtx[g] += xg.T @ xg
            xty[g] += xg.T @ y[m]
            n_g[g] += m.sum()

        in_deck = deck > 0
        in_hand = hand > 0
        seen = (hand + drawn) > 0
        unseen = in_deck & ~seen
        deck_games += in_deck.sum(axis=0)
        hand_games += in_hand.sum(axis=0)
        hand_wins += (in_hand * y[:, None]).sum(axis=0).astype(np.int64)
        seen_games += seen.sum(axis=0)
        seen_wins += (seen * y[:, None]).sum(axis=0).astype(np.int64)
        unseen_games += unseen.sum(axis=0)
        unseen_wins += (unseen * y[:, None]).sum(axis=0).astype(np.int64)
        used += len(chunk)
        if progress:
            progress(read, used)

    return Sums(names, xtx, xty, n_g, deck_games, hand_games, seen_games, seen_wins,
                unseen_games, unseen_wins, hand_wins, used, read)


def _solve(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    k = xtx.shape[0]
    ridge = 1e-9 * np.trace(xtx) / k
    return np.linalg.solve(xtx + ridge * np.eye(k), xty)


@dataclass(frozen=True)
class RealEffect:
    name: str
    effect: float      # change in win probability per extra copy in the opening hand
    se: float
    ci_low: float
    ci_high: float
    z: float
    flagged: bool
    naive_gih: float   # win rate (seen) - win rate (in deck, never seen): the biased statistic
    deck_games: int
    hand_games: int


@dataclass(frozen=True)
class RealReport:
    effects: tuple
    on_play_effect: float
    games_used: int
    games_read: int
    groups: int


def estimate(sums: Sums, alpha: float = 0.05, min_effect: float = 0.01,
             group_mask: np.ndarray | None = None) -> RealReport:
    """Fit the regression from the stored sums and get jackknife standard errors."""
    G = sums.xtx.shape[0]
    mask = np.ones(G, dtype=bool) if group_mask is None else group_mask
    mask = mask & (sums.n > 0)
    idx = np.flatnonzero(mask)
    total_xtx = sums.xtx[idx].sum(axis=0)
    total_xty = sums.xty[idx].sum(axis=0)
    beta = _solve(total_xtx, total_xty)

    leave = np.array([_solve(total_xtx - sums.xtx[g], total_xty - sums.xty[g]) for g in idx])
    m = len(idx)
    mean = leave.mean(axis=0)
    var = (m - 1) / m * ((leave - mean) ** 2).sum(axis=0)
    se = np.sqrt(var)

    c = len(sums.names)
    z95 = NormalDist().inv_cdf(0.975)
    z_crit = NormalDist().inv_cdf(1 - alpha / (2 * c))
    with np.errstate(divide="ignore", invalid="ignore"):
        seen_wr = sums.seen_wins / sums.seen_games
        unseen_wr = sums.unseen_wins / sums.unseen_games
    effects = []
    for j, name in enumerate(sums.names):
        e, s = float(beta[2 + j]), float(se[2 + j])
        z = e / s if s > 0 else 0.0
        effects.append(RealEffect(
            name, e, s, e - z95 * s, e + z95 * s, z,
            bool(abs(z) > z_crit and abs(e) >= min_effect),
            float(seen_wr[j] - unseen_wr[j]), int(sums.deck_games[j]), int(sums.hand_games[j]),
        ))
    return RealReport(tuple(effects), float(beta[1]), int(sums.n[idx].sum()), sums.games_read, m)


def split_half(sums: Sums, seed: int = 0) -> dict:
    """Estimate on two random halves of the draft groups; how well do they agree?"""
    G = sums.xtx.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(G)
    a = np.zeros(G, dtype=bool)
    a[order[: G // 2]] = True
    ra, rb = estimate(sums, group_mask=a), estimate(sums, group_mask=~a)
    ea = np.array([e.effect for e in ra.effects])
    eb = np.array([e.effect for e in rb.effects])
    sa = np.array([e.se for e in ra.effects])
    sb = np.array([e.se for e in rb.effects])
    strong = (np.abs(ea) > 2 * sa) & (np.abs(eb) > 2 * sb)
    return {
        "correlation": float(np.corrcoef(ea, eb)[0, 1]),
        "sign_agreement_all": float(np.mean(np.sign(ea) == np.sign(eb))),
        "cards_clear_in_both_halves": int(strong.sum()),
        "sign_agreement_clear": float(np.mean(np.sign(ea[strong]) == np.sign(eb[strong]))) if strong.any() else float("nan"),
        "half_a": ra, "half_b": rb,
    }


def comparison(report: RealReport) -> dict:
    """How much does the biased 'games in hand' statistic agree with the natural-experiment estimate?"""
    ours = np.array([e.effect for e in report.effects])
    naive = np.array([e.naive_gih for e in report.effects])
    spearman = float(np.corrcoef(_rank(ours), _rank(naive))[0, 1])
    return {
        "pearson": float(np.corrcoef(ours, naive)[0, 1]),
        "spearman": spearman,
        "naive_scale": float(np.std(naive) / np.std(ours)),   # how much bigger the naive spread is
        "sign_agreement": float(np.mean(np.sign(ours) == np.sign(naive))),
    }


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    return ranks


def format_report(report: RealReport, top: int = 12) -> str:
    eff = sorted(report.effects, key=lambda e: -e.effect)
    lines = [
        f"Card power from real games: {report.games_used:,} games with no mulligan "
        f"(of {report.games_read:,} read), {len(report.effects)} cards, {report.groups} draft groups",
        "effect = win-probability change per extra copy in the opening hand, versus a basic land;",
        "naive  = 'games in hand' style statistic (seen vs never seen), which is biased by game length",
        f"on the play: {report.on_play_effect:+.1%} win probability",
        "",
        f"{'card':<34}{'effect':>8}{'95% interval':>20}{'naive':>8}{'deck games':>12}  flag",
    ]

    def row(e):
        flag = "STRONG" if e.flagged and e.effect > 0 else "WEAK" if e.flagged else ""
        return (f"{e.name[:33]:<34}{e.effect:>+8.1%}  [{e.ci_low:>+6.1%}, {e.ci_high:>+6.1%}]"
                f"{e.naive_gih:>+8.1%}{e.deck_games:>12,}  {flag}")

    lines.append(f"--- top {top} ---")
    lines += [row(e) for e in eff[:top]]
    lines.append(f"--- bottom {top} ---")
    lines += [row(e) for e in eff[-top:]]
    flagged = [e for e in report.effects if e.flagged]
    lines.append("")
    lines.append(f"flagged: {sum(e.effect > 0 for e in flagged)} strong, {sum(e.effect < 0 for e in flagged)} weak "
                 f"of {len(report.effects)} cards (Bonferroni, and at least 1 point)")
    return "\n".join(lines)


def write_csv(report: RealReport, path) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["card", "effect", "se", "ci_low", "ci_high", "z", "flagged", "naive_gih",
                    "deck_games", "hand_games"])
        for e in sorted(report.effects, key=lambda e: -e.effect):
            w.writerow([e.name, f"{e.effect:.5f}", f"{e.se:.5f}", f"{e.ci_low:.5f}", f"{e.ci_high:.5f}",
                        f"{e.z:.2f}", int(e.flagged), f"{e.naive_gih:.5f}", e.deck_games, e.hand_games])
