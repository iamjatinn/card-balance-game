import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from manaduel.analysis.real_data import (  # noqa: E402
    accumulate,
    comparison,
    estimate,
    format_report,
    split_half,
    write_csv,
)

CARDS = [f"Card{i}" for i in range(10)]
TRUE = {"Card0": 0.04, "Card1": -0.03}   # every other card has no effect


def make_file(path, drafts=5000, games_per_draft=5, seed=0):
    """A fake file in the 17Lands layout where we KNOW each card's effect.

    Decks are chosen (not random) and a deck's quality is linked to how many Card0 it
    holds, a confounder. The opening hand is a random draw from the deck, and later
    draws depend on game length, which depends on the result: so the naive
    'games in hand' statistic is biased, while the opening-hand regression is not."""
    rng = np.random.default_rng(seed)
    cols = ["draft_id", "won", "on_play", "num_mulligans"]
    cols += [f"opening_hand_{c}" for c in CARDS + ["Plains", "Island"]]
    cols += [f"drawn_{c}" for c in CARDS + ["Plains", "Island"]]
    cols += [f"deck_{c}" for c in CARDS + ["Plains", "Island"]]
    rows = []
    for d in range(drafts):
        counts = rng.integers(0, 3, size=len(CARDS))
        lands = 40 - counts.sum()
        deck = np.concatenate([counts, [lands // 2, lands - lands // 2]])
        pool = np.repeat(np.arange(len(deck)), deck)
        quality = 0.03 * (counts[0] - 1)
        for g in range(games_per_draft):
            order = rng.permutation(pool)
            hand = np.bincount(order[:7], minlength=len(deck))
            on_play = rng.random() < 0.5
            expected = deck * 7 / 40
            p = 0.5 + 0.02 * on_play + quality + (TRUE.get("Card0", 0) * (hand[0] - expected[0])
                                                  + TRUE.get("Card1", 0) * (hand[1] - expected[1]))
            won = rng.random() < p
            n_draws = 4 + 6 * int(won) + rng.integers(0, 3)   # winners play longer games
            drawn = np.bincount(order[7:7 + n_draws], minlength=len(deck))
            rows.append([f"draft{d}", won, on_play, 0, *hand, *drawn, *deck])
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


@pytest.fixture(scope="module")
def sums(tmp_path_factory):
    path = tmp_path_factory.mktemp("lands") / "fake.csv"
    make_file(path)
    return accumulate(path, groups=40, chunk_rows=7000)


def test_reads_every_game_and_every_card(sums):
    assert sums.games_read == sums.games_used == 25_000
    assert sums.names == CARDS


def test_planted_effects_are_recovered_and_null_cards_are_not_flagged(sums):
    report = estimate(sums, min_effect=0.01)
    by = {e.name: e for e in report.effects}
    for name, truth in TRUE.items():
        e = by[name]
        assert abs(e.effect - truth) < 4 * e.se, (name, e.effect, truth, e.se)
        assert e.flagged
    for name in CARDS[2:]:
        assert abs(by[name].effect) < 4 * by[name].se
    assert not any(by[n].flagged for n in CARDS[2:])
    assert abs(report.on_play_effect - 0.02) < 0.02


def test_the_naive_statistic_is_biased_by_game_length(sums):
    report = estimate(sums)
    by = {e.name: e for e in report.effects}
    null_naive = [by[n].naive_gih for n in CARDS[2:]]
    # winners draw more cards, so ANY card looks better when 'seen'; the regression effect stays near 0
    assert min(null_naive) > 0.03
    assert max(abs(by[n].effect) for n in CARDS[2:]) < 0.015


def test_split_halves_agree_on_the_planted_cards(sums):
    s = split_half(sums, seed=1)
    ea = {e.name: e.effect for e in s["half_a"].effects}
    eb = {e.name: e.effect for e in s["half_b"].effects}
    assert ea["Card0"] > 0 and eb["Card0"] > 0
    assert ea["Card1"] < 0 and eb["Card1"] < 0
    assert -1.0 <= s["correlation"] <= 1.0


def test_comparison_and_outputs(sums, tmp_path):
    report = estimate(sums)
    comp = comparison(report)
    assert set(comp) == {"pearson", "spearman", "naive_scale", "sign_agreement"}
    text = format_report(report, top=3)
    assert "Card0" in text and "flagged" in text
    write_csv(report, tmp_path / "r.csv")
    assert (tmp_path / "r.csv").read_text(encoding="utf-8").count("\n") == len(CARDS) + 1


def test_mulligan_games_are_excluded(tmp_path):
    path = tmp_path / "m.csv"
    make_file(path, drafts=400, games_per_draft=5)
    df = pd.read_csv(path)
    df.loc[df.index % 2 == 0, "num_mulligans"] = 1
    df.to_csv(path, index=False)
    s = accumulate(path, groups=10, chunk_rows=500)
    assert s.games_read == 2000 and s.games_used == 1000
