"""Figures for the README. Every figure is drawn from saved data, never typed in by hand."""

import csv
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BOT_COLORS = {"random": "#8c8c96", "aggressive": "#e0604a", "defensive": "#4a90e0", "greedy": "#e0a030"}
CARD_COLORS = {"strike": "#e0604a", "fireball": "#b03030", "heal": "#4ab070", "shield": "#4a90e0", "drain": "#a050c0",
               "insight": "#30a0b0", "venom": "#7a9a20", "flurry": "#e08030"}


def plot_learning_curves(agg, path, title=None) -> None:
    """Win rate against each bot during training, mean +/- one standard deviation across runs."""
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=130)
    x = agg.steps / 1000
    for i, bot in enumerate(agg.bots):
        color = BOT_COLORS.get(bot, None)
        ax.plot(x, agg.mean[:, i] * 100, label=bot, color=color, lw=1.8)
        ax.fill_between(x, (agg.mean[:, i] - agg.std[:, i]) * 100, (agg.mean[:, i] + agg.std[:, i]) * 100,
                        color=color, alpha=0.15, lw=0)
    ax.plot(x, agg.overall_mean * 100, color="black", lw=2.6, label="average of the four")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.set_xlabel("training steps (thousands)")
    ax.set_ylabel("PPO win rate (%)")
    ax.set_ylim(40, 100)
    ax.set_title(title or f"PPO learning curve against each bot ({agg.runs} runs, mean ± 1 std)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


GROUP_COLORS = ["#4a90e0", "#e0604a", "#4ab070", "#a050c0", "#e0a030"]


def plot_group_comparison(groups: dict, path, title=None) -> None:
    """Training curves for several variants side by side. `groups` maps a label to an
    `Aggregate` (see multiseed.aggregate). Top-left is the average over the four bots;
    the other panels are the win rate against each bot. Lines are means over runs, bands are
    +/- one standard deviation across runs."""
    first = next(iter(groups.values()))
    bots = list(first.bots)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), dpi=130, sharex=True)
    panels = [("average of the four bots", None)] + [(f"vs {b}", i) for i, b in enumerate(bots)]
    for ax, (label, bot_index) in zip(axes.ravel(), panels):
        for color, (name, agg) in zip(GROUP_COLORS, groups.items()):
            x = agg.steps / 1000
            mean = agg.overall_mean if bot_index is None else agg.mean[:, bot_index]
            std = agg.overall_std if bot_index is None else agg.std[:, bot_index]
            ax.plot(x, mean * 100, color=color, lw=2, label=f"{name} ({agg.runs} runs)")
            ax.fill_between(x, (mean - std) * 100, (mean + std) * 100, color=color, alpha=0.15, lw=0)
        ax.set_title(label, fontsize=10)
        ax.axhline(50, color="gray", ls=":", lw=1)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("training steps (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("win rate (%)")
    axes.ravel()[-1].axis("off")                      # six slots, five panels
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(handles, labels, loc="center", fontsize=10, frameon=False)
    fig.suptitle(title or "Training variants compared (mean +/- 1 std across runs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_card_effects(report, path, title=None) -> None:
    """Each card's power relative to the reference card, with 95% confidence intervals."""
    effects = sorted(report.effects, key=lambda e: e.effect)
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=130)
    y = np.arange(len(effects))
    for yi, e in zip(y, effects):
        ax.barh(yi, e.effect * 100, color=CARD_COLORS.get(e.name, "gray"), alpha=0.85, height=0.6)
        if e.se > 0:
            ax.errorbar(e.effect * 100, yi, xerr=[[(e.effect - e.ci_low) * 100], [(e.ci_high - e.effect) * 100]],
                        color="black", capsize=4, lw=1.4)
        if e.flagged:  # label just past the end of the error bar
            edge = (e.ci_high if e.effect > 0 else e.ci_low) * 100
            ax.text(edge + (0.3 if e.effect > 0 else -0.3), yi, "flagged", ha="left" if e.effect > 0 else "right",
                    va="center", fontsize=8, color="black")
    lo = min(e.ci_low for e in effects) * 100
    hi = max(e.ci_high for e in effects) * 100
    ax.set_xlim(min(lo - 2.5, -1), max(hi + 2.5, 1))
    ax.set_yticks(y)
    ax.set_yticklabels([e.name for e in effects])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(f"win probability change per extra starting copy (points), vs the {report.reference} card")
    ax.set_title(title or f"Card power in the current game ({report.n_games:,} mirror games, 95% intervals)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_real_vs_naive(report, path, label_top: int = 6, title=None) -> None:
    """Natural-experiment effect (x) against the biased 'games in hand' statistic (y), one dot per card."""
    eff = report.effects
    x = np.array([e.effect for e in eff]) * 100
    y = np.array([e.naive_gih for e in eff]) * 100
    xerr = np.array([[e.effect - e.ci_low for e in eff], [e.ci_high - e.effect for e in eff]]) * 100
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=130)
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#bbbbbb", lw=0.7, zorder=1)
    flagged = np.array([e.flagged for e in eff])
    ax.scatter(x[~flagged], y[~flagged], s=14, color="#4a90e0", alpha=0.75, label="not flagged", zorder=2)
    ax.scatter(x[flagged], y[flagged], s=22, color="#e0604a", label="flagged", zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], ls=":", color="gray", lw=1)
    gap = np.abs(y - x)
    for i in np.argsort(-gap)[:label_top]:
        ax.annotate(eff[i].name[:22], (x[i], y[i]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("natural-experiment effect: win probability per extra opening-hand copy (points)")
    ax.set_ylabel("naive 'games in hand' gap: seen vs never seen (points)")
    ax.set_title(title or "Real games: unbiased effect vs the usual games-in-hand statistic")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_deck_matrix(matrix, path, title=None) -> None:
    """Heat map of how often each row deck beats each column deck."""
    n = len(matrix.names)
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * n, 0.8 + 0.8 * n), dpi=130)
    im = ax.imshow(matrix.win * 100, cmap="RdBu", vmin=25, vmax=75)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", color="gray")
            else:
                clear = matrix.low[i, j] > 0.5 or matrix.high[i, j] < 0.5   # interval excludes 50%
                ax.text(j, i, f"{matrix.win[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                        fontweight="bold" if clear else "normal", color="black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.names, fontsize=8)
    ax.set_xlabel("opponent's deck")
    ax.set_ylabel("deck (win rate in %)")
    ax.set_title(title or f"Deck vs deck, {matrix.games} games per pairing (bold = clearly not 50%)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_validation_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_detection(rows, path, n_cards=5, alpha=0.05, title=None) -> None:
    """How strongly the tool reacted (|z|) to each planted change, against how big the change was."""
    z_crit = NormalDist().inv_cdf(1 - alpha / (2 * n_cards))
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=130)
    for card in sorted({r["card"] for r in rows}):
        color = CARD_COLORS.get(card, "gray")
        hit = [(abs(float(r["relative_change"])) * 100, abs(float(r["z"])))
               for r in rows if r["card"] == card and int(r["detected"]) == 1]
        miss = [(abs(float(r["relative_change"])) * 100, abs(float(r["z"])))
                for r in rows if r["card"] == card and int(r["detected"]) != 1]
        ax.scatter([p[0] for p in hit], [p[1] for p in hit], s=46, color=color, label=card,
                   edgecolor="black", linewidth=0.6, alpha=0.9)
        ax.scatter([p[0] for p in miss], [p[1] for p in miss], s=46, facecolor="white", edgecolor=color,
                   linewidth=1.8, zorder=3)
    ax.axhline(z_crit, color="black", ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xt = [10, 15, 25, 35, 50, 70, 100, 135]
    yt = [1, 2, 3, 5, 10, 20, 40]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{v}%" for v in xt]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in yt]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(9, 150)
    ax.set_ylim(0.8, 60)
    ax.text(150, z_crit * 1.06, "detection threshold ", va="bottom", ha="right", fontsize=8)
    ax.text(9.4, 0.86, "hollow marker = not detected", fontsize=8, va="bottom")
    ax.set_xlabel("size of the planted change (% of the card's main stat, log scale)")
    ax.set_ylabel("tool's reaction |z| (log scale)")
    detected = sum(int(r["detected"]) for r in rows)
    ax.set_title(title or f"Detecting planted card changes: {detected}/{len(rows)} detected")
    ax.grid(alpha=0.25, which="major")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
