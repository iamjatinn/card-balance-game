"""Figures for the README. Every figure is drawn from saved data, never typed in by hand."""

import csv
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BOT_COLORS = {"random": "#8c8c96", "aggressive": "#e0604a", "defensive": "#4a90e0", "greedy": "#e0a030"}
CARD_COLORS = {"strike": "#e0604a", "fireball": "#b03030", "heal": "#4ab070", "shield": "#4a90e0", "drain": "#a050c0"}


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
