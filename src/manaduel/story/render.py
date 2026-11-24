"""Turn a recorded training run (see record.py) into pictures for sharing.

Everything drawn here comes from the saved files; nothing is typed in by hand:
  learning_curve.png      training results and checkpoint evaluations against match number
  learning_animation.gif  the curve and per-bot win rates growing from match 1 to match N
  behaviour.png           which cards the agent plays, before and after training
  before_after.gif        the untrained and the trained agent in the same game against the same bot
  linkedin_card.png       one summary image (1200 x 627)
  linkedin_post.md        a draft caption using the real numbers (edit it before posting)
"""

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

BG = "#14161f"
PANEL = "#1c1f2b"
TEXT = "#e8eaf2"
MUTED = "#8b91a8"
GRID = "#2c3043"
ACCENT = "#5fd18a"
BEFORE = "#7a8099"
LINE = "#5b9bf0"
BOT_COLORS = {"random": "#9aa0b4", "aggressive": "#e0604a", "defensive": "#4a90e0", "greedy": "#e0a030"}
CARD_COLORS = {"strike": "#e0604a", "fireball": "#b03030", "heal": "#4ab070", "shield": "#4a90e0",
               "drain": "#a050c0", "insight": "#30a0b0", "venom": "#7a9a20", "flurry": "#e08030"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass
class Story:
    meta: dict
    match: np.ndarray        # training match numbers 1..N
    result: np.ndarray       # 1 / 0.5 / 0 for each training match
    ck: dict                 # checkpoint columns by name (numpy arrays)
    bots: list
    cards: list

    @property
    def n_matches(self) -> int:
        return int(self.ck["matches"][-1])


def load_story(out_dir) -> Story:
    out = Path(out_dir)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    with open(out / "matches.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(out / "checkpoints.csv", newline="", encoding="utf-8") as f:
        ck_rows = list(csv.DictReader(f))
    if not rows or len(ck_rows) < 2:
        raise ValueError(f"{out} does not hold a finished recording (need matches and at least 2 checkpoints)")
    columns = list(ck_rows[0])
    ck = {c: np.array([float(r[c]) for r in ck_rows]) for c in columns}
    bots = [b for b in meta["opponents"]]
    cards = [c[len("plays_"):] for c in columns if c.startswith("plays_")]
    return Story(meta, np.array([int(r["match"]) for r in rows]), np.array([float(r["result"]) for r in rows]),
                 ck, bots, cards)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average over the last `window` values (a shorter window at the start)."""
    values = np.asarray(values, dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(1, len(values) + 1)
    start = np.maximum(0, idx - window)
    return (csum[idx] - csum[start]) / (idx - start)


def window_for(story: Story) -> int:
    return int(max(100, len(story.result) // 40))


def duration_text(seconds: float) -> str:
    """'45 s' for short runs, '8 min' for longer ones."""
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"


def _style(ax):
    ax.set_axisbelow(True)          # gridlines behind the bars and lines, not over them
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.6, alpha=0.7)


# --------------------------------------------------------------------------
# Static figures
# --------------------------------------------------------------------------

def _curves(ax, story: Story, upto: int | None = None, show_points: bool = True):
    """Draw the training curve and the checkpoint evaluations (up to match `upto`)."""
    w = window_for(story)
    roll = rolling_mean(story.result, w) * 100
    m = story.match
    stop = len(m) if upto is None else int(np.searchsorted(m, upto, side="right"))
    train_line, = ax.plot(m[:stop], roll[:stop], color=LINE, lw=1.6, alpha=0.9,
                          label=f"training matches (average of the last {w:,})")
    cm, cmean = story.ck["matches"], story.ck["mean"] * 100
    cstop = int(np.searchsorted(cm, upto, side="right")) if upto is not None else len(cm)
    eval_line, = ax.plot(cm[:cstop], cmean[:cstop], color=ACCENT, lw=2.4, marker="o", ms=4,
                         label="checkpoint test (fresh games, no exploration)")
    return train_line, eval_line


def plot_learning_curve(story: Story, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130, facecolor=BG)
    _style(ax)
    _curves(ax, story)
    ax.set_xlabel("training matches played", color=MUTED)
    ax.set_ylabel("win rate against the bots (%)", color=MUTED)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, story.n_matches * 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    first, last = story.ck["mean"][0] * 100, story.ck["mean"][-1] * 100
    ax.annotate(f"untrained: {first:.0f}%", (0, first), xytext=(story.n_matches * 0.06, first - 12),
                color=BEFORE, fontsize=10, arrowprops=dict(arrowstyle="-", color=BEFORE))
    ax.annotate(f"after {story.n_matches:,} matches: {last:.0f}%", (story.n_matches, last),
                xytext=(story.n_matches * 0.45, last - 22), color=ACCENT, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACCENT))
    ax.set_title("An agent learns a card game from scratch", color=TEXT, fontsize=13, pad=12)
    leg = ax.legend(loc="lower right", fontsize=8, facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def pick_checkpoints(story: Story, count: int = 4) -> list[int]:
    """Indices of checkpoints to compare: the start, the end, and some in between."""
    n = len(story.ck["matches"])
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, count)})


def plot_behaviour(story: Story, path) -> None:
    """Average number of times per game the agent plays each card, at several points in training."""
    picks = pick_checkpoints(story, 4)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130, facecolor=BG)
    _style(ax)
    ax.grid(axis="x", visible=False)
    bottoms = np.zeros(len(picks))
    for card in story.cards:
        values = np.array([story.ck[f"plays_{card}"][i] for i in picks])
        ax.bar(range(len(picks)), values, bottom=bottoms, color=CARD_COLORS.get(card, "#999999"),
               label=card, width=0.62, edgecolor=BG, linewidth=0.8)
        bottoms += values
    ax.set_xticks(range(len(picks)))
    ax.set_xticklabels([f"match {int(story.ck['matches'][i]):,}" + ("\n(untrained)" if i == 0 else "")
                        for i in picks], color=MUTED)
    ax.set_ylabel("cards played per game", color=MUTED)
    for x, i in enumerate(picks):
        ax.text(x, bottoms[x] + 0.2, f"{bottoms[x]:.1f} cards\n{story.ck['turns'][i]:.0f}-turn games",
                ha="center", va="bottom", color=TEXT, fontsize=8)
    ax.set_ylim(0, bottoms.max() * 1.25)
    ax.set_title("How the agent's play changes as it learns", color=TEXT, fontsize=13, pad=12)
    leg = ax.legend(ncol=len(story.cards), loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8,
                    facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def _bar_panel(ax, story: Story, win_start: dict, win_now: dict, now_label: str):
    _style(ax)
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(story.bots))[::-1]
    h = 0.34
    ax.barh(ys + h / 2, [win_start[b] * 100 for b in story.bots], height=h, color=BEFORE, label="untrained")
    bars = ax.barh(ys - h / 2, [win_now[b] * 100 for b in story.bots], height=h, color=ACCENT, label=now_label)
    ax.set_yticks(ys)
    ax.set_yticklabels(story.bots, color=TEXT, fontsize=10)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("win rate against that bot (%)", color=MUTED, fontsize=9)
    texts = []
    for y, b in zip(ys, story.bots):
        ax.text(win_start[b] * 100 + 1.5, y + h / 2, f"{win_start[b] * 100:.0f}", va="center", color=MUTED, fontsize=8)
        texts.append(ax.text(win_now[b] * 100 + 1.5, y - h / 2, f"{win_now[b] * 100:.0f}", va="center",
                             color=ACCENT, fontsize=9, fontweight="bold"))
    return bars, texts


def _bar_legend(ax):
    """The legend sits above the bars (as a header) so it never covers a value."""
    leg = ax.legend(title="win rate vs each bot", loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                    fontsize=8, facecolor=BG, edgecolor=BG, labelcolor=TEXT, title_fontsize=10,
                    handlelength=1.2, columnspacing=1.2)
    leg.get_title().set_color(TEXT)


def make_card(story: Story, path, footer: str = "") -> None:
    """The summary image: 1200 x 627, the size LinkedIn shows well in a feed."""
    fig = plt.figure(figsize=(12, 6.27), dpi=100, facecolor=BG)
    fig.text(0.04, 0.925, "A reinforcement-learning agent learns a card game from scratch", color=TEXT,
             fontsize=21, fontweight="bold", va="center")
    fig.text(0.04, 0.865, "PPO. No strategy programmed in: it only gets win / lose feedback after each match.",
             color=MUTED, fontsize=12, va="center")

    ax = fig.add_axes([0.06, 0.28, 0.53, 0.50])
    _style(ax)
    _curves(ax, story)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, story.n_matches * 1.02)
    ax.set_xlabel("training matches played", color=MUTED, fontsize=9)
    ax.set_ylabel("win rate vs the bots (%)", color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    first, last = story.ck["mean"][0] * 100, story.ck["mean"][-1] * 100
    ax.text(story.n_matches * 0.03, first - 11, f"untrained {first:.0f}%", color=BEFORE, fontsize=10)
    ax.text(story.n_matches * 0.98, last - 9, f"{last:.0f}%", color=ACCENT, fontsize=15, fontweight="bold",
            ha="right")
    leg = ax.legend(loc="lower right", fontsize=7, facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)

    ax2 = fig.add_axes([0.70, 0.28, 0.27, 0.50])
    win_start = {b: story.ck[b][0] for b in story.bots}
    win_end = {b: story.ck[b][-1] for b in story.bots}
    _bar_panel(ax2, story, win_start, win_end, "after training")
    _bar_legend(ax2)

    secs = float(story.meta.get("seconds", 0))
    stats = [(f"{story.n_matches:,}", "training matches"),
             (f"{first:.0f}%  →  {last:.0f}%", "average win rate vs the 4 bots"),
             (duration_text(secs), "training time, CPU only (no GPU)")]
    for x, (big, small) in zip((0.04, 0.34, 0.70), stats):
        fig.text(x, 0.135, big, color=ACCENT, fontsize=22, fontweight="bold", va="center")
        fig.text(x, 0.075, small, color=MUTED, fontsize=10, va="center")
    games = int(story.ck["eval_games"][-1])
    fig.text(0.04, 0.02, f"Checkpoint tests: {games} fresh games against each bot, seats swapped, draw = half a win. "
                         f"Opponents: 4 scripted bots (random, aggressive, defensive, greedy).  {footer}".strip(),
             color=MUTED, fontsize=7.5, va="center")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------
# The animation
# --------------------------------------------------------------------------

def _ease(t: float, power: float = 1.7) -> float:
    return float(np.clip(t, 0, 1) ** power)


def animate_learning(story: Story, path, frames: int = 64, hold: int = 14, size=(1000, 562),
                     frame_ms: int = 85) -> int:
    """A GIF in which the learning curve and the per-bot win rates grow from match 1 to match N.
    Early frames are spaced more closely, because most of the change happens early."""
    W, H = size
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.04, 0.93, "An agent learns a card game from scratch", color=TEXT, fontsize=18,
             fontweight="bold", va="center")
    counter = fig.text(0.04, 0.855, "", color=ACCENT, fontsize=24, fontweight="bold", va="center")
    rate_text = fig.text(0.40, 0.855, "", color=TEXT, fontsize=13, va="center")

    ax = fig.add_axes([0.07, 0.14, 0.52, 0.60])
    _style(ax)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, story.n_matches * 1.02)
    ax.set_xlabel("training matches played", color=MUTED, fontsize=9)
    ax.set_ylabel("win rate vs the bots (%)", color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    w = window_for(story)
    roll = rolling_mean(story.result, w) * 100
    train_line, = ax.plot([], [], color=LINE, lw=1.5, alpha=0.9, label="training matches")
    eval_line, = ax.plot([], [], color=ACCENT, lw=2.4, label="checkpoint test")
    dot, = ax.plot([], [], "o", color=ACCENT, ms=8)
    leg = ax.legend(loc="lower right", fontsize=8, facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)

    ax2 = fig.add_axes([0.70, 0.14, 0.27, 0.60])
    win_start = {b: story.ck[b][0] for b in story.bots}
    bars, texts = _bar_panel(ax2, story, win_start, win_start, "now")
    _bar_legend(ax2)

    cm = story.ck["matches"]
    images = []

    def frame_at(m: float):
        stop = int(np.searchsorted(story.match, m, side="right"))
        train_line.set_data(story.match[:stop], roll[:stop])
        # the evaluation line: checkpoints reached so far, plus a point interpolated to "now"
        reached = cm <= m
        xs = list(cm[reached]) + [m]
        ys = list(story.ck["mean"][reached] * 100) + [float(np.interp(m, cm, story.ck["mean"]) * 100)]
        eval_line.set_data(xs, ys)
        dot.set_data([m], [ys[-1]])
        counter.set_text(f"Match {int(round(m)):,}")
        rate_text.set_text(f"win rate vs the 4 bots: {ys[-1]:.0f}%")
        for bar, text, b in zip(bars, texts, story.bots):
            value = float(np.interp(m, cm, story.ck[b]) * 100)
            bar.set_width(value)
            text.set_x(value + 1.5)
            text.set_text(f"{value:.0f}")
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return Image.fromarray(arr.copy()).convert("P", palette=Image.ADAPTIVE, colors=96)

    for f in range(frames):
        images.append(frame_at(story.n_matches * _ease(f / (frames - 1))))
    plt.close(fig)
    durations = [frame_ms] * frames
    images += [images[-1]] * hold
    durations += [frame_ms] * hold
    durations[-1] = 2500                       # linger on the final picture
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    return len(images)


# --------------------------------------------------------------------------
# Before / after replay
# --------------------------------------------------------------------------

def pick_example(config, before_model, after_model, bot_name: str, seeds=range(400), max_turns: int = 17):
    """Find a game (same bot, same card draws) that the untrained agent loses and the trained one wins,
    short enough for a GIF. This is a chosen example; the numbers in the card are the honest summary."""
    from manaduel.agents.registry import ALL_BOT_FACTORIES
    from manaduel.agents.rl_agent import PPOAgent
    from manaduel.engine.game import play_game

    factory = ALL_BOT_FACTORIES[bot_name]
    fallback = None
    for seed in seeds:
        before = play_game(config, (PPOAgent(before_model), factory(seed)), seed=seed)
        after = play_game(config, (PPOAgent(after_model), factory(seed)), seed=seed)
        if after.winner != 0 or after.turns > max_turns:
            continue
        if before.winner == 1 and before.turns <= max_turns:
            return seed, before, after
        if before.winner != 0 and fallback is None:
            fallback = (seed, before, after)
    if fallback is None:
        raise RuntimeError("no suitable example game found; try a different bot or more seeds")
    return fallback


def _label_font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _replay_dict(config, result, names, seed) -> dict:
    return {"version": 1, "agents": list(names), "seed": seed, "winner": result.winner,
            "turns": result.turns, "rules": asdict(config.rules), "cards": [asdict(c) for c in config.cards],
            "start_hands": result.start_hands, "events": result.events}


def _replay_frames(config, result, names, seed, per_beat: int, size):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from manaduel.viewer.pygame_viewer import Fonts, H, W, build_view, render_to_surface

    view = build_view(_replay_dict(config, result, names, seed))
    fonts = Fonts()
    frames = []
    for idx in range(len(view.beats)):
        for f in range(per_beat):
            frames.append((idx, f / per_beat))
    frames.append((len(view.beats), 1.0))
    images = []
    for idx, p in frames:
        surf = render_to_surface(view, idx, p, fonts)
        images.append(Image.frombytes("RGB", (W, H), pygame.image.tostring(surf, "RGB")).resize(size, Image.LANCZOS))
    return images


def build_before_after(config, before_model, after_model, bot_name: str, path, replay_dir=None,
                       n_matches: int = 0, per_beat: int = 5, size=(480, 270), frame_ms: int = 110) -> dict:
    """Two games side by side: the untrained agent (left) and the trained agent (right), same opponent,
    same cards drawn. Returns what was shown."""
    seed, before, after = pick_example(config, before_model, after_model, bot_name)
    if replay_dir:
        from manaduel.sim.replay import save_replay

        Path(replay_dir).mkdir(parents=True, exist_ok=True)
        save_replay(Path(replay_dir) / "before.json", before, config, ("Untrained agent", bot_name), seed=seed)
        save_replay(Path(replay_dir) / "after.json", after, config, ("Trained agent", bot_name), seed=seed)
    left = _replay_frames(config, before, ("Untrained agent", bot_name), seed, per_beat, size)
    right = _replay_frames(config, after, ("Trained agent", bot_name), seed, per_beat, size)
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))              # let the shorter game rest on its final frame
    right += [right[-1]] * (n - len(right))
    header = 34
    gap = 6
    font = _label_font(19)
    captions = ("Match 0: untrained agent", f"After {n_matches:,} matches: trained agent" if n_matches
                else "Trained agent")
    frames = []
    for a, b in zip(left, right):
        canvas = Image.new("RGB", (size[0] * 2 + gap, size[1] + header), (20, 22, 34))
        canvas.paste(a, (0, header))
        canvas.paste(b, (size[0] + gap, header))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), captions[0], fill=(160, 166, 190), font=font)
        draw.text((size[0] + gap + 10, 6), captions[1], fill=(95, 209, 138), font=font)
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))
    durations = [frame_ms] * len(frames)
    durations[-1] = 2500
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return {"seed": seed, "bot": bot_name, "before_winner": before.winner, "after_winner": after.winner,
            "before_turns": before.turns, "after_turns": after.turns, "frames": len(frames)}


# --------------------------------------------------------------------------
# The caption
# --------------------------------------------------------------------------

def learned_facts(story: Story) -> list[str]:
    """What visibly changed in the agent's play, from the saved checkpoint tests. Only changes the
    numbers actually support are reported, so the caption can never claim something that did not happen."""
    ck, facts = story.ck, []
    m0, m1 = ck["hp_margin"][0], ck["hp_margin"][-1]
    if m1 - m0 >= 1.0:
        facts.append(f"it finishes games with an average HP lead of {m1:+.1f} (untrained: {m0:+.1f})")
    best = None
    for card in story.cards:
        a, b = ck[f"plays_{card}"][0], ck[f"plays_{card}"][-1]
        if a >= 0.3 and b / a >= 1.4 and (best is None or b / a > best[1] / best[0]):
            best = (a, b, card)
    if best:
        a, b, card = best
        facts.append(f"it plays {card.title()} {b / a:.1f}x as often ({a:.1f} to {b:.1f} times per game)")
    t0, t1 = ck["turns"][0], ck["turns"][-1]
    if abs(t1 - t0) >= 2:
        facts.append(f"games last {t1:.0f} turns on average instead of {t0:.0f}")
    return facts


def caption_text(story: Story, example: dict | None = None) -> str:
    first, last = story.ck["mean"][0] * 100, story.ck["mean"][-1] * 100
    greedy_end = story.ck["greedy"][-1] * 100 if "greedy" in story.ck else None
    took = duration_text(float(story.meta.get("seconds", 0)))
    games = int(story.ck["eval_games"][-1])
    lines = [
        "DRAFT: edit before posting. Every number below comes from this run's saved files.",
        "",
        "I built a small card game from scratch and trained an AI to play it using reinforcement learning (PPO).",
        "",
        "I did not program any strategy. The agent only gets win / lose feedback after each match.",
        "",
        f"- Match 0 (untrained): wins {first:.0f}% of games against my four scripted bots (on average)",
        f"- After {story.n_matches:,} matches (about {took} of training on a CPU, no GPU): {last:.0f}%"
        + (f", and {greedy_end:.0f}% against the strongest bot (greedy)" if greedy_end is not None else ""),
        f"- Tested on {games} fresh games per bot, seats swapped, draws count as half",
        "",
        *(["What changed in its play (measured on the test games):"]
          + [f"- {fact}" for fact in learned_facts(story)] + [""] if learned_facts(story) else []),
        "Honest limits: it beats simple scripted bots; it is not a strong general player. "
        "The point of the project was to understand the whole RL loop: environment, action masking, "
        "reward design, training and evaluation.",
        "",
        "#ReinforcementLearning #MachineLearning #Python #PPO #AI",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Everything at once
# --------------------------------------------------------------------------

def render_story(out_dir, config, bot_name: str = "greedy", footer: str = "", log=print) -> dict:
    from manaduel.agents.rl_agent import load_ppo_model

    out = Path(out_dir)
    story = load_story(out)
    made = {}
    plot_learning_curve(story, out / "learning_curve.png")
    made["learning_curve"] = out / "learning_curve.png"
    plot_behaviour(story, out / "behaviour.png")
    made["behaviour"] = out / "behaviour.png"
    make_card(story, out / "linkedin_card.png", footer=footer)
    made["card"] = out / "linkedin_card.png"
    log("static figures done; drawing the animation")
    frames = animate_learning(story, out / "learning_animation.gif")
    made["animation"] = out / "learning_animation.gif"
    log(f"animation: {frames} frames; building the before/after replay")
    before = load_ppo_model(out / "models" / "ckpt_00000000.zip")
    after = load_ppo_model(out / "models" / "final.zip")
    example = build_before_after(config, before, after, bot_name, out / "before_after.gif",
                                 replay_dir=out / "replays", n_matches=story.n_matches)
    made["before_after"] = out / "before_after.gif"

    from manaduel.story.matches import animate_journey, animate_matches

    log("drawing every match as a square, then the journey through training")
    animate_matches(story, out / "matches_animation.gif")
    made["matches"] = out / "matches_animation.gif"
    journey = animate_journey(story, config, out / "models", out / "journey.gif", bot_name)
    made["journey"] = out / "journey.gif"
    made["journey_info"] = journey
    (out / "linkedin_post.md").write_text(caption_text(story, example), encoding="utf-8")
    made["post"] = out / "linkedin_post.md"
    made["example"] = example
    return made
