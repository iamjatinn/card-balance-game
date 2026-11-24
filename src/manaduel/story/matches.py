"""Two ways to SEE the agent fail, learn and then win.

animate_matches  every training match is one coloured square (red = lost, green = won, grey = draw),
                 filled in the order the matches were played, with a running record. The early
                 matches are mostly red; later ones are mostly green. These are the real results.
animate_journey  one real game at several stages of training, always against the same bot and with
                 the same cards drawn: the checkpoint's policy plays it, so you can watch the early
                 agent lose and the later ones win. It re-plays each checkpoint's policy; it is not a
                 recording of the exact training matches (those were not saved move by move).
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from manaduel.story.render import (  # noqa: E402
    ACCENT,
    BG,
    LINE,
    MUTED,
    TEXT,
    Story,
    _ease,
    _label_font,
    _replay_frames,
    _style,
    duration_text,
    rolling_mean,
)

WIN_RGB = (95, 209, 138)
LOSS_RGB = (224, 96, 74)
DRAW_RGB = (122, 128, 153)
EMPTY_RGB = (28, 31, 43)


def grid_shape(n: int, aspect: float = 1.55) -> tuple[int, int]:
    """Rows and columns for n squares laid out in a block about `aspect` times wider than tall."""
    cols = int(math.ceil(math.sqrt(n * aspect)))
    return int(math.ceil(n / cols)), cols


def result_colors(results) -> np.ndarray:
    """An (n, 3) array of colours: green for a win (1), red for a loss (0), grey for a draw (0.5)."""
    results = np.asarray(results, dtype=float)
    colors = np.empty((len(results), 3), dtype=np.uint8)
    colors[results >= 0.99] = WIN_RGB
    colors[results <= 0.01] = LOSS_RGB
    colors[(results > 0.01) & (results < 0.99)] = DRAW_RGB
    return colors


def segment_win_rates(story: Story, k: int = 500) -> tuple[float, float]:
    """Win rate (draw = half) over the first k and the last k training matches."""
    k = min(k, len(story.result) // 2)
    return float(story.result[:k].mean()), float(story.result[-k:].mean())


def match_frame_counts(n: int, frames: int, power: float = 2.2) -> list[int]:
    """How many matches are shown in each frame. Early frames are spaced closely, because the failing
    phase is where most of the learning happens."""
    counts = [max(1, int(round(n * _ease(f / (frames - 1), power)))) for f in range(frames)]
    return list(dict.fromkeys(counts))          # drop repeats (the very first frames can coincide)


ZOOM_MATCHES = 600     # the zoomed grid shows this many of the very first matches, with big squares


def _grid_axes(fig, rect, n: int, aspect: float):
    """An axes holding a grid of n squares (initially empty); returns (axes, image, pixels, rows, cols)."""
    rows, cols = grid_shape(n, aspect)
    ax = fig.add_axes(rect)
    ax.axis("off")
    pixels = np.zeros((rows * cols, 3), dtype=np.uint8)
    pixels[:] = EMPTY_RGB
    image = ax.imshow(pixels.reshape(rows, cols, 3), interpolation="nearest", aspect="auto")
    return ax, image, pixels, rows, cols


def animate_matches(story: Story, path, frames: int = 100, hold: int = 16, size=(1200, 675),
                    frame_ms: int = 80) -> int:
    """Every training match as a square, in order. Left: a zoom on the first matches (big squares, so the
    early failures are easy to see). Middle: all matches. Right: the running win rate."""
    W, H = size
    n = len(story.result)
    zoom_n = min(ZOOM_MATCHES, n)
    colors = result_colors(story.result)
    wins = np.cumsum(story.result >= 0.99)
    losses = np.cumsum(story.result <= 0.01)
    draws = np.cumsum((story.result > 0.01) & (story.result < 0.99))
    window = int(max(100, n // 60))
    roll = rolling_mean(story.result, window) * 100
    first_k, last_k = segment_win_rates(story)
    k = min(500, n // 2)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.03, 0.935, "Every match the agent played: it fails, learns, then wins", color=TEXT, fontsize=19,
             fontweight="bold", va="center")
    counter = fig.text(0.03, 0.865, "", color=ACCENT, fontsize=26, fontweight="bold", va="center")
    record = fig.text(0.28, 0.865, "", color=TEXT, fontsize=13, va="center")

    _, zoom_img, zoom_px, zr, zc = _grid_axes(fig, [0.03, 0.17, 0.26, 0.58], zoom_n, 0.80)
    fig.text(0.03, 0.775, f"the first {zoom_n} matches (zoomed in)", color=TEXT, fontsize=11, va="center")
    _, full_img, full_px, fr, fc = _grid_axes(fig, [0.33, 0.17, 0.34, 0.58], n, 0.96)
    fig.text(0.33, 0.775, f"all {n:,} matches", color=TEXT, fontsize=11, va="center")

    for x, label, color in ((0.03, "won", WIN_RGB), (0.10, "lost", LOSS_RGB), (0.17, "drew", DRAW_RGB)):
        fig.text(x, 0.115, "■", color=tuple(c / 255 for c in color), fontsize=15, va="center")
        fig.text(x + 0.016, 0.115, label, color=MUTED, fontsize=11, va="center")
    fig.text(0.03, 0.06, "each square = one training match, in the order played (left to right, top to bottom)",
             color=MUTED, fontsize=10, va="center")

    ax2 = fig.add_axes([0.74, 0.22, 0.23, 0.31])
    _style(ax2)
    ax2.set_xlim(0, n)
    ax2.set_ylim(0, 100)
    ax2.set_title(f"win rate, last {window:,} matches", color=TEXT, fontsize=10)
    ax2.set_xlabel("matches", color=MUTED, fontsize=9)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0"))
    line, = ax2.plot([], [], color=LINE, lw=1.8)
    dot, = ax2.plot([], [], "o", color=ACCENT, ms=7)
    big = fig.text(0.855, 0.66, "", color=ACCENT, fontsize=34, fontweight="bold", ha="center", va="center")
    fig.text(0.855, 0.595, "win rate (draws count half)", color=MUTED, fontsize=9, ha="center", va="center")
    end_note = fig.text(0.855, 0.075, "", color=TEXT, fontsize=11, ha="center", va="center", linespacing=1.6)

    images = []
    shown = {"zoom": 0, "full": 0}
    counts = match_frame_counts(n, frames)

    def fill(pixels, key, m):
        if m > shown[key]:
            pixels[shown[key]:m] = colors[shown[key]:m]
            shown[key] = m

    def draw(m: int, final: bool = False):
        fill(zoom_px, "zoom", min(m, zoom_n))
        fill(full_px, "full", m)
        zoom_img.set_data(zoom_px.reshape(zr, zc, 3))
        full_img.set_data(full_px.reshape(fr, fc, 3))
        counter.set_text(f"Match {m:,}")
        record.set_text(f"so far:  {wins[m - 1]:,} won   {losses[m - 1]:,} lost   {draws[m - 1]:,} drew")
        line.set_data(np.arange(1, m + 1), roll[:m])
        dot.set_data([m], [roll[m - 1]])
        big.set_text(f"{roll[m - 1]:.0f}%")
        if final:
            end_note.set_text(f"first {k} matches: {first_k * 100:.0f}%\nlast {k} matches: {last_k * 100:.0f}%")
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return Image.fromarray(arr.copy()).convert("P", palette=Image.ADAPTIVE, colors=96)

    for m in counts:
        images.append(draw(m))
    last = draw(n, final=True)
    images += [last] * hold
    plt.close(fig)
    durations = [frame_ms] * len(images)
    durations[-1] = 3000                        # linger on the final picture
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    return len(images)


# --------------------------------------------------------------------------
# The journey: one game at several stages of training
# --------------------------------------------------------------------------

def pick_stage_checkpoints(story: Story, stages: int = 5) -> list[int]:
    """Checkpoint rows to show: the untrained agent, then roughly 3%, 12% and 40% of the way, then the end."""
    n = story.n_matches
    targets = [0.0, 0.03, 0.12, 0.40][: max(1, stages - 1)] + [1.0]
    picks = []
    for t in targets:
        idx = int(np.searchsorted(story.ck["matches"], t * n, side="left"))
        idx = min(idx, len(story.ck["matches"]) - 1)
        if idx not in picks:
            picks.append(idx)
    return picks


def pick_journey_game(config, models: list, bot_name: str, seeds=range(500), max_turns: int = 16,
                      expected_first_win: int = 1):
    """Find a game (same bot, same cards) that the untrained agent loses and the final agent wins, that
    never goes win-then-loss as training advances, and whose first win comes at `expected_first_win`
    (the stage where the agent's REAL win rate against this bot first reaches 50%), so the example agrees
    with the honest statistics. Short games are preferred (small GIF). It is still a chosen example; the
    win rate printed on each stage is the honest picture. Returns (seed, results)."""
    from manaduel.agents.registry import ALL_BOT_FACTORIES
    from manaduel.agents.rl_agent import PPOAgent
    from manaduel.engine.game import play_game

    factory = ALL_BOT_FACTORIES[bot_name]
    best, best_score = None, None
    for seed in seeds:
        results = [play_game(config, (PPOAgent(m), factory(seed)), seed=seed) for m in models]
        if any(r.turns > max_turns for r in results):
            continue
        won = [r.winner == 0 for r in results]
        if won[0] or not won[-1]:
            continue                                   # want: the untrained agent loses, the last one wins
        monotone = all(not (won[i] and not won[i + 1]) for i in range(len(won) - 1))
        losing_stages = won.index(True)                # stages lost before the first win
        total_turns = sum(r.turns for r in results)
        score = (monotone, -abs(losing_stages - expected_first_win), -total_turns)
        if best_score is None or score > best_score:
            best, best_score = (seed, results), score
    if best is None:
        raise RuntimeError("no suitable journey game found; try a different bot or more seeds")
    return best


def animate_journey(story: Story, config, models_dir, path, bot_name: str = "greedy", stages: int = 5,
                    per_beat: int = 4, size=(480, 270), frame_ms: int = 100) -> dict:
    """One game per stage, side by side in time: untrained, early, middle, later, final."""
    from manaduel.agents.rl_agent import load_ppo_model

    picks = pick_stage_checkpoints(story, stages)
    models = [load_ppo_model(Path(models_dir) / f"ckpt_{int(story.ck['steps'][i]):08d}.zip") for i in picks]
    rates = [float(story.ck[bot_name][i]) for i in picks] if bot_name in story.ck else [0.0] * len(picks)
    expected = next((k for k, r in enumerate(rates) if r >= 0.5), len(picks) - 1)
    seed, results = pick_journey_game(config, models, bot_name, expected_first_win=expected)
    header = 62
    font_big, font_small = _label_font(22), _label_font(15)
    frames, durations = [], []
    stage_info = []
    for i, result in zip(picks, results):
        matches = int(story.ck["matches"][i])
        rate = float(story.ck[bot_name][i]) * 100 if bot_name in story.ck else float("nan")
        games = int(story.ck["eval_games"][i])
        label = "untrained" if matches == 0 else f"after {matches:,} matches"
        names = (f"Agent @ match {matches:,}", bot_name)
        stage_frames = _replay_frames(config, result, names, seed, per_beat, size)
        verdict = "WON" if result.winner == 0 else "LOST" if result.winner == 1 else "DREW"
        colour = (95, 209, 138) if verdict == "WON" else (224, 96, 74) if verdict == "LOST" else (160, 166, 190)
        for k, frame in enumerate(stage_frames):
            canvas = Image.new("RGB", (size[0], size[1] + header), (20, 22, 34))
            canvas.paste(frame, (0, header))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 6), f"Stage: {label}", fill=(232, 234, 242), font=font_big)
            draw.text((10, 36), f"its win rate vs the {bot_name} bot at this point: {rate:.0f}% "
                                f"({games} test games)", fill=(139, 145, 168), font=font_small)
            if k == len(stage_frames) - 1:
                draw.text((size[0] - 110, 8), verdict, fill=colour, font=font_big)
            frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))
            durations.append(1800 if k == len(stage_frames) - 1 else frame_ms)
        stage_info.append({"matches": matches, "win_rate_vs_bot": rate / 100, "this_game": verdict.lower(),
                           "turns": result.turns})
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return {"seed": seed, "bot": bot_name, "stages": stage_info, "frames": len(frames),
            "duration": duration_text(sum(durations) / 1000)}
