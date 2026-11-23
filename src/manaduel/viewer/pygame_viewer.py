"""Graphical replay viewer: watch a saved game as two bots fighting.

It only READS a replay JSON (written by `watch --save`); it never touches the
game engine. Each card played becomes one animated "beat".

    python -m manaduel.cli view replays/greedy_vs_defensive_seed8.json

Keys: SPACE pause/play, LEFT/RIGHT step, UP/DOWN speed, R restart, Q or ESC quit.
"""

import math
import os
from dataclasses import dataclass

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from manaduel.sim.replay import load_replay  # noqa: E402

W, H = 960, 540
BEAT_SECONDS = 1.5

BG = (20, 22, 34)
FLOOR = (38, 42, 62)
TEXT = (230, 232, 245)
DIM = (120, 126, 150)
HP_BACK = (50, 30, 36)
HP_FRONT = (96, 214, 120)
HP_LOW = (230, 90, 70)
SHIELD_COL = (96, 160, 240)
MANA_COL = (250, 200, 70)
PLAYER_COL = [(86, 142, 230), (232, 128, 76)]

ATTACK_COL = (232, 92, 72)
HEAL_COL = (92, 206, 124)
DRAIN_COL = (192, 96, 206)


# --------------------------------------------------------------------------
# Turning a replay into animated beats
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Snap:
    """Both players' numbers at one moment."""
    hp: tuple
    shield: tuple
    mana: tuple
    hand: tuple
    poison: tuple = (0, 0)


@dataclass(frozen=True)
class Beat:
    index: int
    turn: int
    player: int
    card: dict
    before: Snap
    after: Snap
    absorbed: int   # damage the opponent's shield soaked up


@dataclass(frozen=True)
class View:
    names: tuple
    card_names: tuple
    cards: tuple
    max_hp: int
    max_mana: int
    max_shield: int
    start: Snap
    beats: tuple
    winner: object
    turns: int


def _snap(hp, shield, mana, hand, poison=(0, 0)) -> Snap:
    return Snap(tuple(hp), tuple(shield), tuple(mana), tuple(tuple(h) for h in hand), tuple(poison))


def build_view(replay: dict) -> View:
    rules = replay["rules"]
    cards = replay["cards"]
    by_name = {c["name"]: c for c in cards}
    snap = _snap(
        [rules["max_hp"]] * 2,
        [0, 0],
        [min(1, rules["max_mana"]), 0],
        replay["start_hands"] or [[0] * len(cards)] * 2,
    )
    start = snap
    beats = []
    for e in replay["events"]:
        after = _snap(e["hp"], e["shield"], e["mana"], e["hand"], e.get("poison", (0, 0)))
        if e["action"] != "end_turn":
            card = by_name[e["action"]]
            # the opponent's shield only ever shrinks on our turn, by absorbing damage
            absorbed = max(0, snap.shield[1 - e["player"]] - after.shield[1 - e["player"]])
            beats.append(Beat(len(beats), e["turn"], e["player"], card, snap, after, absorbed))
        snap = after  # end_turn events carry the refilled mana and drawn card into the next beat
    return View(
        names=tuple(replay["agents"]),
        card_names=tuple(c["name"] for c in cards),
        cards=tuple(cards),
        max_hp=rules["max_hp"],
        max_mana=rules["max_mana"],
        max_shield=rules.get("max_shield", 6),
        start=start,
        beats=tuple(beats),
        winner=replay["winner"],
        turns=replay["turns"],
    )


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _ease(x):
    x = _clamp(x)
    return x * x * (3 - 2 * x)


def _lerp(a, b, t):
    return a + (b - a) * t


POISON_COL = (140, 176, 44)
DRAW_COL = (48, 170, 190)
COMBO_COL = (232, 140, 48)


def card_color(card: dict):
    """A card's colour, from what it does (replays from before the extra mechanics lack those keys)."""
    if card.get("poison"):
        return POISON_COL
    if card.get("draw"):
        return DRAW_COL
    if card.get("combo"):
        return COMBO_COL
    if card["damage"] and card["heal"]:
        return DRAIN_COL
    if card["damage"]:
        return ATTACK_COL
    if card["heal"]:
        return HEAL_COL
    return SHIELD_COL


class Fonts:
    def __init__(self):
        pygame.font.init()
        self.big = pygame.font.Font(None, 54)
        self.mid = pygame.font.Font(None, 30)
        self.small = pygame.font.Font(None, 22)
        self.tiny = pygame.font.Font(None, 18)


def _text(surface, font, text, color, center=None, left=None, top=None):
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center is not None:
        rect.center = center
    else:
        rect.topleft = (left, top)
    surface.blit(img, rect)


def _shown(view: View, idx: int, p: float):
    """Numbers to display: smoothly moving from a beat's before-snapshot to its after-snapshot."""
    if idx >= len(view.beats):
        last = view.beats[-1].after if view.beats else view.start
        return last, last, 1.0
    beat = view.beats[idx]
    return beat.before, beat.after, _ease((p - 0.35) / 0.4)


def _draw_robot(surface, cx, ground, color, flash, shield_alpha, facing):
    body = pygame.Rect(0, 0, 84, 110)
    body.midbottom = (cx, ground)
    shade = tuple(int(_lerp(c, 255, flash)) if i == 0 else int(_lerp(c, 90, flash)) for i, c in enumerate(color))
    pygame.draw.rect(surface, shade, body, border_radius=12)
    pygame.draw.rect(surface, (255, 255, 255), body, width=2, border_radius=12)
    head = pygame.Rect(0, 0, 60, 50)
    head.midbottom = (cx, body.top - 6)
    pygame.draw.rect(surface, shade, head, border_radius=10)
    pygame.draw.rect(surface, (255, 255, 255), head, width=2, border_radius=10)
    for dx in (-14, 14):
        pygame.draw.circle(surface, (255, 255, 255), (head.centerx + dx + facing * 4, head.centery), 7)
        pygame.draw.circle(surface, (20, 20, 30), (head.centerx + dx + facing * 6, head.centery), 3)
    pygame.draw.line(surface, (255, 255, 255), (cx, head.top), (cx, head.top - 14), 3)
    pygame.draw.circle(surface, MANA_COL, (cx, head.top - 16), 5)
    arm = pygame.Rect(0, 0, 18, 56)
    arm.midtop = (body.centerx + facing * 52, body.top + 10)
    pygame.draw.rect(surface, shade, arm, border_radius=8)
    if shield_alpha > 0:
        ring = pygame.Surface((190, 220), pygame.SRCALPHA)
        pygame.draw.ellipse(ring, (*SHIELD_COL, int(120 * shield_alpha)), ring.get_rect(), width=6)
        pygame.draw.ellipse(ring, (*SHIELD_COL, int(40 * shield_alpha)), ring.get_rect())
        surface.blit(ring, ring.get_rect(center=(cx, body.centery - 24)))


def _draw_bars(surface, fonts, view, side, cx, hp, shield, mana, poison=0.0):
    bar = pygame.Rect(0, 0, 270, 22)
    bar.center = (cx, 118)
    pygame.draw.rect(surface, HP_BACK, bar, border_radius=8)
    frac = _clamp(hp / view.max_hp)
    fill = bar.copy()
    fill.width = int(bar.width * frac)
    pygame.draw.rect(surface, HP_FRONT if frac > 0.35 else HP_LOW, fill, border_radius=8)
    pygame.draw.rect(surface, (255, 255, 255), bar, width=2, border_radius=8)
    _text(surface, fonts.small, f"{max(0, round(hp))}/{view.max_hp}", TEXT, center=bar.center)

    for i in range(view.max_shield):
        r = pygame.Rect(bar.left + i * 24, bar.bottom + 8, 20, 12)
        on = i < round(shield)
        pygame.draw.rect(surface, SHIELD_COL if on else (44, 50, 72), r, border_radius=3)
    for i in range(view.max_mana):
        cxm, cym = bar.left + 10 + i * 24, bar.bottom + 38
        pts = [(cxm, cym - 9), (cxm + 8, cym), (cxm, cym + 9), (cxm - 8, cym)]
        on = i < round(mana)
        pygame.draw.polygon(surface, MANA_COL if on else (60, 56, 40), pts)
    if round(poison) > 0:
        _text(surface, fonts.small, f"poison {round(poison)}", POISON_COL, left=bar.left, top=bar.bottom + 56)
    _text(surface, fonts.mid, view.names[side], PLAYER_COL[side], center=(cx, 80))


def _draw_hand(surface, fonts, view, side, hand, highlight=None):
    n = len(view.card_names)
    cols = min(n, 5)                      # more than five card types wrap onto a second row
    rows = -(-n // cols)
    chip_w, chip_h, gap = 84, (56 if rows == 1 else 48), 6
    total = cols * chip_w + (cols - 1) * gap
    x0 = 24 if side == 0 else W - 24 - total
    y0 = 420 if rows == 1 else 408
    for i, name in enumerate(view.card_names):
        count = hand[i] if i < len(hand) else 0
        r = pygame.Rect(x0 + (i % cols) * (chip_w + gap), y0 + (i // cols) * (chip_h + gap), chip_w, chip_h)
        base = card_color(view.cards[i])
        col = base if count > 0 else (46, 50, 70)
        pygame.draw.rect(surface, col, r, border_radius=8)
        border = (255, 255, 255) if highlight == name else (90, 96, 120)
        pygame.draw.rect(surface, border, r, width=3 if highlight == name else 1, border_radius=8)
        _text(surface, fonts.tiny, name, TEXT if count else DIM, center=(r.centerx, r.top + 14))
        _text(surface, fonts.mid, f"x{count}", TEXT if count else DIM, center=(r.centerx, r.bottom - 16))


def _float_text(surface, fonts, text, color, x, y, c):
    """Text that drifts upward and fades during the settle phase c in [0, 1]."""
    img = fonts.mid.render(text, True, color)
    img.set_alpha(int(255 * (1 - _clamp(c - 0.4, 0, 0.6) / 0.6)))
    surface.blit(img, img.get_rect(center=(x, y - 40 * c)))


def draw(surface, fonts, view: View, idx: int, p: float, help_text: str = "") -> None:
    """Draw one frame. idx is the beat (len(beats) = game over), p is progress in that beat."""
    surface.fill(BG)
    pygame.draw.rect(surface, FLOOR, (0, 380, W, H - 380))
    xs = (230, 730)
    ground = 372

    before, after, k = _shown(view, idx, p)
    finished = idx >= len(view.beats)
    beat = None if finished else view.beats[idx]

    # animation phases
    a = _clamp(p / 0.25)
    b = _clamp((p - 0.25) / 0.45)
    c = _clamp((p - 0.7) / 0.3)

    hp = [_lerp(before.hp[i], after.hp[i], k) for i in range(2)]
    shield = [_lerp(before.shield[i], after.shield[i], k) for i in range(2)]
    mana = [_lerp(before.mana[i], after.mana[i], k) for i in range(2)]
    poison = [_lerp(before.poison[i], after.poison[i], k) for i in range(2)]
    hand = before.hand if (beat is None or k < 0.5) else after.hand
    if beat is not None and beat.index == 0 and p < 0.35:
        hand = view.start.hand

    actor = beat.player if beat else None
    target = 1 - actor if beat else None
    damage = beat.card["damage"] if beat else 0
    gains_shield = bool(beat and beat.card["shield"] > 0)
    heals = bool(beat and beat.card["heal"] > 0)
    poisons = int(beat.card.get("poison", 0)) if beat else 0

    for side in (0, 1):
        facing = 1 if side == 0 else -1
        x = xs[side]
        flash = 0.0
        if beat and side == target and damage and b >= 1.0:
            flash = _clamp(1 - c * 1.5)
        shake = 0
        if beat and side == target and damage and b >= 1.0 and c < 0.6:
            shake = math.sin(c * 40) * 6 * (1 - c)
        lunge = 0
        if beat and side == actor and damage:
            lunge = math.sin(math.pi * b) * 48 * facing
        shield_vis = _clamp(shield[side] / max(1, view.max_shield) * 1.4)
        if beat and side == actor and gains_shield:
            shield_vis = max(shield_vis, math.sin(math.pi * b))
        _draw_robot(surface, int(x + lunge + shake), ground, PLAYER_COL[side], flash, shield_vis, facing)
        _draw_bars(surface, fonts, view, side, x, hp[side], shield[side], mana[side], poison[side])
        _draw_hand(surface, fonts, view, side, hand[side], highlight=beat.card["name"] if beat and side == actor else None)

    if beat:
        col = card_color(beat.card)
        # card banner
        banner = pygame.Rect(0, 0, 250, 46)
        banner.center = (W // 2, 70 - int((1 - a) * 30))
        pygame.draw.rect(surface, col, banner, border_radius=12)
        pygame.draw.rect(surface, (255, 255, 255), banner, width=2, border_radius=12)
        _text(surface, fonts.mid, f"{beat.card['name'].upper()}  ({beat.card['cost']} mana)", (15, 15, 25), center=banner.center)
        # projectile
        if (damage or poisons) and 0 < b < 1:
            sx = xs[actor] + (60 if actor == 0 else -60)
            tx = xs[target]
            px = _lerp(sx, tx, _ease(b))
            py = 300 - math.sin(math.pi * b) * 24
            pygame.draw.circle(surface, col, (int(px), int(py)), 16)
            pygame.draw.circle(surface, (255, 255, 255), (int(px), int(py)), 16, 2)
        # healing sparkles
        if heals and b > 0:
            for j in range(6):
                ang = j * math.tau / 6 + b * 4
                sxp = xs[actor] + math.cos(ang) * 60
                syp = 290 - b * 70 + math.sin(ang) * 16
                pygame.draw.circle(surface, HEAL_COL, (int(sxp), int(syp)), 5)
        # floating numbers (shown during the settle phase)
        if c > 0 or b >= 1:
            cc = c
            # Labels sit in the open space beside each robot (toward the middle), never on its head.
            lx = [xs[0] + 135, xs[1] - 135]
            if damage:
                dealt = round(before.hp[target] - after.hp[target])
                if dealt > 0:
                    _float_text(surface, fonts, f"-{dealt}", ATTACK_COL, lx[target], 300, cc)
                if beat.absorbed:
                    _float_text(surface, fonts, f"{beat.absorbed} blocked", SHIELD_COL, lx[target], 262, cc)
            if beat.card["heal"]:
                gained = round(after.hp[actor] - before.hp[actor])
                if gained > 0:
                    _float_text(surface, fonts, f"+{gained}", HEAL_COL, lx[actor], 300, cc)
            if beat.card["shield"]:
                gained = round(after.shield[actor] - before.shield[actor])
                if gained > 0:
                    _float_text(surface, fonts, f"+{gained} shield", SHIELD_COL, lx[actor], 300, cc)
            if poisons:
                _float_text(surface, fonts, f"poison +{poisons}", POISON_COL, lx[target], 300, cc)
            outer = [xs[0] - 135, xs[1] + 135]       # the far side, away from the move log in the middle
            if beat.card.get("draw"):
                _float_text(surface, fonts, f"+{beat.card['draw']} cards", DRAW_COL, outer[actor], 300, cc)
            if beat.card.get("combo") and damage:
                bonus = round(before.hp[target] - after.hp[target]) + beat.absorbed - damage
                if bonus > 0:
                    _float_text(surface, fonts, f"COMBO +{bonus}", COMBO_COL, outer[actor], 300, cc)

    # header and move log
    turn = beat.turn if beat else view.turns
    _text(surface, fonts.small, f"Turn {turn}", DIM, center=(W // 2, 20))
    shown = view.beats[max(0, idx - 3): idx + 1] if not finished else view.beats[-4:]
    for row, bt in enumerate(shown):
        who = view.names[bt.player]
        _text(surface, fonts.tiny, f"T{bt.turn:02d} {who} plays {bt.card['name']}", DIM if bt is not beat else TEXT,
              center=(W // 2, 190 + row * 20))  # the empty space between the two robots

    if finished:
        veil = pygame.Surface((W, 120), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 170))
        surface.blit(veil, (0, 190))
        if view.winner is None:
            msg, col = f"DRAW after {view.turns} turns", TEXT
        else:
            msg, col = f"{view.names[view.winner]} WINS on turn {view.turns}", PLAYER_COL[view.winner]
        _text(surface, fonts.big, msg, col, center=(W // 2, 250))
    if help_text:
        _text(surface, fonts.tiny, help_text, DIM, center=(W // 2, 524))


def render_to_surface(view: View, idx: int, p: float, fonts: Fonts | None = None) -> "pygame.Surface":
    surface = pygame.Surface((W, H))
    draw(surface, fonts or Fonts(), view, idx, p)
    return surface


# --------------------------------------------------------------------------
# Playback controller (no drawing, easy to test)
# --------------------------------------------------------------------------

class Player:
    def __init__(self, view: View, speed: float = 1.0):
        self.view = view
        self.speed = speed
        self.playing = True
        self.idx = 0
        self.t = 0.0

    @property
    def finished(self) -> bool:
        return self.idx >= len(self.view.beats)

    @property
    def progress(self) -> float:
        return 1.0 if self.finished else _clamp(self.t / BEAT_SECONDS)

    def update(self, dt: float) -> None:
        if not self.playing or self.finished:
            return
        self.t += dt * self.speed
        while self.t >= BEAT_SECONDS and not self.finished:
            self.t -= BEAT_SECONDS
            self.idx += 1
        if self.finished:
            self.t = 0.0

    def step(self, direction: int) -> None:
        self.playing = False
        n = len(self.view.beats)
        self.idx = max(0, min(n, self.idx + direction))
        self.t = BEAT_SECONDS * 0.95

    def restart(self) -> None:
        self.idx, self.t, self.playing = 0, 0.0, True

    def change_speed(self, factor: float) -> None:
        self.speed = max(0.25, min(8.0, self.speed * factor))


# --------------------------------------------------------------------------
# Window, PNG and GIF
# --------------------------------------------------------------------------

HELP = "SPACE pause/play   LEFT/RIGHT step   UP/DOWN speed   R restart   Q quit"


def run_window(path, speed: float = 1.0) -> None:
    view = build_view(load_replay(path))
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"Card duel replay: {view.names[0]} vs {view.names[1]}")
    fonts = Fonts()
    clock = pygame.time.Clock()
    player = Player(view, speed)
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    player.playing = not player.playing
                elif event.key == pygame.K_RIGHT:
                    player.step(1)
                elif event.key == pygame.K_LEFT:
                    player.step(-1)
                elif event.key == pygame.K_UP:
                    player.change_speed(1.5)
                elif event.key == pygame.K_DOWN:
                    player.change_speed(1 / 1.5)
                elif event.key == pygame.K_r:
                    player.restart()
        player.update(dt)
        status = f"{HELP}   speed x{player.speed:.2f}{'' if player.playing else '   [paused]'}"
        draw(screen, fonts, view, player.idx, player.progress, status)
        pygame.display.flip()
    pygame.quit()


def save_png(path, replay_path, beat: int | None = None, progress: float = 0.85) -> None:
    """Render one frame to a PNG (no window needed). Defaults to the middle of the game."""
    view = build_view(load_replay(replay_path))
    idx = len(view.beats) // 2 if beat is None else beat
    pygame.image.save(render_to_surface(view, idx, progress), str(path))


def save_gif(path, replay_path, fps: int = 15, speed: float = 1.6, size=(640, 360), max_beats=None) -> int:
    """Export the whole replay as an animated GIF. Returns the number of frames."""
    from PIL import Image

    view = build_view(load_replay(replay_path))
    fonts = Fonts()
    beats = len(view.beats) if max_beats is None else min(max_beats, len(view.beats))
    per_beat = max(2, round(BEAT_SECONDS / speed * fps))
    frames = []
    for idx in range(beats):
        for f in range(per_beat):
            surf = render_to_surface(view, idx, f / per_beat, fonts)
            frames.append(surf)
    hold = [render_to_surface(view, len(view.beats), 1.0, fonts)] * (fps * 2) if beats == len(view.beats) else []
    images = []
    for surf in frames + hold:
        img = Image.frombytes("RGB", (W, H), pygame.image.tostring(surf, "RGB")).resize(size, Image.LANCZOS)
        images.append(img.convert("P", palette=Image.ADAPTIVE, colors=64))
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return len(images)
