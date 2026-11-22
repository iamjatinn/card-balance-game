"""Does the card-effect tool actually work? Test it on changes we plant on purpose.

The real game is not perfectly balanced, so "does the tool flag the planted
card?" would be muddied by flags that were already there. Instead we ask the
question a patch tester cares about: "did this change move the card's power,
and can the tool see it?"

  1. Measure the baseline game once.
  2. For each planted change (e.g. fireball damage 7 -> 9), measure the changed
     game on fresh deals and compare each card's power with the baseline.
  3. DETECTED if the planted card's change is statistically significant
     (Bonferroni-corrected over all cards) and in the expected direction.
     ATTRIBUTED if the planted card also has the largest change of any card.
  4. FALSE-POSITIVE check: repeat step 2 with NO change (new deals only). Any
     card flagged as "changed" there is a false alarm.

Effects are measured against the MEAN card here: that reference moves linearly
(planting +d on one card shifts that card by 0.8d and each other card by -0.2d),
which keeps the comparison clean. Trials use independent deals, so the
difference between two measurements has standard error sqrt(se1^2 + se2^2).
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from statistics import NormalDist

from manaduel.analysis.autobalance import main_stat
from manaduel.analysis.card_effects import collect, estimate_effects
from manaduel.analysis.sweep import apply_overrides


@dataclass(frozen=True)
class Trial:
    card: str
    field: str
    old: int
    new: int
    rep: int
    change: float        # change in the planted card's power vs baseline
    z: float             # that change divided by its standard error
    detected: bool
    attributed: bool     # planted card has the largest |z| of all cards

    @property
    def relative(self) -> float:
        return (self.new - self.old) / self.old


@dataclass(frozen=True)
class NullTrial:
    rep: int
    max_abs_z: float
    false_alarm: bool    # some card looked "changed" although nothing was changed


@dataclass(frozen=True)
class ValidationResult:
    trials: list[Trial]
    null_trials: list[NullTrial]
    n_games: int
    alpha: float


def _measure(job):
    """Worker: play the games for one configuration and fit the effects (mean reference)."""
    config, factory, n_games, seed = job
    return estimate_effects(collect(config, factory, n_games, seed), reference="mean")


def _run_all(jobs, workers: int):
    if workers <= 1:
        return [_measure(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_measure, jobs, chunksize=1))


def _compare(planted, baseline, z_crit):
    """Per-card change in power and its z-score, planted game vs baseline game."""
    out = []
    for p, b in zip(planted.effects, baseline.effects):
        se = math.sqrt(p.se ** 2 + b.se ** 2)
        change = p.effect - b.effect
        out.append((change, change / se if se > 0 else 0.0))
    return out


def run_validation(config, factory, deltas=(1, 2, 4, -1, -2), n_games: int = 10_000,
                   baseline_games: int = 20_000, reps: int = 2, null_reps: int = 20,
                   seed: int = 0, alpha: float = 0.05, workers: int | None = None) -> ValidationResult:
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    k = len(config.cards)
    z_crit = NormalDist().inv_cdf(1 - alpha / (2 * k))

    # Plan every configuration to measure, then run them all (in parallel).
    planned = []   # (card, field, old, new, rep, config)
    for card in config.cards:
        field = main_stat(card)
        old = getattr(card, field)
        for delta in deltas:
            new = old + delta
            if new < 1:
                continue
            changed = apply_overrides(config, {f"cards.{card.name}.{field}": new})
            for rep in range(reps):
                planned.append((card.name, field, old, new, rep, changed))

    # Deal seeds: baseline uses seed 0 block; every other measurement gets its own block.
    block = 10_000_019
    jobs = [(config, factory, baseline_games, seed)]
    for i, (*_, changed) in enumerate(planned):
        jobs.append((changed, factory, n_games, seed + block * (i + 1)))
    for r in range(null_reps):
        jobs.append((config, factory, n_games, seed + block * (len(planned) + r + 1)))

    reports = _run_all(jobs, workers)
    baseline = reports[0]
    planted_reports = reports[1 : 1 + len(planned)]
    null_reports = reports[1 + len(planned):]

    names = [c.name for c in config.cards]
    trials = []
    for (card, field, old, new, rep, _), report in zip(planned, planted_reports):
        compared = _compare(report, baseline, z_crit)
        i = names.index(card)
        change, z = compared[i]
        sign = 1 if new > old else -1
        top = max(range(k), key=lambda j: abs(compared[j][1]))
        trials.append(Trial(card, field, old, new, rep, change, z,
                            detected=bool(sign * z > z_crit), attributed=bool(top == i)))

    null_trials = []
    for r, report in enumerate(null_reports):
        compared = _compare(report, baseline, z_crit)
        max_z = max(abs(z) for _, z in compared)
        null_trials.append(NullTrial(r, max_z, bool(max_z > z_crit)))

    return ValidationResult(trials, null_trials, n_games, alpha)


def summarize_by_size(result: ValidationResult) -> list[tuple[str, int, int, int]]:
    """(size label, trials, detected, attributed), grouped by how big the planted change was."""
    groups: dict[str, list[Trial]] = {}
    for t in result.trials:
        pct = abs(t.relative)
        label = "under 25%" if pct < 0.25 else "25% to 50%" if pct < 0.5 else "50% or more"
        groups.setdefault(label, []).append(t)
    order = ["under 25%", "25% to 50%", "50% or more"]
    return [(label, len(groups[label]), sum(t.detected for t in groups[label]),
             sum(t.attributed for t in groups[label])) for label in order if label in groups]


def format_validation(result: ValidationResult) -> str:
    lines = [
        f"Planted-change validation ({result.n_games} games per measurement, "
        f"alpha {result.alpha}, Bonferroni over cards)",
        "",
        f"{'planted change':<24} {'size':>6} {'rep':>3} {'power change':>13} {'z':>6}  detected  attributed",
    ]
    for t in result.trials:
        lines.append(
            f"{t.card + '.' + t.field + ' ' + str(t.old) + '->' + str(t.new):<24} "
            f"{t.relative:>+6.0%} {t.rep:>3} {t.change:>+13.1%} {t.z:>6.1f}  "
            f"{'yes' if t.detected else 'no':>8}  {'yes' if t.attributed else 'no':>10}"
        )
    lines += ["", "Detection by size of the planted change:"]
    for label, n, det, att in summarize_by_size(result):
        lines.append(f"  {label:<12} detected {det:>3}/{n:<3} ({det / n:5.0%})   attributed {att:>3}/{n:<3} ({att / n:5.0%})")
    total = len(result.trials)
    det_all = sum(t.detected for t in result.trials)
    lines.append(f"  {'all':<12} detected {det_all:>3}/{total:<3} ({det_all / total:5.0%})")

    fa = sum(n.false_alarm for n in result.null_trials)
    m = len(result.null_trials)
    lines += [
        "",
        f"False alarms with NO change planted: {fa}/{m} comparisons "
        f"(a perfectly calibrated test would give about {result.alpha:.0%})",
    ]
    return "\n".join(lines)


def write_csv(result: ValidationResult, path) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["card", "field", "old", "new", "relative_change", "rep", "power_change", "z",
                    "detected", "attributed"])
        for t in result.trials:
            w.writerow([t.card, t.field, t.old, t.new, f"{t.relative:.3f}", t.rep,
                        f"{t.change:.4f}", f"{t.z:.2f}", int(t.detected), int(t.attributed)])
