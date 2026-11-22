import math


def wilson_interval(points: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate.

    points = wins + 0.5 * draws, n = games played. Unlike the naive
    'p +- 1.96 * sqrt(p(1-p)/n)', this stays inside [0, 1] and behaves
    sensibly for small n or extreme win rates.
    """
    if n == 0:
        return 0.0, 1.0
    p = points / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)
