def fit_elo(names, results, iterations: int = 500, step: float = 40.0) -> dict[str, float]:
    """Turn head-to-head results into one strength number per bot.

    Elo says: the expected score of A against B is 1 / (1 + 10 ** ((Rb - Ra) / 400)).
    We start everyone at 1000 and nudge ratings, many times, so the expected
    scores move toward the observed ones. Ratings are then re-centred on 1000.

    Observed scores are clamped to [0.01, 0.99]: a perfect 100% record would
    otherwise push a rating to infinity.
    """
    ratings = {n: 1000.0 for n in names}
    for _ in range(iterations):
        push = {n: 0.0 for n in names}
        weight = {n: 0.0 for n in names}
        for r in results:
            observed = min(0.99, max(0.01, r.score_a))
            gap = ratings[r.name_b] - ratings[r.name_a]
            expected = 1 / (1 + 10 ** (gap / 400))
            error = (observed - expected) * r.games
            push[r.name_a] += error
            push[r.name_b] -= error
            weight[r.name_a] += r.games
            weight[r.name_b] += r.games
        for n in names:
            if weight[n]:
                ratings[n] += step * push[n] / weight[n]
    mean = sum(ratings.values()) / len(ratings)
    return {n: ratings[n] - mean + 1000.0 for n in names}
