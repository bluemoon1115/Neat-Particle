from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


Vector = Tuple[float, ...]
SURROGATE_WIDTH = 0.28


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b)) ** 0.5


class SurrogateScorer:
    """Preference heuristic scorer for SPS acquisition experiments."""

    def __init__(self, dimensions: int, width: float = SURROGATE_WIDTH):
        if dimensions < 1:
            raise ValueError("SurrogateScorer needs at least one dimension")
        self.dimensions = dimensions
        self.width = float(width)
        self.positives: List[Vector] = []
        self.negatives: List[Vector] = []

    @property
    def fitted(self) -> bool:
        return bool(self.positives or self.negatives)

    def fit_from_records(self, records: Sequence[Tuple[Sequence[float], Sequence[Sequence[float]]]]) -> None:
        """Collect chosen samples as positives and rejected samples as negatives."""
        positives: List[Vector] = []
        negatives: List[Vector] = []
        for chosen, rejected_vectors in records:
            positives.append(tuple(float(value) for value in chosen))
            negatives.extend(tuple(float(value) for value in rejected) for rejected in rejected_vectors)
        self.positives = positives
        self.negatives = negatives

    def score(self, candidate: Sequence[float], rng: random.Random) -> float:
        """Score a candidate using preference reward, rejection penalty, and novelty."""
        point = tuple(float(value) for value in candidate)
        width_sq = self.width * self.width
        reward = sum(math.exp(-(_distance(point, positive) ** 2) / (2.0 * width_sq)) for positive in self.positives)
        penalty = sum(math.exp(-(_distance(point, negative) ** 2) / (2.0 * width_sq)) for negative in self.negatives)

        observed = self.positives + self.negatives
        uncertainty = 1.0
        if observed:
            uncertainty = min(_distance(point, observed_point) for observed_point in observed) / math.sqrt(self.dimensions)

        return reward - 0.65 * penalty + 0.35 * uncertainty + rng.random() * 0.001
