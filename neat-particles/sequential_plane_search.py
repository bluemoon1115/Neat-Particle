from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from bayesian_preference_optimizer import PreferentialBayesianOptimizer

EPSILON = 1e-6


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _clamp_vector(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(_clamp(value, EPSILON, 1.0 - EPSILON) for value in values)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b)) ** 0.5


def _norm(values: Sequence[float]) -> float:
    return sum(value * value for value in values) ** 0.5


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(values: Sequence[float]) -> Tuple[float, ...]:
    length = _norm(values)
    if length <= EPSILON:
        return tuple(0.0 for _ in values)
    return tuple(value / length for value in values)


@dataclass(frozen=True)
class SearchVector:
    """Normalized SPS vector used to tune selected genome connection weights."""

    vector: Tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", _clamp_vector(self.vector))

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "SearchVector":
        """Build a clamped normalized vector from any numeric sequence."""
        return cls(tuple(vector))


@dataclass(frozen=True)
class PreferenceRecord:
    chosen: Tuple[float, ...]
    rejected: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class PlaneSample:
    search_vector: SearchVector
    u_coeff: int
    v_coeff: int


class SequentialPlaneSearch:
    """Preference-guided 3x3 sequential plane search in normalized coordinates."""

    def __init__(
        self,
        dimensions: int,
        start_vector: Sequence[float],
        seed: Optional[int] = None,
        plane_radius: float = 0.22,
    ):
        if dimensions < 1:
            raise ValueError("SequentialPlaneSearch needs at least one dimension")
        if len(start_vector) != dimensions:
            raise ValueError(f"start_vector needs {dimensions} values, got {len(start_vector)}")

        self.rng = random.Random(seed)
        self.plane_radius = plane_radius
        self.dimensions = dimensions
        self.default_start = _clamp_vector(start_vector)
        self.history: List[PreferenceRecord] = []
        self.x_plus: Tuple[float, ...] = self.default_start
        self.current_samples: List[PlaneSample] = []
        self.representatives: List[Tuple[float, ...]] = []
        self.optimizer = PreferentialBayesianOptimizer(dimensions)
        self.generate_plane()

    def reset(self, start_vector: Optional[Sequence[float]] = None) -> None:
        """Clear preference history and rebuild the plane around a start vector."""
        self.history = []
        self.optimizer.fit_from_records([])
        self.x_plus = _clamp_vector(start_vector or self.default_start)
        self.generate_plane()

    def observe(self, chosen_index: int) -> None:
        """Record the preferred gallery sample and advance to the next plane."""
        if not 0 <= chosen_index < len(self.current_samples):
            raise IndexError(f"chosen_index out of range: {chosen_index}")

        chosen = self.current_samples[chosen_index].search_vector.vector
        rejected = []
        for idx, sample in enumerate(self.current_samples):
            vector = sample.search_vector.vector
            if idx != chosen_index and _distance(vector, chosen) > EPSILON:
                rejected.append(vector)

        self.history.append(PreferenceRecord(chosen=chosen, rejected=tuple(self._unique_vectors(rejected))))
        self.optimizer.fit_from_records((record.chosen, record.rejected) for record in self.history)
        self.x_plus = chosen
        self.generate_plane()

    def transforms(self) -> List[SearchVector]:
        """Return the current 3x3 gallery samples as normalized search vectors."""
        return [sample.search_vector for sample in self.current_samples]

    def generate_plane(self) -> None:
        """Construct a bounded 3x3 plane centered on the current best vector."""
        center = self.x_plus
        x_ei = self._estimate_x_ei()
        u = tuple(value - center[i] for i, value in enumerate(x_ei))
        if _norm(u) <= EPSILON:
            u = self._random_direction()
        u = self._fit_symmetric_step(center, u, self.plane_radius)

        v = self._best_orthogonal_step(center, u)
        coeffs = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (0, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1),
        ]

        samples: List[PlaneSample] = []
        for u_coeff, v_coeff in coeffs:
            vector = tuple(center[i] + u_coeff * u[i] + v_coeff * v[i] for i in range(self.dimensions))
            samples.append(PlaneSample(SearchVector.from_vector(vector), u_coeff, v_coeff))

        representatives = [
            center,
            tuple(center[i] + u[i] for i in range(self.dimensions)),
            tuple(center[i] - u[i] for i in range(self.dimensions)),
            tuple(center[i] + v[i] for i in range(self.dimensions)),
            tuple(center[i] - v[i] for i in range(self.dimensions)),
        ]

        self.current_samples = samples
        self.representatives = [_clamp_vector(vector) for vector in representatives]

    def _estimate_x_ei(self) -> Tuple[float, ...]:
        """Pick the expected-improvement target for the next plane axis."""
        if not self.optimizer.fitted:
            direction = self._random_direction()
            return _clamp_vector(self.x_plus[i] + direction[i] * self.plane_radius for i in range(self.dimensions))

        candidates = [self.x_plus]
        for _ in range(256):
            candidates.append(tuple(self.rng.random() for _ in range(self.dimensions)))
        best, _score = self.optimizer.best_expected_improvement(candidates, self.x_plus)
        return _clamp_vector(best)

    def _acquisition_score(self, candidate: Sequence[float]) -> float:
        """Score a candidate using GP posterior Expected Improvement."""
        if not self.optimizer.fitted:
            return self.rng.random() * 0.001
        return self.optimizer.expected_improvement([_clamp_vector(candidate)], self.x_plus)[0]

    def _random_direction(self) -> Tuple[float, ...]:
        """Create a random unit direction in the SPS search space."""
        direction = [self.rng.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
        return _normalize(direction)

    def _fit_symmetric_step(self, center: Sequence[float], direction: Sequence[float], target_radius: float) -> Tuple[float, ...]:
        """Scale a direction so center +/- step stays inside the normalized cube."""
        normalized = _normalize(direction)
        if _norm(normalized) <= EPSILON:
            normalized = self._random_direction()

        max_radius = target_radius
        for center_value, direction_value in zip(center, normalized):
            if abs(direction_value) <= EPSILON:
                continue
            max_radius = min(max_radius, min(center_value, 1.0 - center_value) / abs(direction_value))

        radius = max(EPSILON, max_radius * 0.98)
        return tuple(direction_value * radius for direction_value in normalized)

    def _best_orthogonal_step(self, center: Sequence[float], u: Sequence[float]) -> Tuple[float, ...]:
        """Choose the second plane axis from random directions orthogonal to u."""
        u_norm = _normalize(u)
        best_v = None
        best_score = -float("inf")

        for _ in range(48):
            raw = self._random_direction()
            projection = _dot(raw, u_norm)
            orthogonal = tuple(raw[i] - projection * u_norm[i] for i in range(self.dimensions))
            v = self._fit_plane_v_step(center, u, orthogonal, self.plane_radius)
            score = self._plane_score(center, u, v)
            if score > best_score:
                best_v = v
                best_score = score

        if best_v is None:
            return tuple(0.0 for _ in range(self.dimensions))
        return best_v

    def _fit_plane_v_step(
        self,
        center: Sequence[float],
        u: Sequence[float],
        direction: Sequence[float],
        target_radius: float,
    ) -> Tuple[float, ...]:
        """Scale v so every visible 3x3 plane sample remains in bounds."""
        normalized = _normalize(direction)
        if _norm(normalized) <= EPSILON:
            normalized = self._random_direction()

        max_radius = target_radius
        for center_value, u_value, direction_value in zip(center, u, normalized):
            if abs(direction_value) <= EPSILON:
                continue
            remaining_margin = min(center_value - abs(u_value), 1.0 - center_value - abs(u_value))
            max_radius = min(max_radius, max(0.0, remaining_margin) / abs(direction_value))

        radius = max(EPSILON, max_radius * 0.98)
        return tuple(direction_value * radius for direction_value in normalized)

    def _plane_score(self, center: Sequence[float], u: Sequence[float], v: Sequence[float]) -> float:
        """Approximate plane quality by averaging Expected Improvement over visible samples."""
        points = []
        for u_coeff in (-1, 0, 1):
            for v_coeff in (-1, 0, 1):
                point = tuple(center[i] + u_coeff * u[i] + v_coeff * v[i] for i in range(self.dimensions))
                points.append(_clamp_vector(point))
        if not self.optimizer.fitted:
            return self.rng.random() * 0.001
        scores = self.optimizer.expected_improvement(points, self.x_plus)
        return sum(scores) / len(scores)

    def _unique_vectors(self, vectors: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
        """Remove near-duplicate vectors before storing preference data."""
        unique = []
        for vector in vectors:
            clamped = _clamp_vector(vector)
            if all(_distance(clamped, existing) > EPSILON for existing in unique):
                unique.append(clamped)
        return tuple(unique)
