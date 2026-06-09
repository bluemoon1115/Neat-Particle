from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# to prevent divided by zero problem
EPSILON = 1e-6


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _clamp_vector(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(_clamp(v, EPSILON, 1.0 - EPSILON) for v in values)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(values: Sequence[float]) -> Tuple[float, ...]:
    n = _norm(values)
    if n <= EPSILON:
        return tuple(0.0 for _ in values)
    return tuple(v / n for v in values)


DIMENSIONS = 9
NON_BIAS_INPUTS = 4


@dataclass(frozen=True)
class InputTransform:
    """Normalized SPS vector for Generic ANN input transforms."""
    vector: Tuple[float, ...]

    def __post_init__(self) -> None:
        # sets the vector in bound right after initializing by clamp function
        if len(self.vector) != DIMENSIONS:
            raise ValueError(f"InputTransform needs {DIMENSIONS} values, got {len(self.vector)}")
        object.__setattr__(self, "vector", _clamp_vector(self.vector))

    @classmethod
    def identity(cls) -> "InputTransform":
        """Return scale=1, offset=0, bias=1 in normalized coordinates."""
        scale = (1.0 - 0.1) / (3.0 - 0.1)
        return cls((scale, scale, scale, scale, 0.5, 0.5, 0.5, 0.5, 1.0))

    # use classmethod and cls so the function can still be called without instance being created
    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "InputTransform":
        return cls(tuple(vector))

    def decoded(self) -> Tuple[Tuple[float, ...], Tuple[float, ...], float]:
        """Decode normalized values into scales, offsets, and bias value."""
        scales = tuple(0.1 + value * 2.9 for value in self.vector[:NON_BIAS_INPUTS])
        offsets = tuple(-1.0 + value * 2.0 for value in self.vector[NON_BIAS_INPUTS:NON_BIAS_INPUTS * 2])
        bias = -1.0 + self.vector[-1] * 2.0
        return scales, offsets, bias

    def apply(self, inputs: Sequence[float]) -> Tuple[float, ...]:
        """Apply scale/offset preprocessing before the ANN sees particle inputs."""
        if len(inputs) != 5:
            raise ValueError(f"InputTransform expects 5 ANN inputs, got {len(inputs)}")
        scales, offsets, bias = self.decoded()
        transformed = [value * scale + offset for value, scale, offset in zip(inputs[:4], scales, offsets)]
        transformed.append(bias)
        return tuple(transformed)

    def short_label(self) -> str:
        """Return a compact label for the SPS gallery."""
        scales, offsets, bias = self.decoded()
        avg_scale = sum(scales) / len(scales)
        avg_offset = sum(offsets) / len(offsets)
        return f"s={avg_scale:.2f} o={avg_offset:+.2f} b={bias:+.2f}"


@dataclass(frozen=True)
class PreferenceRecord:
    chosen: Tuple[float, ...]
    rejected: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class PlaneSample:
    transform: InputTransform
    u_coeff: int
    v_coeff: int


class SequentialPlaneSearch:
    """
    Preference-guided sequential plane search for a normalized design vector.
    
    Think of this as a "human-in-the-loop" tuner.
        Explore: Generates a 2D plane of 9 candidate vectors (3 X 3 grid) centered around the best-known solution.
        Observe: The user picks their favorite sample from the grid.
        Learn: The algorithm records the chosen vector (positive) and all others (negatives), 
            updates its internal model, and creates a new plane centered on the choice.
    
    """

    def __init__(self, start_vector: Optional[Sequence[float]] = None,
                 seed: Optional[int] = None, plane_radius: float = 0.22):
        self.rng = random.Random(seed)
        self.plane_radius = plane_radius
        self.dimensions = DIMENSIONS
        self.default_start = self._default_start(start_vector or InputTransform.identity().vector)
        self.history: List[PreferenceRecord] = []
        self.x_plus: Tuple[float, ...] = self.default_start
        self.current_samples: List[PlaneSample] = []
        self.representatives: List[Tuple[float, ...]] = []
        self.generate_plane()

    def reset(self, start: Optional[object] = None) -> None:
        """Clear preference history and rebuild the plane around a start vector."""
        self.history = []
        if start is None:
            self.x_plus = self.default_start
        elif hasattr(start, "vector"):
            self.x_plus = _clamp_vector(start.vector)
        else:
            self.x_plus = _clamp_vector(start)
        self.generate_plane()

    def observe(self, chosen_index: int) -> None:
        """Record the preferred gallery sample and advance to the next plane."""
        if not 0 <= chosen_index < len(self.current_samples):
            raise IndexError(f"chosen_index out of range: {chosen_index}")

        chosen = self.current_samples[chosen_index].transform.vector
        rejected = []
        for vector in self.representatives:
            if _distance(vector, chosen) > EPSILON:
                rejected.append(vector)
        for idx, sample in enumerate(self.current_samples):
            if idx != chosen_index and _distance(sample.transform.vector, chosen) > EPSILON:
                rejected.append(sample.transform.vector)

        self.history.append(PreferenceRecord(chosen=chosen, rejected=tuple(self._unique_vectors(rejected))))
        self.x_plus = chosen
        self.generate_plane()

    def transforms(self) -> List[InputTransform]:
        """Return the current 3x3 gallery samples."""
        return [sample.transform for sample in self.current_samples]

    def generate_plane(self) -> None:
        """
        Construct a bounded 3x3 plane centered on the current best vector.
        
        To move from one state to the next, it creates a 2D subspace defined by two orthogonal vectors, u and v.
        u: looks for a direction that maximizes the acquisition score (Expected Improvement).
        v: It finds a vector that is mathematically orthogonal to u 
            using the projection formula: v_{orthogonal} = raw - proj_u raw. 
        This ensures the search grid spreads out in a new, independent direction.
        """
        center = self.x_plus
        x_ei = self._estimate_x_ei()
        u = tuple(x - c for x, c in zip(x_ei, center))
        if _norm(u) <= EPSILON:
            u = self._random_direction()
        u = self._fit_symmetric_step(center, u, self.plane_radius)

        v = self._best_orthogonal_step(center, u)
        coeffs = [(-1, -1), (0, -1), (1, -1),
                  (-1, 0), (0, 0), (1, 0),
                  (-1, 1), (0, 1), (1, 1)]

        samples: List[PlaneSample] = []
        for u_coeff, v_coeff in coeffs:
            vector = tuple(center[i] + u_coeff * u[i] + v_coeff * v[i] for i in range(self.dimensions))
            samples.append(PlaneSample(InputTransform.from_vector(vector), u_coeff, v_coeff))

        representatives = [
            center,
            tuple(center[i] + u[i] for i in range(self.dimensions)),
            tuple(center[i] - u[i] for i in range(self.dimensions)),
            tuple(center[i] + v[i] for i in range(self.dimensions)),
            tuple(center[i] - v[i] for i in range(self.dimensions)),
        ]

        self.current_samples = samples
        self.representatives = [_clamp_vector(v) for v in representatives]

    def _estimate_x_ei(self) -> Tuple[float, ...]:
        """Pick a lightweight expected-improvement target for the next plane axis."""
        if not self.history:
            direction = self._random_direction()
            return _clamp_vector(self.x_plus[i] + direction[i] * self.plane_radius for i in range(self.dimensions))

        best = self.x_plus
        best_score = self._acquisition_score(best)
        for _ in range(160):
            candidate = tuple(self.rng.random() for _ in range(self.dimensions))
            score = self._acquisition_score(candidate)
            if score > best_score:
                best = candidate
                best_score = score
        return _clamp_vector(best)

    def _acquisition_score(self, candidate: Sequence[float]) -> float:
        """
            Score a candidate using preference reward, rejection penalty, and novelty.

            The system uses a score to guide the search. It uses three factors:

            Reward: Uses a Gaussian kernel to measure proximity to past "chosen" vectors.

            Penalty: Measures proximity to "rejected" vectors to avoid areas the user disliked.

            Uncertainty (Novelty): Encourages the search to look at unexplored areas (far from existing points), 
                which prevents the system from getting stuck in a local minimum.    
        """
        positives = [record.chosen for record in self.history]
        negatives = [v for record in self.history for v in record.rejected]
        width = 0.28

        reward = sum(math.exp(-(_distance(candidate, p) ** 2) / (2.0 * width * width)) for p in positives)
        penalty = sum(math.exp(-(_distance(candidate, n) ** 2) / (2.0 * width * width)) for n in negatives)
        observed = positives + negatives
        uncertainty = 1.0
        if observed:
            uncertainty = min(_distance(candidate, point) for point in observed) / math.sqrt(self.dimensions)

        # return the final acquisition score
        return reward - 0.65 * penalty + 0.35 * uncertainty + self.rng.random() * 0.001

    def _random_direction(self) -> Tuple[float, ...]:
        direction = [self.rng.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
        return _normalize(direction)

    def _fit_symmetric_step(self, center: Sequence[float], direction: Sequence[float], target_radius: float) -> Tuple[float, ...]:
        """
        Scale a direction so center +/- step stays inside the normalized cube.
        
        Because the vectors are "normalized" (usually constrained within a unit cube [0, 1]), 
        the algorithm cannot simply move in any direction. 
        If it moves too far, it hits the boundary of the space.
        
        These functions calculate the maximum allowable step size before hitting a boundary. 
        They effectively "shrink" the plane size dynamically to ensure that all 9 points in your 3X3 grid stay within valid bounds
        """
        normalized = _normalize(direction)
        if _norm(normalized) <= EPSILON:
            normalized = self._random_direction()

        max_radius = target_radius
        for c, d in zip(center, normalized):
            if abs(d) <= EPSILON:
                continue
            max_radius = min(max_radius, min(c, 1.0 - c) / abs(d))

        radius = max(EPSILON, max_radius * 0.98)
        return tuple(d * radius for d in normalized)

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

    def _fit_plane_v_step(self, center: Sequence[float], u: Sequence[float],
                          direction: Sequence[float], target_radius: float) -> Tuple[float, ...]:
        """Scale v so every visible 3x3 plane sample remains in bounds."""
        normalized = _normalize(direction)
        if _norm(normalized) <= EPSILON:
            normalized = self._random_direction()

        max_radius = target_radius
        for c, u_value, d in zip(center, u, normalized):
            if abs(d) <= EPSILON:
                continue
            remaining_margin = min(c - abs(u_value), 1.0 - c - abs(u_value))
            max_radius = min(max_radius, max(0.0, remaining_margin) / abs(d))

        radius = max(EPSILON, max_radius * 0.98)
        return tuple(d * radius for d in normalized)

    def _plane_score(self, center: Sequence[float], u: Sequence[float], v: Sequence[float]) -> float:
        """Approximate plane quality by averaging acquisition scores over 3x3 samples."""
        total = 0.0
        count = 0
        for u_coeff in (-1, 0, 1):
            for v_coeff in (-1, 0, 1):
                point = tuple(center[i] + u_coeff * u[i] + v_coeff * v[i] for i in range(self.dimensions))
                total += self._acquisition_score(_clamp_vector(point))
                count += 1
        return total / count

    def _unique_vectors(self, vectors: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
        """Remove near-duplicate vectors before storing preference data."""
        unique = []
        for vector in vectors:
            clamped = _clamp_vector(vector)
            if all(_distance(clamped, existing) > EPSILON for existing in unique):
                unique.append(clamped)
        return tuple(unique)

    def _default_start(self, start_vector: Sequence[float]) -> Tuple[float, ...]:
        """Choose the initial normalized vector for this SPS target."""
        if len(start_vector) != self.dimensions:
            raise ValueError(f"start_vector needs {self.dimensions} values, got {len(start_vector)}")
        return _clamp_vector(start_vector)
