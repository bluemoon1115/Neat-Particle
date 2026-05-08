"""Commonly used statistical helper functions.

These originally filled gaps in the Python 2 standard library and are retained
for API compatibility. Implementations are intentionally simple and avoid
relying on newer stdlib helpers where semantics might differ.
"""

from __future__ import annotations

from math import sqrt, exp, nan
from typing import Callable, Dict, Iterable, List, Sequence


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of *values* as a float.

    Returns ``nan`` for empty input rather than raising ``ZeroDivisionError``,
    so per-generation aggregations over a transiently empty population
    (e.g. mid-extinction) yield a sentinel rather than crashing the run.
    """

    vals: List[float] = [float(v) for v in values]
    if not vals:
        return nan
    return sum(vals) / len(vals)


def median(values: Iterable[float]) -> float:
    """Return the middle value of *values*.

    For even-length inputs this returns the upper-middle element,
    matching the historical behaviour used throughout the project.
    """

    vals = list(values)
    vals.sort()
    return vals[len(vals) // 2]


def median2(values: Iterable[float]) -> float:
    """Median that averages the two middle values for even-length inputs."""

    vals = list(values)
    n = len(vals)
    if n <= 2:
        return mean(vals)

    vals.sort()
    if (n % 2) == 1:
        return vals[n // 2]

    i = n // 2
    return (vals[i - 1] + vals[i]) / 2.0


def variance(values: Iterable[float]) -> float:
    """Population variance of *values*.

    Returns ``nan`` for empty input (matches :func:`mean`).
    """

    vals = list(values)
    if not vals:
        return nan
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def stdev(values: Iterable[float]) -> float:
    """Population standard deviation of *values*.

    This is simply ``sqrt(variance(values))``.
    """

    return sqrt(variance(values))


def softmax(values: Iterable[float]) -> List[float]:
    """Compute the softmax of the given value set.

    For each input ``v_i`` this returns ``exp(v_i) / s`` where
    ``s = sum(exp(v_j) for v_j in values)``. Returns ``[]`` for empty input.
    """

    e_values: List[float] = [exp(v) for v in values]
    if not e_values:
        return []
    s = sum(e_values)
    if s == 0.0:
        # All inputs underflowed exp() to 0; fall back to a uniform distribution
        # so callers that interpret the result as probabilities stay well-defined.
        return [1.0 / len(e_values)] * len(e_values)
    inv_s = 1.0 / s
    return [ev * inv_s for ev in e_values]


# Lookup table for commonly used {value} -> value functions.
stat_functions: Dict[str, Callable[[Sequence[float]], float]] = {
    'min': min,
    'max': max,
    'mean': mean,
    'median': median,
    'median2': median2,
}
