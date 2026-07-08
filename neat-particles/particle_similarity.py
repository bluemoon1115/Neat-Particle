from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import neat

from particle_systems import make_system


Histogram = Tuple[float, ...]
Raster = Tuple[float, ...]

# frozen makes the class immutible after being initiated
@dataclass(frozen=True)
class ParticleSimilaritySettings:
    """Sampling settings for deterministic particle behavior comparison."""

    dt: float = 1.0 / 60.0
    scale: float = 0.45
    simulation_steps: int = 120
    sample_stride: int = 5
    grid_size: int = 24
    seed: int = 101
    color_weight: float = 0.3


@dataclass(frozen=True)
class ParticleBehaviorProfile:
    """Compact sampled representation of one particle system run."""

    histogram: Histogram
    raster: Raster


@dataclass(frozen=True)
class ParticleSimilarityResult:
    """Behavioral distance components for two particle systems."""

    histogram_distance: float
    ssim_distance: float
    combined_distance: float


SELECTION_SETTINGS = ParticleSimilaritySettings(sample_stride=5)
FINAL_SETTINGS = ParticleSimilaritySettings(sample_stride=5)


def profile_genome(
    genome: neat.DefaultGenome,
    config: neat.Config,
    settings: ParticleSimilaritySettings,
    *,
    include_raster: bool = True,
) -> ParticleBehaviorProfile:
    """Simulate a genome-controlled particle system and return sampled features."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    system = make_system(seed=settings.seed)
    bin_count = settings.grid_size * settings.grid_size
    occupancy = [0.0 for _ in range(bin_count)]
    red = [0.0 for _ in range(bin_count)]
    green = [0.0 for _ in range(bin_count)]
    blue = [0.0 for _ in range(bin_count)]
    raster = [0.0 for _ in range(settings.grid_size * settings.grid_size)]
    sample_count = 0

    for step in range(1, settings.simulation_steps + 1):
        system.update(net, dt=settings.dt, scale=settings.scale)
        if step % settings.sample_stride == 0:
            sample_count += 1
            _sample_particles(
                system.particles,
                occupancy,
                red,
                green,
                blue,
                raster,
                settings.grid_size,
                include_raster,
            )

    histogram = _build_color_spatial_histogram(occupancy, red, green, blue, settings.color_weight)

    raster_scale = max(1, sample_count * len(system.particles))
    raster = [value / raster_scale for value in raster]
    return ParticleBehaviorProfile(tuple(histogram), tuple(raster))


def histogram_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return normalized L1 histogram distance in [0, 1]."""
    if len(left) != len(right):
        raise ValueError("histograms must have the same length")
    return sum(abs(a - b) for a, b in zip(left, right)) * 0.5


def ssim_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a distance derived from global SSIM, where lower is better."""
    if len(left) != len(right):
        raise ValueError("rasters must have the same length")
    if not left:
        return 0.0

    n = len(left)
    mean_left = sum(left) / n
    mean_right = sum(right) / n
    var_left = sum((value - mean_left) ** 2 for value in left) / n
    var_right = sum((value - mean_right) ** 2 for value in right) / n
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right)) / n

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mean_left * mean_right + c1) * (2.0 * covariance + c2)
    denominator = (mean_left ** 2 + mean_right ** 2 + c1) * (var_left + var_right + c2)
    if denominator == 0.0:
        return 0.0

    similarity = numerator / denominator
    return _clamp01((1.0 - similarity) * 0.5)


def compare_profiles(
    target: ParticleBehaviorProfile,
    candidate: ParticleBehaviorProfile,
    *,
    histogram_weight: float = 0.75,
) -> ParticleSimilarityResult:
    """Compare sampled behavior profiles and return weighted distance components."""
    # Spatial histogram scoring is temporarily disabled while evaluating SSIM-only behavior.
    hist = histogram_distance(target.histogram, candidate.histogram)
    # hist = 0.0

    ssim = ssim_distance(target.raster, candidate.raster)
    hist_weight = _clamp01(histogram_weight)
    combined = hist_weight * hist + (1.0 - hist_weight) * ssim
    # combined = ssim

    return ParticleSimilarityResult(hist, ssim, _clamp01(combined))


def compare_genomes(
    target: neat.DefaultGenome,
    candidate: neat.DefaultGenome,
    config: neat.Config,
    settings: ParticleSimilaritySettings,
    *,
    histogram_weight: float = 0.75,
) -> ParticleSimilarityResult:
    """Simulate and compare two genomes under identical particle initial conditions."""
    target_profile = profile_genome(target, config, settings)
    candidate_profile = profile_genome(candidate, config, settings)
    return compare_profiles(target_profile, candidate_profile, histogram_weight=histogram_weight)


def _sample_particles(
    particles,
    occupancy: List[float],
    red: List[float],
    green: List[float],
    blue: List[float],
    raster: List[float],
    grid_size: int,
    include_raster: bool,
) -> None:
    for particle in particles:
        index = _grid_index(particle.x, particle.y, grid_size)
        particle_red = _clamp01(particle.r)
        particle_green = _clamp01(particle.g)
        particle_blue = _clamp01(particle.b)
        occupancy[index] += 1.0
        red[index] += particle_red
        green[index] += particle_green
        blue[index] += particle_blue
        if include_raster:
            raster[index] += _luminance(particle_red, particle_green, particle_blue)


def _build_color_spatial_histogram(
    occupancy: Sequence[float],
    red: Sequence[float],
    green: Sequence[float],
    blue: Sequence[float],
    color_weight: float,
) -> List[float]:
    total = sum(occupancy)
    if total <= 0.0:
        return [0.0 for _ in range(len(occupancy) * 4)]

    color_weight = _clamp01(color_weight)
    spatial_weight = 1.0 - color_weight
    return (
        [spatial_weight * value / total for value in occupancy]
        + [color_weight * value / total for value in red]
        + [color_weight * value / total for value in green]
        + [color_weight * value / total for value in blue]
    )


def _grid_index(x: float, y: float, grid_size: int) -> int:
    gx = min(grid_size - 1, max(0, int(((x + 1.0) * 0.5) * grid_size)))
    gy = min(grid_size - 1, max(0, int(((y + 1.0) * 0.5) * grid_size)))
    return gy * grid_size + gx


def _luminance(red: float, green: float, blue: float) -> float:
    red = _clamp01(red)
    green = _clamp01(green)
    blue = _clamp01(blue)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
