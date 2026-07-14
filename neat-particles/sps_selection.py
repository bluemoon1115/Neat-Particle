from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import neat
from neat.innovation import InnovationTracker

from particle_similarity import (
    FINAL_SETTINGS,
    SELECTION_SETTINGS,
    ParticleBehaviorProfile,
    ParticleSimilaritySettings,
    compare_profiles,
    histogram_distance,
    profile_genome,
)
from sequential_plane_search import SearchVector, SequentialPlaneSearch


WeightSlot = Tuple[int, int]


@dataclass(frozen=True)
class SpsCandidate:
    index: int
    genome: neat.DefaultGenome
    search_vector: SearchVector
    distance: Optional[float] = None


@dataclass
class SpsRunResult:
    target: neat.DefaultGenome
    final_genome: neat.DefaultGenome
    final_distance: float
    final_histogram_distance: float
    final_ssim_distance: float
    steps: int
    stop_reason: str
    history: List[Dict[str, object]]
    elapsed_seconds: float
    initial_design_space: str
    final_design_space: str
    switch_step: Optional[int]
    switch_reason: Optional[str]
    initial_best_distance: Optional[float]


class ParticleGenomeFactory:
    """Create and mutate particle genomes without importing pygame rendering code."""

    def __init__(self, config: neat.Config, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.innovation_tracker = InnovationTracker()
        self.genome_config = self.config.genome_config
        self.genome_config.innovation_tracker = self.innovation_tracker
        self.next_key = 1

    def _new_key(self) -> int:
        key = self.next_key
        self.next_key += 1
        return key

    def create_random_genome(self) -> neat.DefaultGenome:
        self.genome_config.innovation_tracker = self.innovation_tracker
        if self.seed is not None:
            random.seed(self.rng.randrange(2**32))
        genome = neat.DefaultGenome(self._new_key())
        genome.configure_new(self.genome_config)
        return genome


def load_particle_config(config_path: str) -> neat.Config:
    """Load the particle NEAT config used by both UI and headless tools."""
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.abspath(config_path),
    )


def normalize_weight(weight: float, config: neat.Config) -> float:
    """Map a real genome weight into the normalized SPS interval."""
    weight_lo = float(config.genome_config.weight_min_value)
    weight_hi = float(config.genome_config.weight_max_value)
    if weight_hi <= weight_lo:
        return 0.5
    clamped = max(weight_lo, min(weight_hi, weight))
    return (clamped - weight_lo) / (weight_hi - weight_lo)


def decode_weight(value: float, config: neat.Config) -> float:
    """Map one normalized SPS coordinate back into a genome weight."""
    weight_lo = float(config.genome_config.weight_min_value)
    weight_hi = float(config.genome_config.weight_max_value)
    return weight_lo + max(0.0, min(1.0, value)) * (weight_hi - weight_lo)


def generic_weight_slots(
    genome: neat.DefaultGenome,
    config: neat.Config,
    design_space: str = "velocity",
) -> List[WeightSlot]:
    """Select Generic connection weights for one SPS design space."""
    px, py, pz, distance, bias_input = config.genome_config.input_keys
    vx, vy, vz, r, g, b = config.genome_config.output_keys

    distance_to_velocity_slots = [
        (distance, vx), (distance, vy), (distance, vz),
    ]
    distance_to_color_slots = [
        (distance, r), (distance, g), (distance, b),
    ]
    bias_to_velocity_slots = [
        (bias_input, vx), (bias_input, vy), (bias_input, vz),
    ]
    bias_to_color_slots = [
        (bias_input, r), (bias_input, g), (bias_input, b),
    ]
    position_to_velocity_slots = [
        (px, vx), (px, vy), (px, vz),
        (py, vx), (py, vy), (py, vz),
        (pz, vx), (pz, vy), (pz, vz),
    ]

    if design_space == "velocity":
        preferred_slots = distance_to_velocity_slots + bias_to_velocity_slots + position_to_velocity_slots
    elif design_space == "color":
        preferred_slots = distance_to_color_slots + bias_to_color_slots
    else:
        raise ValueError(f"unknown SPS design space: {design_space!r}")

    slots = [
        key
        for key in preferred_slots
        if key in genome.connections and genome.connections[key].enabled
    ]
    return slots or [
        key
        for key, connection in sorted(genome.connections.items())
        if connection.enabled
    ] or sorted(genome.connections)


def genome_to_weight_vector(
    genome: neat.DefaultGenome,
    slots: Sequence[WeightSlot],
    config: neat.Config,
) -> Tuple[float, ...]:
    """Read selected genome weights and normalize them for SPS."""
    return tuple(normalize_weight(genome.connections[key].weight, config) for key in slots)


def apply_weight_vector(
    genome: neat.DefaultGenome,
    slots: Sequence[WeightSlot],
    search_vector: SearchVector,
    config: neat.Config,
) -> neat.DefaultGenome:
    """Create one SPS genome variant by decoding vector values into weights."""
    variant = copy.deepcopy(genome)
    for key, value in zip(slots, search_vector.vector):
        if key in variant.connections:
            variant.connections[key].weight = decode_weight(value, config)
    variant.fitness = None
    return variant


def build_sps_candidates(
    bound_genome: neat.DefaultGenome,
    slots: Sequence[WeightSlot],
    search: SequentialPlaneSearch,
    config: neat.Config,
) -> List[SpsCandidate]:
    """Build the 3x3 SPS candidate genomes from the current search plane."""
    return [
        SpsCandidate(
            index=index,
            genome=apply_weight_vector(bound_genome, slots, search_vector, config),
            search_vector=search_vector,
        )
        for index, search_vector in enumerate(search.transforms())
    ]


def create_sps_search(
    bound_genome: neat.DefaultGenome,
    slots: Sequence[WeightSlot],
    config: neat.Config,
    seed: Optional[int] = None,
) -> SequentialPlaneSearch:
    """Create an SPS search centered on a genome's current selected weights."""
    start_vector = genome_to_weight_vector(bound_genome, slots, config)
    return SequentialPlaneSearch(len(slots), start_vector, seed=seed)


def select_nearest_candidate(
    candidates: Sequence[SpsCandidate],
    target: neat.DefaultGenome,
    config: neat.Config,
    *,
    settings: ParticleSimilaritySettings = SELECTION_SETTINGS,
    target_profile: Optional[ParticleBehaviorProfile] = None,
) -> Tuple[int, float, List[float]]:
    """Return the candidate with the nearest sampled particle behavior."""
    if target_profile is None:
        target_profile = profile_genome(target, config, settings, include_raster=False)

    distances = []
    for candidate in candidates:
        candidate_profile = profile_genome(candidate.genome, config, settings, include_raster=False)
        distances.append(histogram_distance(target_profile.histogram, candidate_profile.histogram))

    best_index = min(range(len(distances)), key=distances.__getitem__)
    return best_index, distances[best_index], distances


def run_auto_sps_selection(
    target: neat.DefaultGenome,
    config: neat.Config,
    *,
    seed: Optional[int],
    threshold: float,
    max_steps: int,
) -> SpsRunResult:
    """Run SPS without rendering until threshold or max steps is reached."""
    started_at = time.perf_counter()
    factory = ParticleGenomeFactory(config, seed=seed)
    bound_genome = factory.create_random_genome()
    initial_design_space = "velocity"
    design_space = initial_design_space
    slots = generic_weight_slots(bound_genome, config, design_space)
    search = create_sps_search(bound_genome, slots, config, seed=seed)
    history: List[Dict[str, object]] = []
    initial_best_distance: Optional[float] = None
    switch_step: Optional[int] = None
    switch_reason: Optional[str] = None

    final_genome = bound_genome
    selection_target_profile = profile_genome(target, config, SELECTION_SETTINGS, include_raster=False)
    # initial_similarity = compare_profiles
    final_histogram_distance = compare_profiles(
        selection_target_profile,
        profile_genome(final_genome, config, SELECTION_SETTINGS),
    ).histogram_distance
    final_distance = final_histogram_distance
    final_ssim_distance = 0.0
    # final_histogram_distance = initial_similarity.histogram_distance
    # final_ssim_distance = initial_similarity.ssim_distance
    # final_distance = final_ssim_distance
    stop_reason = "threshold"

    for step in range(max_steps):
        candidates = build_sps_candidates(bound_genome, slots, search, config)
        best_index, best_distance, distances = select_nearest_candidate(
            candidates,
            target,
            config,
            settings=SELECTION_SETTINGS,
            target_profile=selection_target_profile,
        )
        chosen = candidates[best_index]
        step_number = step + 1
        if initial_best_distance is None:
            initial_best_distance = best_distance

        final_genome = chosen.genome
        final_histogram_distance = best_distance
        # final_histogram_distance = 0.0
        # final_ssim_distance = best_distance
        final_distance = best_distance
        history_record: Dict[str, object] = {
            "step": step_number,
            "design_space": design_space,
            "chosen_index": best_index,
            "best_distance": best_distance,
            "distances": distances,
            "center_index": 4,
        }
        history.append(history_record)

        bound_genome = copy.deepcopy(chosen.genome)
        if best_distance <= threshold:
            stop_reason = "threshold"
            break

        if design_space == "velocity":
            steps_remaining = max_steps - step_number
            if step_number > 1 and best_distance <= 0.5 * initial_best_distance:
                switch_reason = "half_initial_distance"
            elif steps_remaining <= max_steps / 2:
                switch_reason = "half_max_steps_remaining"

            if switch_reason is not None:
                design_space = "color"
                switch_step = step_number
                history_record["switch_to_design_space"] = design_space
                history_record["switch_reason"] = switch_reason
                slots = generic_weight_slots(bound_genome, config, design_space)
                search = create_sps_search(bound_genome, slots, config, seed=seed)
                continue

        search.observe(best_index)
    else:
        stop_reason = "max_steps"

    final_target_profile = profile_genome(target, config, FINAL_SETTINGS)
    final_candidate_profile = profile_genome(final_genome, config, FINAL_SETTINGS)
    final_similarity = compare_profiles(final_target_profile, final_candidate_profile)
    final_histogram_distance = final_similarity.histogram_distance
    final_ssim_distance = final_similarity.ssim_distance
    final_distance = final_similarity.combined_distance

    elapsed_seconds = time.perf_counter() - started_at
    return SpsRunResult(
        target=target,
        final_genome=final_genome,
        final_distance=final_distance,
        final_histogram_distance=final_histogram_distance,
        final_ssim_distance=final_ssim_distance,
        steps=len(history),
        stop_reason=stop_reason,
        history=history,
        elapsed_seconds=elapsed_seconds,
        initial_design_space=initial_design_space,
        final_design_space=design_space,
        switch_step=switch_step,
        switch_reason=switch_reason,
        initial_best_distance=initial_best_distance,
    )
