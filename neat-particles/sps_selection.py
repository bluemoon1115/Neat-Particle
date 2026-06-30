from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import neat
from neat.innovation import InnovationTracker

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
    steps: int
    stop_reason: str
    history: List[Dict[str, object]]
    elapsed_seconds: float


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


def generic_weight_slots(genome: neat.DefaultGenome, config: neat.Config) -> List[WeightSlot]:
    """Select Generic RGB-input connection weights for SPS."""
    input_keys = config.genome_config.input_keys
    preferred_inputs = {input_keys[0], input_keys[1], input_keys[2]}
    slots = [
        key
        for key, connection in sorted(genome.connections.items())
        if connection.enabled and key[0] in preferred_inputs
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
    genome_config,
) -> Tuple[int, float, List[float]]:
    """Return the index and distance of the candidate nearest to the target."""
    distances = [candidate.genome.distance(target, genome_config) for candidate in candidates]
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
    slots = generic_weight_slots(bound_genome, config)
    search = create_sps_search(bound_genome, slots, config, seed=seed)
    history: List[Dict[str, object]] = []

    final_genome = bound_genome
    final_distance = final_genome.distance(target, config.genome_config)
    stop_reason = "threshold"

    for step in range(max_steps):
        candidates = build_sps_candidates(bound_genome, slots, search, config)
        best_index, best_distance, distances = select_nearest_candidate(
            candidates,
            target,
            config.genome_config,
        )
        chosen = candidates[best_index]
        final_genome = chosen.genome
        final_distance = best_distance
        history.append(
            {
                "step": step + 1,
                "chosen_index": best_index,
                "best_distance": best_distance,
                "distances": distances,
                "center_index": 4,
            }
        )

        bound_genome = copy.deepcopy(chosen.genome)
        if best_distance <= threshold:
            stop_reason = "threshold"
            break
        search.observe(best_index)
    else:
        stop_reason = "max_steps"

    elapsed_seconds = time.perf_counter() - started_at
    return SpsRunResult(
        target=target,
        final_genome=final_genome,
        final_distance=final_distance,
        steps=len(history),
        stop_reason=stop_reason,
        history=history,
        elapsed_seconds=elapsed_seconds,
    )
