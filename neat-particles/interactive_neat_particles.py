from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import neat
from neat.innovation import InnovationTracker

from draw_genome import draw_genome
from genome_targets import load_genome_target, save_genome_target, timestamped_target_path
from particle_systems import BaseSystem, make_system
from sequential_plane_search import SearchVector, SequentialPlaneSearch


@dataclass
class Candidate:
    index: int
    species_key: int
    genome: neat.DefaultGenome
    net: object
    system: BaseSystem
    label: Optional[str] = None
    selected: bool = False


class InteractiveBreeder:
    """
    Minimal IEC-style breeding loop:
    - keep 9 candidates
    - user selects one or more parents
    - spawn 9 offspring by cloning + mutation (NEAT complexification is in genome.mutate)
    """

    def __init__(self, config: neat.Config, seed: Optional[int] = None):
        self.config = config
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
        genome = neat.DefaultGenome(self._new_key())
        genome.configure_new(self.genome_config)
        return genome

    def clone_and_mutate(self, parent: neat.DefaultGenome) -> neat.DefaultGenome:
        self.genome_config.innovation_tracker = self.innovation_tracker
        child = copy.deepcopy(parent)
        child.key = self._new_key()
        child.fitness = None
        child.mutate(self.genome_config)
        return child

    def randomize_weights(self, genome: neat.DefaultGenome) -> None:
        # Refresh weights and node biases while preserving topology.
        for connection in genome.connections.values():
            connection.weight = self.rng.uniform(-1.0, 1.0)
        for node in genome.nodes.values():
            node.bias = self.rng.uniform(-1.0, 1.0)

    def spawn_generation(self, parents: List[neat.DefaultGenome], n: int = 9, keep_elites: bool = True) -> List[neat.DefaultGenome]:
        # Reset generation-specific innovation dedup tracking.
        self.innovation_tracker.reset_generation()

        if not parents:
            return [self.create_random_genome() for _ in range(n)]

        out: List[neat.DefaultGenome] = []
        if keep_elites:
            for parent in parents[: min(len(parents), n)]:
                elite = copy.deepcopy(parent)
                elite.key = self._new_key()
                elite.fitness = None
                out.append(elite)

        while len(out) < n:
            parent = self.rng.choice(parents)
            out.append(self.clone_and_mutate(parent))

        return out[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.45, help="S in P_t = P_{t-1} + S*V*T")
    parser.add_argument("--target", default=None, help="Optional exported target genome JSON file to show grid distances.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "config-generic.ini")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg_path,
    )
    target_genome: Optional[neat.DefaultGenome] = None
    target_path: Optional[str] = None
    if args.target:
        target_path = os.path.abspath(args.target)
        target_genome, _target_data = load_genome_target(target_path, config)

    breeder = InteractiveBreeder(config, seed=args.seed)
    weight_lo = float(config.genome_config.weight_min_value)
    weight_hi = float(config.genome_config.weight_max_value)

    try:
        import pygame
    except Exception as exc:  # pragma: no cover
        print("pygame is required for this example. Install it with: pip install pygame")
        print(f"Import error: {exc}")
        return 2

    pygame.init()
    pygame.display.set_caption("NEAT Particles (Connection weight SPS)")
    screen = pygame.display.set_mode((1300, 720), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    small = pygame.font.SysFont("consolas", 12)

    def normalize_weight(weight: float) -> float:
        """Map a real genome weight into the normalized SPS interval."""
        if weight_hi <= weight_lo:
            return 0.5
        clamped = max(weight_lo, min(weight_hi, weight))
        return (clamped - weight_lo) / (weight_hi - weight_lo)

    def decode_weight(value: float) -> float:
        """Map one normalized SPS coordinate back into a genome weight."""
        return weight_lo + max(0.0, min(1.0, value)) * (weight_hi - weight_lo)

    def generic_weight_slots(genome: neat.DefaultGenome) -> List[Tuple[int, int]]:
        """Select Generic Dc-input and Bias-input connection weights for SPS."""
        input_keys = config.genome_config.input_keys
        px_key = input_keys[0]
        py_key = input_keys[1]
        pz_key = input_keys[2]
        distance_key = input_keys[3]
        bias_input_key = input_keys[4]
        preferred_inputs = {px_key, py_key, pz_key, distance_key}

        slots = [key for key, connection in sorted(genome.connections.items()) if connection.enabled and key[0] in preferred_inputs]
        return slots

        # Fallback prevents SPS from breaking if mutations deleted/disabled preferred links.
        return [key for key, connection in sorted(genome.connections.items()) if connection.enabled] or sorted(genome.connections)

    def genome_to_weight_vector(genome: neat.DefaultGenome, slots: Sequence[Tuple[int, int]]) -> Tuple[float, ...]:
        """Read selected genome weights and normalize them for SPS."""
        return tuple(normalize_weight(genome.connections[key].weight) for key in slots)

    def apply_weight_vector(
        genome: neat.DefaultGenome,
        slots: Sequence[Tuple[int, int]],
        search_vector: SearchVector,
    ) -> neat.DefaultGenome:
        """Create one SPS genome variant by decoding vector values into weights."""
        variant = copy.deepcopy(genome)
        for key, value in zip(slots, search_vector.vector):
            if key in variant.connections:
                variant.connections[key].weight = decode_weight(value)
        variant.fitness = None
        return variant

    def weight_summary(genome: neat.DefaultGenome, slots: Sequence[Tuple[int, int]]) -> str:
        """Return a compact label showing the tuned Dc/Bias weight averages."""
        input_keys = config.genome_config.input_keys
        distance_key = input_keys[3]
        bias_input_key = input_keys[4]
        distance_weights = [genome.connections[key].weight for key in slots if key[0] == distance_key]
        bias_weights = [genome.connections[key].weight for key in slots if key[0] == bias_input_key]

        def avg(values: Sequence[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return f"Dc={avg(distance_weights):+.2f} Bias={avg(bias_weights):+.2f}"

    def make_candidate(
        idx: int,
        genome,
        system_seed: Optional[int] = None,
        label: Optional[str] = None,
        species_key: Optional[int] = None,
    ) -> Candidate:
        """Build the renderable particle candidate for one genome."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        seed = system_seed if system_seed is not None else (args.seed or 0) + idx * 1337
        system = make_system(seed=seed)
        return Candidate(index=idx, species_key=species_key or idx, genome=genome, net=net, system=system, label=label)

    def build_batch(genomes, key_start: int = 1) -> List[Candidate]:
        """Build the normal IEC gallery from a list of genomes."""
        return [
            make_candidate(i, genome, species_key=key_start + i - 1)
            for i, genome in enumerate(genomes, start=1)
        ]

    candidates = build_batch([breeder.create_random_genome() for _ in range(9)])
    mode = "IEC"
    sps_search: Optional[SequentialPlaneSearch] = None
    sps_bound_genome: Optional[neat.DefaultGenome] = None
    sps_bound_species_key = 1
    sps_weight_slots: List[Tuple[int, int]] = []
    sps_candidates: List[Candidate] = []
    paused = False
    generation = 0
    last_export_path: Optional[str] = None

    margin = 12
    bottom_h = 70

    def cell_rect(i: int):
        """Return the current rectangle for one 3x3 gallery cell."""
        usable_w = screen.get_width() - margin * 4
        usable_h = screen.get_height() - margin * 4 - bottom_h
        cell_w = usable_w // 3
        cell_h = usable_h // 3
        row = i // 3
        col = i % 3
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        return (x, y, cell_w, cell_h)

    def active_candidates() -> List[Candidate]:
        """Return the gallery candidates for the current interaction mode."""
        return sps_candidates if mode == "SPS" else candidates

    def target_distance_label(genome: neat.DefaultGenome) -> str:
        """Return a compact distance label for the optional target genome."""
        if target_genome is None:
            return ""
        distance = genome.distance(target_genome, config.genome_config)
        return f" dist={distance:.4f}"

    def candidate_index_at_pos(pos) -> int:
        """Convert a mouse position into a gallery index."""
        mx, my = pos
        for i, _candidate in enumerate(active_candidates()):
            x, y, w, h = cell_rect(i)
            if x <= mx <= x + w and y <= my <= y + h:
                return i
        return -1

    def build_sps_batch(bound_genome: neat.DefaultGenome, seed: Optional[int]) -> List[Candidate]:
        """Build the 3x3 SPS gallery from weight-tuned genome variants."""
        shared_seed = (seed or 0) + 67
        batch = []
        for i, search_vector in enumerate(sps_search.transforms()):
            variant = apply_weight_vector(bound_genome, sps_weight_slots, search_vector)
            # detail information of the SPS species
            label = f"base={weight_summary(variant, sps_weight_slots)}"
            batch.append(make_candidate(
                i + 1,
                variant,
                system_seed=shared_seed,
                label=label,
                species_key=sps_bound_species_key,
            ))
        return batch

    def bind_sps_to_selection() -> None:
        """Bind SPS to the selected IEC genome, or candidate 1 if none is selected."""
        nonlocal mode, sps_bound_genome, sps_bound_species_key, sps_weight_slots, sps_search, sps_candidates
        selected_indices = [i for i, candidate in enumerate(candidates) if candidate.selected]
        source = candidates[selected_indices[0] if selected_indices else 0]
        sps_bound_genome = copy.deepcopy(source.genome)
        sps_bound_species_key = source.species_key
        sps_weight_slots = generic_weight_slots(sps_bound_genome)
        start_vector = genome_to_weight_vector(sps_bound_genome, sps_weight_slots)
        sps_search = SequentialPlaneSearch(len(sps_weight_slots), start_vector, seed=args.seed)
        sps_candidates = build_sps_batch(sps_bound_genome, args.seed)
        mode = "SPS"

    def reset_current_mode() -> None:
        """Reset only the active interaction mode."""
        nonlocal candidates, sps_candidates, generation
        if mode == "IEC":
            generation = 0
            candidates = build_batch([breeder.create_random_genome() for _ in range(9)])
            return

        if sps_bound_genome is None:
            bind_sps_to_selection()
            return

        start_vector = genome_to_weight_vector(sps_bound_genome, sps_weight_slots)
        sps_search.reset(start_vector)
        sps_candidates = build_sps_batch(sps_bound_genome, args.seed)

    def choose_sps_sample(idx: int) -> None:
        """Accept one SPS sample as preferred and generate the next search plane."""
        nonlocal sps_bound_genome, sps_candidates
        if sps_bound_genome is None:
            bind_sps_to_selection()
        sps_search.observe(idx)
        sps_bound_genome = copy.deepcopy(sps_candidates[idx].genome)
        sps_candidates = build_sps_batch(sps_bound_genome, args.seed)

    def return_sps_to_iec() -> None:
        """Replace the earliest selected IEC candidate with the current SPS genome."""
        nonlocal candidates, mode
        if sps_bound_genome is None:
            mode = "IEC"
            return

        selected_indices = [i for i, candidate in enumerate(candidates) if candidate.selected]
        target_idx = selected_indices[0] if selected_indices else 0
        target = candidates[target_idx]
        replacement = make_candidate(
            target.index,
            copy.deepcopy(sps_bound_genome),
            label="SPS return",
            species_key=target.species_key,
        )
        replacement.selected = True
        candidates[target_idx] = replacement
        mode = "IEC"

    def export_current_target() -> None:
        """Export the current manual target genome for headless SPS auto-selection."""
        nonlocal last_export_path
        target_dir = os.path.join(base_dir, "targets")
        if mode == "SPS":
            if not sps_candidates:
                bind_sps_to_selection()
            source = sps_candidates[4]
        else:
            selected = [candidate for candidate in candidates if candidate.selected]
            source = selected[0] if selected else candidates[0]

        path = timestamped_target_path(target_dir, source.species_key)
        last_export_path = save_genome_target(
            path,
            source.genome,
            config,
            candidate_key=source.species_key,
            mode=mode,
            generation=generation,
            label=source.label,
        )
        print(f"Exported target genome: {last_export_path}")

    running = True
    while running:
        dt = clock.tick(args.fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_TAB:
                    if mode == "IEC":
                        bind_sps_to_selection()
                    else:
                        return_sps_to_iec()
                elif event.key == pygame.K_b:
                    bind_sps_to_selection()
                elif event.key == pygame.K_r:
                    reset_current_mode()
                elif event.key == pygame.K_e:
                    export_current_target()
                elif event.key == pygame.K_w and mode == "IEC":
                    for candidate in candidates:
                        breeder.randomize_weights(candidate.genome)
                        candidate.net = neat.nn.FeedForwardNetwork.create(candidate.genome, config)
                elif event.key == pygame.K_n and mode == "IEC":
                    parents = [candidate.genome for candidate in candidates if candidate.selected]
                    if parents:
                        generation += 1
                        candidates = build_batch(
                            breeder.spawn_generation(parents, n=9, keep_elites=True),
                            key_start=generation * 9 + 1,
                        )
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if mode == "SPS" and 0 <= idx < len(sps_candidates):
                        choose_sps_sample(idx)
                    elif mode == "IEC" and 0 <= idx < len(candidates):
                        candidates[idx].selected = not candidates[idx].selected
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                idx = candidate_index_at_pos(event.pos)
                if idx >= 0:
                    if mode == "IEC":
                        candidates[idx].selected = not candidates[idx].selected
                    else:
                        choose_sps_sample(idx)

        if not paused:
            for candidate in active_candidates():
                candidate.system.update(candidate.net, dt=dt, scale=args.scale)

        screen.fill((18, 18, 22))

        for i, candidate in enumerate(active_candidates()):
            x, y, w, h = cell_rect(i)
            if mode == "IEC":
                border = (240, 220, 120) if candidate.selected else (70, 70, 80)
            else:
                border = (120, 180, 240) if i == 4 else (70, 70, 80)
            pygame.draw.rect(screen, border, (x, y, w, h), 2)

            left_w = int(w * 0.62)
            preview = (x + 6, y + 6, left_w - 12, h - 12)
            netrect = (x + left_w, y + 6, w - left_w - 6, h - 12)

            pygame.draw.rect(screen, (10, 10, 14), preview, 0)
            pygame.draw.rect(screen, (12, 12, 16), netrect, 0)

            candidate.system.draw(screen, preview)
            draw_genome(screen, netrect, candidate.genome, config, small)

            title = (
                f"key={candidate.species_key}{target_distance_label(candidate.genome)}: {candidate.label}"
                if candidate.label
                else f"key={candidate.species_key}{target_distance_label(candidate.genome)}"
            )
            screen.blit(font.render(title, True, (220, 220, 220)), (x + 8, y + 6))

        bar_y = screen.get_height() - bottom_h
        pygame.draw.rect(screen, (12, 12, 16), (0, bar_y, screen.get_width(), bottom_h), 0)
        if mode == "SPS":
            bound_key = sps_bound_species_key if sps_bound_genome is not None else "none"
            steps = len(sps_search.history) if sps_search is not None else 0
            status = f"mode=SPS-weight  system=generic  bound_key={bound_key}  weight slots={len(sps_weight_slots)}  sps_steps={steps}  paused={paused}"
            help1 = "click / 1..9: choose preferred sample   Tab: IEC   B: rebind genome   R: reset SPS"
        else:
            status = f"mode=IEC  system=generic  generation={generation}  paused={paused}"
            help1 = "click / 1..9: select   N: new gen   B/Tab: SPS bind   R: reset   W: randomize weights   E: export genome"
        if target_path:
            status = f"{status}  target={os.path.basename(target_path)}"
        screen.blit(font.render(status, True, (230, 230, 230)), (12, bar_y + 15))
        screen.blit(font.render(help1, True, (200, 200, 200)), (12, bar_y + 32))
        if last_export_path:
            screen.blit(font.render(f"export={last_export_path}", True, (170, 220, 170)), (12, bar_y + 49))


        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
