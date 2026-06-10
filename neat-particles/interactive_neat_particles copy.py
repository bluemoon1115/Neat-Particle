'''
SPS for input transform parameters (vector size = 9)
'''

from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import dataclass
from typing import List, Optional
import pygame

import neat
from neat.innovation import InnovationTracker

from draw_genome import draw_genome
from particle_systems import BaseSystem, make_system
from sequential_plane_search import InputTransform, SequentialPlaneSearch


@dataclass
class Candidate:
    index: int
    genome: neat.DefaultGenome
    net: object
    system: BaseSystem
    input_transform: Optional[InputTransform] = None
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
        k = self.next_key
        self.next_key += 1
        return k

    def create_random_genome(self) -> neat.DefaultGenome:
        self.genome_config.innovation_tracker = self.innovation_tracker
        g = neat.DefaultGenome(self._new_key())
        g.configure_new(self.genome_config)
        return g

    def clone_and_mutate(self, parent: neat.DefaultGenome) -> neat.DefaultGenome:
        self.genome_config.innovation_tracker = self.innovation_tracker
        child = copy.deepcopy(parent)
        child.key = self._new_key()
        child.fitness = None
        child.mutate(self.genome_config)
        return child

    def randomize_weights(self, genome: neat.DefaultGenome) -> None:
        # A "refresh with different weights" button: keep topology but randomize weights/biases.
        for cg in genome.connections.values():
            cg.weight = self.rng.uniform(-1.0, 1.0)
        for ng in genome.nodes.values():
            ng.bias = self.rng.uniform(-1.0, 1.0)

    def spawn_generation(self, parents: List[neat.DefaultGenome], n: int = 9, keep_elites: bool = True) -> List[neat.DefaultGenome]:
        # Reset generation-specific innovation dedup tracking (matches DefaultReproduction behavior).
        self.innovation_tracker.reset_generation()

        if not parents:
            return [self.create_random_genome() for _ in range(n)]

        out: List[neat.DefaultGenome] = []

        if keep_elites:
            # Copy selected genomes through unchanged (still assign fresh keys).
            for p in parents[: min(len(parents), n)]:
                elite = copy.deepcopy(p)
                elite.key = self._new_key()
                elite.fitness = None
                out.append(elite)

        while len(out) < n:
            parent = self.rng.choice(parents)
            out.append(self.clone_and_mutate(parent))

        return out[:n]


def main() -> int:
# execution arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.45, help="S in P_t = P_{t-1} + S*V*T")
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

    breeder = InteractiveBreeder(config, seed=args.seed)

# setup for the GUI
    pygame.init()
    pygame.display.set_caption("NEAT Particles")
    screen = pygame.display.set_mode((1300, 720), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    small = pygame.font.SysFont("consolas", 12)

    def make_candidate(idx: int, genome, system_seed: Optional[int] = None,
                       input_transform: Optional[InputTransform] = None,
                       label: Optional[str] = None) -> Candidate:
        """Build a renderable Generic gallery cell from a genome."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        seed = system_seed if system_seed is not None else (args.seed or 0) + idx * 67
        system = make_system(seed=seed)
        return Candidate(index=idx, genome=genome, net=net, system=system,
                         input_transform=input_transform, label=label)

    def build_batch(genomes) -> List[Candidate]:
        cands: List[Candidate] = []
        for i, g in enumerate(genomes, start=1):
            cands.append(make_candidate(i, g))
        return cands

    candidates = build_batch([breeder.create_random_genome() for _ in range(9)])
    mode = "IEC"
    sps_search = None
    sps_bound_genome = None
    sps_bound_index = 0
    sps_candidates: List[Candidate] = []
    paused = False
    generation = 0

    # Grid layout.
    margin = 12
    cell_w = (screen.get_width() - margin * 4) // 3
    cell_h = (screen.get_height() - margin * 4 - 70) // 3
    bottom_h = 70

    def cell_rect(i: int):
        # defines the cell box position
        # i: 0..8
        row = i // 3
        col = i % 3
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        return (x, y, cell_w, cell_h)

    # check which grid we are clicking and returns the index of grid
    def candidate_index_at_pos(pos):
        mx, my = pos
        active_candidates = sps_candidates if mode == "SPS" else candidates
        for i, c in enumerate(active_candidates):
            x, y, w, h = cell_rect(i)
            if x <= mx <= x + w and y <= my <= y + h:
                return i
        return -1

    def build_sps_batch(bound_genome, seed) -> List[Candidate]:
        """Build the 3x3 SPS gallery from the current search-plane samples."""
        shared_seed = (seed or 0) + 67
        batch = []
        for i, transform in enumerate(sps_search.transforms()):
            label = f"key={bound_genome.key} {transform.short_label()}"
            batch.append(make_candidate(i + 1, bound_genome, system_seed=shared_seed,
                                        input_transform=transform, label=label))
        return batch

    def bind_sps_to_selection() -> None:
        """Bind SPS to the selected IEC genome, or candidate 1 if none is selected."""
        nonlocal sps_bound_genome, sps_bound_index, sps_candidates, sps_search, mode
        selected_indices = [i for i, c in enumerate(candidates) if c.selected]
        sps_bound_index = selected_indices[0] if selected_indices else 0
        source = candidates[sps_bound_index]
        sps_bound_genome = copy.deepcopy(source.genome)
        sps_search = SequentialPlaneSearch(seed=args.seed)
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
        else:
            sps_search.reset()
            sps_candidates = build_sps_batch(sps_bound_genome, args.seed)

    def choose_sps_sample(idx: int) -> None:
        """Accept one SPS sample as preferred and generate the next search plane."""
        nonlocal sps_candidates
        if sps_bound_genome is None:
            bind_sps_to_selection()
        sps_search.observe(idx)
        sps_candidates = build_sps_batch(sps_bound_genome, args.seed)

    running = True
    while running:
        dt = clock.tick(args.fps) / 1000.0

# in game key events
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
                        if sps_bound_genome is None:
                            bind_sps_to_selection()
                        else:
                            mode = "SPS"
                            sps_candidates = build_sps_batch(sps_bound_genome, args.seed)
                    else:
                        mode = "IEC"
                elif event.key == pygame.K_b:
                    bind_sps_to_selection()
                elif event.key == pygame.K_r:
                    reset_current_mode()
                elif event.key == pygame.K_w:
                    if mode == "IEC":
                        for c in candidates:
                            breeder.randomize_weights(c.genome)
                            c.net = neat.nn.FeedForwardNetwork.create(c.genome, config)
                elif event.key == pygame.K_n:
                    if mode == "IEC":
                        parents = [c.genome for c in candidates if c.selected]
                        if parents:
                            generation += 1
                            new_genomes = breeder.spawn_generation(parents, n=9, keep_elites=True)
                            candidates = build_batch(new_genomes)
                else:
                    if pygame.K_1 <= event.key <= pygame.K_9:
                        idx = event.key - pygame.K_1
                        if mode == "IEC":
                            if 0 <= idx < len(candidates):
                                candidates[idx].selected = not candidates[idx].selected
                        elif 0 <= idx < len(sps_search.current_samples):
                            choose_sps_sample(idx)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                idx = candidate_index_at_pos(event.pos)
                if idx >= 0:
                    if mode == "IEC":
                        candidates[idx].selected = not candidates[idx].selected
                    else:
                        choose_sps_sample(idx)

        if not paused:
            active_candidates = sps_candidates if mode == "SPS" else candidates
            for c in active_candidates:
                c.system.update(c.net, dt=dt, scale=args.scale, input_transform=c.input_transform)

        # Draw.
        screen.fill((18, 18, 22))

        active_candidates = sps_candidates if mode == "SPS" else candidates
        for i, c in enumerate(active_candidates):
            x, y, w, h = cell_rect(i)
            if mode == "IEC":
                border = (240, 220, 120) if c.selected else (70, 70, 80)
            else:
                border = (120, 180, 240) if i == 4 else (70, 70, 80)
            pygame.draw.rect(screen, border, (x, y, w, h), 2)

            # Split: left = particle preview, right = genome diagram.
            left_w = int(w * 0.62)
            preview = (x + 6, y + 6, left_w - 12, h - 12)
            netrect = (x + left_w, y + 6, w - left_w - 6, h - 12)

            # Backgrounds.
            pygame.draw.rect(screen, (10, 10, 14), preview, 0)
            pygame.draw.rect(screen, (12, 12, 16), netrect, 0)

            c.system.draw(screen, preview)
            draw_genome(screen, netrect, c.genome, config, small)

            if c.label is not None:
                title = f"{c.index}: {c.label}"
            else:
                title = f"{c.index}: key={c.genome.key}"
            txt = font.render(title, True, (220, 220, 220))
            screen.blit(txt, (x + 8, y + 6))

        # Bottom help/status bar.
        bar_y = screen.get_height() - bottom_h
        pygame.draw.rect(screen, (12, 12, 16), (0, bar_y, screen.get_width(), bottom_h), 0)
        if mode == "SPS":
            bound_key = sps_bound_genome.key if sps_bound_genome is not None else "none"
            status = f"mode=SPS-input-transform  system=generic  bound_key={bound_key}  sps_steps={len(sps_search.history)}  paused={paused}"
            help1 = "click / 1..9: choose preferred sample   Tab: IEC   B: rebind genome   R: reset SPS"
        else:
            status = f"mode=IEC  system=generic  generation={generation}  paused={paused}"
            help1 = "click / 1..9: select   N: new gen   B/Tab: SPS bind   R: reset   W: randomize weights"
        screen.blit(font.render(status, True, (230, 230, 230)), (12, bar_y + 15))
        screen.blit(font.render(help1, True, (200, 200, 200)), (12, bar_y + 32))

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
