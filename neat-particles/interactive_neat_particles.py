from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import neat
from neat.innovation import InnovationTracker

from draw_genome import draw_genome
from particle_systems import BaseSystem, make_system


@dataclass
class Candidate:
    index: int
    genome: neat.DefaultGenome
    net: object
    system: BaseSystem
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


def _config_path_for_system(system: str, base_dir: str) -> str:
    system = system.lower().strip()
    if system == "plane":
        return os.path.join(base_dir, "config-plane.ini")
    return os.path.join(base_dir, "config-generic.ini")


def main() -> int:
# execution arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["generic", "trail", "beam", "plane"], default="generic")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.45, help="S in P_t = P_{t-1} + S*V*T")
    args = parser.parse_args()

# pygame debug
    try:
        import pygame
    except Exception as e:  # pragma: no cover
        print("pygame is required for this example. Install it with: pip install pygame")
        print(f"Import error: {e}")
        return 2
# calls for config files for plane or generic system
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = _config_path_for_system(args.system, base_dir)

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
    pygame.display.set_caption(f"NEAT Particles (IEC) - {args.system}")
    screen = pygame.display.set_mode((1200, 900))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    small = pygame.font.SysFont("consolas", 12)

    def make_candidate(idx: int, genome) -> Candidate:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        system = make_system(args.system, seed=(args.seed or 0) + idx * 1337)
        return Candidate(index=idx, genome=genome, net=net, system=system)

    def build_batch(genomes) -> List[Candidate]:
        cands: List[Candidate] = []
        for i, g in enumerate(genomes, start=1):
            cands.append(make_candidate(i, g))
        return cands

    candidates = build_batch([breeder.create_random_genome() for _ in range(9)])
    paused = False
    generation = 0

    # Grid layout.
    margin = 12
    cell_w = (screen.get_width() - margin * 4) // 3
    cell_h = (screen.get_height() - margin * 4 - 70) // 3
    bottom_h = 70

    def cell_rect(i: int):
        # i: 0..8
        row = i // 3
        col = i % 3
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        return (x, y, cell_w, cell_h)

    def candidate_at_pos(pos):
        mx, my = pos
        for i, c in enumerate(candidates):
            x, y, w, h = cell_rect(i)
            if x <= mx <= x + w and y <= my <= y + h:
                return c
        return None

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
                elif event.key == pygame.K_r:
                    generation = 0
                    candidates = build_batch([breeder.create_random_genome() for _ in range(9)])
                elif event.key == pygame.K_w:
                    for c in candidates:
                        breeder.randomize_weights(c.genome)
                        c.net = neat.nn.FeedForwardNetwork.create(c.genome, config)
                elif event.key == pygame.K_n:
                    parents = [c.genome for c in candidates if c.selected]
                    if parents:
                        generation += 1
                        new_genomes = breeder.spawn_generation(parents, n=9, keep_elites=True)
                        candidates = build_batch(new_genomes)
                else:
                    # 1..9 toggles selection.
                    if pygame.K_1 <= event.key <= pygame.K_9:
                        idx = event.key - pygame.K_1
                        if 0 <= idx < len(candidates):
                            candidates[idx].selected = not candidates[idx].selected
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                c = candidate_at_pos(event.pos)
                if c is not None:
                    c.selected = not c.selected

        if not paused:
            for c in candidates:
                c.system.update(c.net, dt=dt, scale=args.scale)

        # Draw.
        screen.fill((18, 18, 22))

        for i, c in enumerate(candidates):
            x, y, w, h = cell_rect(i)
            border = (240, 220, 120) if c.selected else (70, 70, 80)
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

            title = f"{c.index}: key={c.genome.key}"
            txt = font.render(title, True, (220, 220, 220))
            screen.blit(txt, (x + 8, y + 6))

        # Bottom help/status bar.
        bar_y = screen.get_height() - bottom_h
        pygame.draw.rect(screen, (12, 12, 16), (0, bar_y, screen.get_width(), bottom_h), 0)
        status = f"system={args.system}  generation={generation}  paused={paused}"
        help1 = "click / 1..9: select   N: new gen   R: reset   W: randomize weights   Space: pause   Esc: quit"
        screen.blit(font.render(status, True, (230, 230, 230)), (12, bar_y + 10))
        screen.blit(font.render(help1, True, (200, 200, 200)), (12, bar_y + 32))

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

