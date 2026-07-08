from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import neat

from draw_genome import draw_genome
from genome_targets import load_genome_target, save_genome_target
from particle_systems import make_system
from sps_selection import load_particle_config, run_auto_sps_selection


REPORT_HISTORY_INTERVAL = 5


def _default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config-generic.ini")


def _compact_history(history: list[dict[str, object]], interval: int = REPORT_HISTORY_INTERVAL) -> list[dict[str, object]]:
    """Keep every interval step for reports, plus the final termination step."""
    if interval < 1:
        raise ValueError("history interval must be at least 1")
    if not history:
        return []

    final_step = history[-1].get("step")
    return [
        record
        for record in history
        if record.get("step") == final_step or int(record.get("step", 0)) % interval == 0
    ]


def _write_run_outputs(args, config: neat.Config, result) -> tuple[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(args.output_dir or os.path.join(base_dir, "auto-runs"))
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_path = os.path.join(output_dir, f"final_{timestamp}_seed_{args.seed}.json")
    report_path = os.path.join(output_dir, f"report_{timestamp}_seed_{args.seed}.json")

    save_genome_target(
        final_path,
        result.final_genome,
        config,
        candidate_key=None,
        mode="AUTO_SPS",
        generation=result.steps,
        label=f"behavior_distance={result.final_distance:.6f}",
    )

    report = {
        "target_path": os.path.abspath(args.target),
        "final_genome_path": os.path.abspath(final_path),
        "config_path": os.path.abspath(args.config),
        "seed": args.seed,
        "threshold": args.threshold,
        "max_steps": args.max_steps,
        "stop_reason": result.stop_reason,
        "steps": result.steps,
        "elapsed_seconds": result.elapsed_seconds,
        "final_distance": result.final_distance,
        "final_histogram_distance": result.final_histogram_distance,
        "final_ssim_distance": result.final_ssim_distance,
        "history_interval": REPORT_HISTORY_INTERVAL,
        "history": _compact_history(result.history),
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    return os.path.abspath(final_path), os.path.abspath(report_path)


def _draw_panel(
    screen,
    font,
    small,
    rect,
    title: str,
    genome: neat.DefaultGenome,
    config: neat.Config,
    system,
) -> None:
    import pygame

    x, y, w, h = rect
    pygame.draw.rect(screen, (62, 66, 78), rect, 2)
    screen.blit(font.render(title, True, (235, 235, 235)), (x + 10, y + 8))

    content_y = y + 32
    content_h = h - 42
    preview = (x + 8, content_y, int(w * 0.62) - 14, content_h)
    netrect = (x + int(w * 0.62), content_y, w - int(w * 0.62) - 8, content_h)

    pygame.draw.rect(screen, (10, 10, 14), preview, 0)
    pygame.draw.rect(screen, (12, 12, 16), netrect, 0)
    system.draw(screen, preview)
    draw_genome(screen, netrect, genome, config, small)


def show_result_view(
    target: neat.DefaultGenome,
    final_genome: neat.DefaultGenome,
    config: neat.Config,
    *,
    seed: Optional[int],
    scale: float,
    fps: int,
    stop_reason: str,
    final_distance: float,
    steps: int,
    elapsed_seconds: float,
) -> None:
    """Open the final-only pygame comparison view."""
    try:
        import pygame
    except Exception as exc:  # pragma: no cover
        print("pygame is required for the final result view. Install it with: pip install pygame")
        print(f"Import error: {exc}")
        return

    pygame.init()
    pygame.display.set_caption("NEAT Particles Auto SPS Result")
    screen = pygame.display.set_mode((1300, 720), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    small = pygame.font.SysFont("consolas", 12)
    target_system = make_system(seed=(seed or 0) + 101)
    final_system = make_system(seed=(seed or 0) + 101)
    target_net = neat.nn.FeedForwardNetwork.create(target, config)
    final_net = neat.nn.FeedForwardNetwork.create(final_genome, config)

    running = True
    paused = False
    while running:
        dt = clock.tick(fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            target_system.update(target_net, dt=dt, scale=scale)
            final_system.update(final_net, dt=dt, scale=scale)

        screen.fill((18, 18, 22))
        margin = 16
        top_h = 42
        panel_w = (screen.get_width() - margin * 3) // 2
        panel_h = screen.get_height() - top_h - margin * 2

        summary = (
            f"stop={stop_reason}  steps={steps}  elapsed={elapsed_seconds:.3f}s  "
            f"behavior_distance={final_distance:.6f}  "
            f"Space: pause  Esc: close"
        )
        screen.blit(font.render(summary, True, (230, 230, 230)), (margin, 14))
        _draw_panel(
            screen,
            font,
            small,
            (margin, top_h, panel_w, panel_h),
            "Target",
            target,
            config,
            target_system,
        )
        _draw_panel(
            screen,
            font,
            small,
            (margin * 2 + panel_w, top_h, panel_w, panel_h),
            "Final selected",
            final_genome,
            config,
            final_system,
        )
        pygame.display.flip()

    pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless SPS auto-selection for NEAT particles.")
    parser.add_argument("--target", required=True, help="Path to an exported target genome JSON file.")
    parser.add_argument("--config", default=_default_config_path(), help="Path to particle NEAT config.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--scale", type=float, default=0.45)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-view", action="store_true", help="Skip the final pygame comparison window.")
    args = parser.parse_args()

    config = load_particle_config(args.config)
    target, _target_data = load_genome_target(args.target, config)
    result = run_auto_sps_selection(
        target,
        config,
        seed=args.seed,
        threshold=args.threshold,
        max_steps=args.max_steps,
    )
    final_path, report_path = _write_run_outputs(args, config, result)

    print(f"Stop reason: {result.stop_reason}")
    print(f"Steps: {result.steps}")
    print(f"Elapsed seconds: {result.elapsed_seconds:.6f}")
    print(f"Final behavioral distance: {result.final_distance:.6f}")
    print(f"Final histogram distance: {result.final_histogram_distance:.6f}")
    print(f"Final SSIM distance: {result.final_ssim_distance:.6f}")
    print(f"Seed: {args.seed}")
    print(f"Final genome: {final_path}")
    print(f"Run report: {report_path}")

    if not args.no_view:
        show_result_view(
            result.target,
            result.final_genome,
            config,
            seed=args.seed,
            scale=args.scale,
            fps=args.fps,
            stop_reason=result.stop_reason,
            final_distance=result.final_distance,
            steps=result.steps,
            elapsed_seconds=result.elapsed_seconds,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
