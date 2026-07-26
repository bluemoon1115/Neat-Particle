from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import neat

from draw_genome import draw_genome
from genome_targets import load_genome_target
from particle_similarity import FINAL_SETTINGS, compare_genomes
from particle_systems import make_system
from sps_selection import load_particle_config


Color = Tuple[int, int, int]

INPUT_LABELS = ("Px", "Py", "Pz", "Dc", "Bias")
OUTPUT_LABELS = ("Vx", "Vy", "Vz", "R", "G", "B")


def _default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config-generic.ini")


def _coerce_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_coerce_text(item) for item in value) + "]"
    return str(value)


def _wrap_text(text: str, font, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if font.size(candidate)[0] <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def _draw_lines(
    surface,
    font,
    lines: Iterable[str],
    x: int,
    y: int,
    *,
    color: Color = (218, 218, 218),
    line_height: int = 17,
    max_y: Optional[int] = None,
) -> int:
    for line in lines:
        if max_y is not None and y + line_height > max_y:
            surface.blit(font.render("...", True, color), (x, y))
            return y + line_height
        surface.blit(font.render(line, True, color), (x, y))
        y += line_height
    return y


def _target_stats(data: Dict[str, Any]) -> Dict[str, int]:
    connections = data.get("connections", [])
    enabled = sum(1 for connection in connections if connection.get("enabled", True))
    return {
        "nodes": len(data.get("nodes", [])),
        "connections": len(connections),
        "enabled_connections": enabled,
        "disabled_connections": len(connections) - enabled,
    }


def _format_key_values(items: Sequence[Tuple[str, Any]]) -> List[str]:
    return [f"{key}: {_coerce_text(value)}" for key, value in items]


def _metadata_lines(data: Dict[str, Any], json_path: str) -> List[str]:
    metadata = data.get("metadata", {})
    shape = data.get("config_shape", {})
    stats = _target_stats(data)

    lines = ["Target file", f"  {os.path.basename(json_path)}", ""]
    lines.extend(
        _format_key_values(
            (
                ("format", data.get("format")),
                ("format_version", data.get("format_version")),
                ("created", data.get("created_timestamp")),
            )
        )
    )
    lines.append("")
    lines.append("Metadata")
    lines.extend(
        "  " + line
        for line in _format_key_values(
            (
                ("genome_key", metadata.get("genome_key")),
                ("candidate_key", metadata.get("candidate_key")),
                ("mode", metadata.get("mode")),
                ("generation", metadata.get("generation")),
                ("label", metadata.get("label")),
                ("fitness", metadata.get("fitness")),
            )
        )
    )
    lines.append("")
    lines.append("Config shape")
    lines.extend(
        "  " + line
        for line in _format_key_values(
            (
                ("num_inputs", shape.get("num_inputs")),
                ("num_outputs", shape.get("num_outputs")),
                ("input_keys", shape.get("input_keys")),
                ("output_keys", shape.get("output_keys")),
                ("feed_forward", shape.get("feed_forward")),
            )
        )
    )
    lines.append("")
    lines.append("Genome stats")
    lines.extend(
        "  " + line
        for line in _format_key_values(
            (
                ("nodes", stats["nodes"]),
                ("connections", stats["connections"]),
                ("enabled", stats["enabled_connections"]),
                ("disabled", stats["disabled_connections"]),
            )
        )
    )
    return lines


def _connection_lines(data: Dict[str, Any], limit: int = 14) -> List[str]:
    lines = ["Connections"]
    for connection in data.get("connections", [])[:limit]:
        key = connection.get("key", ["?", "?"])
        state = "on" if connection.get("enabled", True) else "off"
        weight = connection.get("weight")
        innovation = connection.get("innovation")
        lines.append(f"  {key[0]} -> {key[1]}  w={_coerce_text(weight)}  {state}  inno={innovation}")

    extra = len(data.get("connections", [])) - limit
    if extra > 0:
        lines.append(f"  ... {extra} more")
    return lines


def _draw_panel_title(surface, font, rect, title: str) -> None:
    import pygame

    x, y, w, _h = rect
    pygame.draw.rect(surface, (49, 53, 62), rect, 2)
    surface.blit(font.render(title, True, (235, 235, 235)), (x + 10, y + 8))
    pygame.draw.line(surface, (49, 53, 62), (x, y + 32), (x + w, y + 32), 1)


def _draw_io_legend(surface, font, rect, config: neat.Config) -> None:
    x, y, w, h = rect
    max_y = y + h - 8
    input_pairs = zip(config.genome_config.input_keys, INPUT_LABELS)
    output_pairs = zip(config.genome_config.output_keys, OUTPUT_LABELS)
    lines = ["Input keys"]
    lines.extend(f"  {key}: {label}" for key, label in input_pairs)
    lines.append("")
    lines.append("Output keys")
    lines.extend(f"  {key}: {label}" for key, label in output_pairs)
    _draw_lines(surface, font, lines, x + 10, y + 42, color=(198, 205, 220), max_y=max_y)


def run_viewer(
    json_path: str,
    config_path: str,
    *,
    target2_path: Optional[str],
    seed: Optional[int],
    fps: int,
    scale: float,
    show_disabled: bool,
) -> None:
    """Display one exported genome target as particles, graph, and JSON details."""
    try:
        import pygame
    except Exception as exc:  # pragma: no cover
        print("pygame is required for this viewer. Install it with: pip install pygame")
        print(f"Import error: {exc}")
        return

    config = load_particle_config(config_path)
    genome, data = load_genome_target(json_path, config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    system = make_system(seed=seed)
    genome2 = None
    net2 = None
    system2 = None
    similarity = None
    if target2_path is not None:
        genome2, _data2 = load_genome_target(target2_path, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        system2 = make_system(seed=seed)
        similarity = compare_genomes(genome, genome2, config, FINAL_SETTINGS)

    pygame.init()
    pygame.display.set_caption(f"NEAT Particle Target: {os.path.basename(json_path)}")
    screen = pygame.display.set_mode((1320, 760), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    small = pygame.font.SysFont("consolas", 12)
    title_font = pygame.font.SysFont("consolas", 16, bold=True)

    paused = False
    running = True
    metadata = _metadata_lines(data, json_path)
    connections = _connection_lines(data)

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
                elif event.key == pygame.K_r:
                    system = make_system(seed=seed)
                    if target2_path is not None:
                        system2 = make_system(seed=seed)

        if not paused:
            system.update(net, dt=dt, scale=scale)
            if target2_path is not None:
                system2.update(net2, dt=dt, scale=scale)

        width, height = screen.get_size()
        margin = 14
        footer_h = 34
        content_h = height - margin * 2 - footer_h
        left_w = max(360, int(width * 0.46))
        middle_w = max(300, int(width * 0.26))
        right_w = width - left_w - middle_w - margin * 4
        if right_w < 260:
            right_w = 260
            middle_w = max(260, width - left_w - right_w - margin * 4)

        particle_rect = (margin, margin, left_w, content_h)
        graph_rect = (margin * 2 + left_w, margin, middle_w, content_h)
        details_rect = (margin * 3 + left_w + middle_w, margin, right_w, content_h)
        footer_rect = (0, height - footer_h, width, footer_h)

        screen.fill((18, 18, 22))
        pygame.draw.rect(screen, (10, 10, 14), particle_rect, 0)
        pygame.draw.rect(screen, (12, 12, 16), graph_rect, 0)
        pygame.draw.rect(screen, (12, 12, 16), details_rect, 0)

        _draw_panel_title(screen, title_font, particle_rect, "Particle Animation")
        _draw_panel_title(screen, title_font, graph_rect, "ANN Graph")
        _draw_panel_title(screen, title_font, details_rect, "Export Details")

        px, py, pw, ph = particle_rect
        animation_rect = (px + 8, py + 40, pw - 16, ph - 48)
        if target2_path is None:
            system.draw(screen, animation_rect)
        else:
            gap = 10
            half_h = (animation_rect[3] - gap) // 2
            first_rect = (animation_rect[0], animation_rect[1], animation_rect[2], half_h)
            second_rect = (
                animation_rect[0],
                animation_rect[1] + half_h + gap,
                animation_rect[2],
                animation_rect[3] - half_h - gap,
            )
            pygame.draw.rect(screen, (8, 8, 12), first_rect, 0)
            pygame.draw.rect(screen, (8, 8, 12), second_rect, 0)
            system.draw(screen, first_rect)
            system2.draw(screen, second_rect)
            screen.blit(small.render(f"target: {os.path.basename(json_path)}", True, (220, 220, 220)), (first_rect[0] + 8, first_rect[1] + 8))
            screen.blit(small.render(f"target2: {os.path.basename(target2_path)}", True, (220, 220, 220)), (second_rect[0] + 8, second_rect[1] + 8))

        gx, gy, gw, gh = graph_rect
        graph_inner = (gx + 8, gy + 40, gw - 16, int(gh * 0.68))
        legend_rect = (gx + 8, gy + 48 + int(gh * 0.68), gw - 16, gh - int(gh * 0.68) - 56)
        draw_genome(screen, graph_inner, genome, config, small, show_disabled=show_disabled)
        pygame.draw.line(screen, (49, 53, 62), (gx + 8, legend_rect[1] - 8), (gx + gw - 8, legend_rect[1] - 8), 1)
        _draw_io_legend(screen, small, legend_rect, config)

        dx, dy, dw, dh = details_rect
        y = dy + 42
        max_y = dy + dh - 10
        wrapped_metadata: List[str] = []
        for line in metadata:
            if len(line) < 2 or line.startswith("  "):
                wrapped_metadata.extend(_wrap_text(line, small, dw - 22))
            else:
                wrapped_metadata.append(line)
        y = _draw_lines(screen, small, wrapped_metadata, dx + 10, y, max_y=max_y)
        y += 8
        _draw_lines(screen, small, connections, dx + 10, y, color=(198, 205, 220), max_y=max_y)

        pygame.draw.rect(screen, (12, 12, 16), footer_rect, 0)
        status = (
            f"file={os.path.basename(json_path)}  fps={fps}  scale={scale:.3g}  "
            f"paused={paused}  Space: pause  R: reset particles  Esc: close"
        )
        if target2_path is not None:
            status = (
                f"file={os.path.basename(json_path)}  target2={os.path.basename(target2_path)}  "
                f"behavior_distance={similarity.combined_distance:.6f}  "
                f"hist={similarity.histogram_distance:.6f}  ssim={similarity.ssim_distance:.6f}  "
                f"fps={fps}  scale={scale:.3g}  "
                f"paused={paused}  Space: pause  R: reset particles  Esc: close"
            )
        screen.blit(font.render(status, True, (230, 230, 230)), (margin, height - footer_h + 10))
        pygame.display.flip()

    pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="View a neat-particles exported genome target JSON file."
    )
    parser.add_argument("--target", required=True, help="Path to a JSON file produced by genome_to_target_data/save_genome_target.")
    parser.add_argument("--target2", default=None, help="Optional second target JSON to render and compare.")
    parser.add_argument("--config", default=_default_config_path(), help="Path to particle NEAT config.")
    parser.add_argument("--seed", type=int, default=None, help="Particle simulation seed.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.45, help="Particle velocity scale.")
    parser.add_argument("--show-disabled", action="store_true", help="Draw disabled genome connections as dashed lines.")
    args = parser.parse_args()

    target_path = os.path.abspath(args.target)
    if not os.path.exists(target_path):
        print(f"Target JSON does not exist: {target_path}")
        return 2
    target2_path = os.path.abspath(args.target2) if args.target2 else None
    if target2_path is not None and not os.path.exists(target2_path):
        print(f"Second target JSON does not exist: {target2_path}")
        return 2

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"Config file does not exist: {config_path}")
        return 2

    try:
        with open(target_path, "r", encoding="utf-8") as file:
            json.load(file)
        if target2_path is not None:
            with open(target2_path, "r", encoding="utf-8") as file:
                json.load(file)
    except json.JSONDecodeError as exc:
        print(f"Target file is not valid JSON: {exc}")
        return 2

    run_viewer(
        target_path,
        config_path,
        target2_path=target2_path,
        seed=args.seed,
        fps=args.fps,
        scale=args.scale,
        show_disabled=args.show_disabled,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
