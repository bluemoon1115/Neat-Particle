from __future__ import annotations

import math
from typing import Dict, Tuple


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _weight_to_color(weight: float):
    # green = positive, red = negative (similar to examples/visualize.py)
    w = _clamp(weight, -1.0, 1.0)
    t = abs(w)
    if w >= 0:
        return (0, int(_lerp(122, 255, t)), 0)
    return (int(_lerp(122, 255, t)), 0, 0)


def draw_genome(surface, rect, genome, config, font, show_disabled: bool = False) -> None:
    """
    Draws a small genome diagram into a pygame surface.

    Layout (matches the paper's zoom mode description):
    - inputs: top row
    - outputs: bottom row
    - hidden: middle row(s), sorted by key
    """
    import pygame  # local import so the module can be imported without pygame installed

    x0, y0, w, h = rect
    if w <= 10 or h <= 10:
        return

    inputs = list(config.genome_config.input_keys)
    outputs = list(config.genome_config.output_keys)
    hidden = [k for k in genome.nodes.keys() if k not in outputs]
    hidden.sort()

    # Node positions.
    node_pos: Dict[int, Tuple[int, int]] = {}

    def place_row(keys, y, pad=6):
        if not keys:
            return
        n = len(keys)
        for i, k in enumerate(keys):
            t = 0.5 if n == 1 else i / (n - 1)
            x = int(_lerp(x0 + pad, x0 + w - pad, t))
            node_pos[k] = (x, y)

    place_row(inputs, y0 + 10)
    place_row(outputs, y0 + h - 12)

    # Hidden nodes: place in up to 2 rows.
    if hidden:
        rows = 2 if len(hidden) > 6 else 1
        for idx, k in enumerate(hidden):
            r = idx % rows
            c = idx // rows
            cols = math.ceil(len(hidden) / rows)
            t = 0.5 if cols == 1 else c / (cols - 1)
            x = int(_lerp(x0 + 8, x0 + w - 8, t))
            y = int(_lerp(y0 + 22, y0 + h - 26, (r + 1) / (rows + 1)))
            node_pos[k] = (x, y)

    # Connections.
    for cg in genome.connections.values():
        if (not cg.enabled) and (not show_disabled):
            continue
        a, b = cg.key
        if a not in node_pos or b not in node_pos:
            continue
        color = _weight_to_color(cg.weight)
        width = int(1 + min(3, abs(cg.weight) / 3.0))
        style = 0 if cg.enabled else 1
        if style == 0:
            pygame.draw.line(surface, color, node_pos[a], node_pos[b], width)
        else:
            # dotted-ish: short segments
            ax, ay = node_pos[a]
            bx, by = node_pos[b]
            steps = 6
            for i in range(steps):
                if i % 2 == 0:
                    t0 = i / steps
                    t1 = (i + 1) / steps
                    p0 = (int(_lerp(ax, bx, t0)), int(_lerp(ay, by, t0)))
                    p1 = (int(_lerp(ax, bx, t1)), int(_lerp(ay, by, t1)))
                    pygame.draw.line(surface, color, p0, p1, 1)

    # Nodes.
    def draw_node(k, pos, fill, outline=(30, 30, 30)):
        pygame.draw.circle(surface, fill, pos, 4)
        pygame.draw.circle(surface, outline, pos, 4, 1)
        label = str(k)
        txt = font.render(label, True, (220, 220, 220))
        surface.blit(txt, (pos[0] + 6, pos[1] - 6))

    for k in inputs:
        if k in node_pos:
            draw_node(k, node_pos[k], (110, 110, 110))
    for k in hidden:
        if k in node_pos:
            draw_node(k, node_pos[k], (200, 200, 200))
    for k in outputs:
        if k in node_pos:
            draw_node(k, node_pos[k], (120, 160, 220))

