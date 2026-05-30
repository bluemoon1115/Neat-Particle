# from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _dist3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _dist2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def _unit_sphere_point(rng: random.Random) -> Tuple[float, float, float]:
    # Rejection sampling inside a unit sphere.
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        if x * x + y * y + z * z <= 1.0:
            return x, y, z


def _to_rgb01(v: float) -> int:
    # With sigmoid outputs, values are typically in [0, 1].
    return int(_clamp(v, 0.0, 1.0) * 255.0)


def _vel_from_sigmoid(v: float, speed: float) -> float:
    # Map [0, 1] -> [-1, 1], then scale.
    return (v * 2.0 - 1.0) * speed


@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    ttl: float = 2.0
    age: float = 0.0


@dataclass
class TrailDot:
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float
    ttl: float
    age: float = 0.0


class BaseSystem:
    name: str = "base"

    def reset(self, rng: random.Random) -> None:
        raise NotImplementedError

    def update(self, net, dt: float, scale: float) -> None:
        raise NotImplementedError

    def draw(self, surface, rect) -> None:
        raise NotImplementedError


class GenericSystem(BaseSystem):
    name = "generic"

    def __init__(self, num_particles: int = 80, bounds: float = 1.0, seed: Optional[int] = None):
        self.num_particles = num_particles
        self.bounds = bounds
        self.rng = random.Random(seed)
        self.particles: List[Particle] = []
        self.reset(self.rng)

    def reset(self, rng: random.Random) -> None:
        self.particles = []
        for _ in range(self.num_particles):
            ux, uy, uz = _unit_sphere_point(rng)
            # Spawn near center.
            self.particles.append(Particle(x=ux * 0.2, y=uy * 0.2, z=uz * 0.2, ttl=rng.uniform(1.0, 3.0)))

    def _respawn(self, p: Particle) -> None:
        ux, uy, uz = _unit_sphere_point(self.rng)
        p.x, p.y, p.z = ux * 0.2, uy * 0.2, uz * 0.2
        p.vx = p.vy = p.vz = 0.0
        p.r = p.g = p.b = 1.0
        p.age = 0.0
        p.ttl = self.rng.uniform(1.0, 3.0)

    def step_particle(self, net, p: Particle, dt: float, scale: float) -> None:
        dc = _dist3(p.x, p.y, p.z)
        inputs = (p.x, p.y, p.z, dc, 1.0)  # bias
        vx, vy, vz, r, g, b = net.activate(inputs)
        p.vx = _vel_from_sigmoid(vx, speed=1.0)
        p.vy = _vel_from_sigmoid(vy, speed=1.0)
        p.vz = _vel_from_sigmoid(vz, speed=1.0)
        p.r, p.g, p.b = r, g, b

        p.x = p.x + scale * p.vx * dt
        p.y = p.y + scale * p.vy * dt
        p.z = p.z + scale * p.vz * dt

        p.age += dt
        if p.age >= p.ttl:
            self._respawn(p)
            return

        # Keep within bounds by wrapping.
        b = self.bounds
        if (p.x < -b) | (p.x > b) | (p.y < -b) | (p.y > b) | (p.z < -b) | (p.z > b):
            self._respawn(p)
        # if p.x < -b:
        #     p.x += 2 * b
        # elif p.x > b:
        #     p.x -= 2 * b
        # if p.y < -b:
        #     p.y += 2 * b
        # elif p.y > b:
        #     p.y -= 2 * b
        # if p.z < -b:
        #     p.z += 2 * b
        # elif p.z > b:
        #     p.z -= 2 * b


    def update(self, net, dt: float, scale: float) -> None:
        for p in self.particles:
            self.step_particle(net, p, dt, scale)

    def draw(self, surface, rect) -> None:
        import pygame

        x0, y0, w, h = rect
        cx = x0 + w // 2
        cy = y0 + h // 2
        s = min(w, h) * 0.45
        for p in self.particles:
            px = int(cx + p.x * s)
            py = int(cy + p.y * s)
            color = (_to_rgb01(p.r), _to_rgb01(p.g), _to_rgb01(p.b))
            pygame.draw.circle(surface, color, (px, py), 2)


class TrailSystem(GenericSystem):
    name = "trail"

    def __init__(self, num_particles: int = 45, trail_len: int = 20, seed: Optional[int] = None):
        self.trail_len = trail_len
        self.trails: List[List[TrailDot]] = []
        super().__init__(num_particles=num_particles, seed=seed)

    def reset(self, rng: random.Random) -> None:
        super().reset(rng)
        self.trails = [[] for _ in range(self.num_particles)]

    def update(self, net, dt: float, scale: float) -> None:
        for idx, p in enumerate(self.particles):
            self.step_particle(net, p, dt, scale)
            td = TrailDot(x=p.x, y=p.y, z=p.z, r=p.r, g=p.g, b=p.b, ttl=0.7)
            tr = self.trails[idx]
            tr.append(td)
            if len(tr) > self.trail_len:
                tr.pop(0)

        # Age trail dots.
        for tr in self.trails:
            for dot in tr:
                dot.age += dt
            tr[:] = [d for d in tr if d.age < d.ttl]

    def draw(self, surface, rect) -> None:
        import pygame

        x0, y0, w, h = rect
        cx = x0 + w // 2
        cy = y0 + h // 2
        s = min(w, h) * 0.45

        for tr in self.trails:
            for dot in tr:
                t = 1.0 - (dot.age / dot.ttl if dot.ttl > 0 else 1.0)
                px = int(cx + dot.x * s)
                py = int(cy + dot.y * s)
                color = (int(_to_rgb01(dot.r) * t), int(_to_rgb01(dot.g) * t), int(_to_rgb01(dot.b) * t))
                pygame.draw.circle(surface, color, (px, py), 1)

        super().draw(surface, rect)


class BeamSystem(BaseSystem):
    name = "beam"

    def __init__(self, num_ctrl: int = 6, seed: Optional[int] = None):
        self.num_ctrl = num_ctrl
        self.rng = random.Random(seed)
        self.ctrl: List[Particle] = []
        # A fixed target (paper: some point away from system position).
        self.target = (0.0, -0.8, 0.0)
        self.reset(self.rng)

    def reset(self, rng: random.Random) -> None:
        self.ctrl = []
        for i in range(self.num_ctrl):
            t = i / max(1, self.num_ctrl - 1)
            # Initial control points along a line towards target.
            x = 0.0
            y = _clamp(-0.1 - t * 0.7, -1.0, 1.0)
            z = 0.0
            self.ctrl.append(Particle(x=x, y=y, z=z, ttl=rng.uniform(2.0, 4.0)))

    def update(self, net, dt: float, scale: float) -> None:
        tx, ty, tz = self.target
        for p in self.ctrl:
            dtgt = _dist3(p.x - tx, p.y - ty, p.z - tz)
            inputs = (p.x, p.y, p.z, dtgt, 1.0)
            vx, vy, vz, r, g, b = net.activate(inputs)
            p.vx = _vel_from_sigmoid(vx, speed=0.9)
            p.vy = _vel_from_sigmoid(vy, speed=0.9)
            p.vz = _vel_from_sigmoid(vz, speed=0.9)
            p.r, p.g, p.b = r, g, b

            p.x = p.x + scale * p.vx * dt
            p.y = p.y + scale * p.vy * dt
            p.z = p.z + scale * p.vz * dt

    def _bezier_point(self, pts: Sequence[Particle], t: float) -> Tuple[float, float, float]:
        # De Casteljau.
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        zs = [p.z for p in pts]
        n = len(pts)
        for r in range(1, n):
            for i in range(n - r):
                xs[i] = xs[i] * (1 - t) + xs[i + 1] * t
                ys[i] = ys[i] * (1 - t) + ys[i + 1] * t
                zs[i] = zs[i] * (1 - t) + zs[i + 1] * t
        return xs[0], ys[0], zs[0]

    def draw(self, surface, rect) -> None:
        import pygame

        x0, y0, w, h = rect
        cx = x0 + w // 2
        cy = y0 + h // 2
        s = min(w, h) * 0.45

        # Draw the curve.
        if len(self.ctrl) >= 2:
            prev = None
            for i in range(40):
                t = i / 39.0
                x, y, _z = self._bezier_point(self.ctrl, t)
                px = int(cx + x * s)
                py = int(cy + y * s)
                if prev is not None:
                    pygame.draw.line(surface, (180, 180, 220), prev, (px, py), 2)
                prev = (px, py)

        # Draw control points with their colors.
        for p in self.ctrl:
            px = int(cx + p.x * s)
            py = int(cy + p.y * s)
            color = (_to_rgb01(p.r), _to_rgb01(p.g), _to_rgb01(p.b))
            pygame.draw.circle(surface, color, (px, py), 3)


class PlaneSystem(BaseSystem):
    name = "plane"

    def __init__(self, num_quads: int = 12, fixed_y: float = 0.0, seed: Optional[int] = None):
        self.num_quads = num_quads
        self.fixed_y = fixed_y
        self.rng = random.Random(seed)
        self.quads: List[List[Particle]] = []
        self.reset(self.rng)

    def reset(self, rng: random.Random) -> None:
        self.quads = []
        for _ in range(self.num_quads):
            # Quad center in XZ.
            cx = rng.uniform(-0.4, 0.4)
            cz = rng.uniform(-0.4, 0.4)
            size = rng.uniform(0.05, 0.12)
            corners = [
                Particle(x=cx - size, y=self.fixed_y, z=cz - size, ttl=rng.uniform(2.0, 4.0)),
                Particle(x=cx + size, y=self.fixed_y, z=cz - size, ttl=rng.uniform(2.0, 4.0)),
                Particle(x=cx + size, y=self.fixed_y, z=cz + size, ttl=rng.uniform(2.0, 4.0)),
                Particle(x=cx - size, y=self.fixed_y, z=cz + size, ttl=rng.uniform(2.0, 4.0)),
            ]
            self.quads.append(corners)

    def update(self, net, dt: float, scale: float) -> None:
        for corners in self.quads:
            # center in XZ
            mx = sum(p.x for p in corners) / 4.0
            mz = sum(p.z for p in corners) / 4.0
            for p in corners:
                dc = _dist2(p.x - mx, p.z - mz)
                # Implemented input set matches user request: fixed Y, Px, Pz, Dc, Bias
                inputs = (self.fixed_y, p.x, p.z, dc, 1.0)
                vx, vz, r, g, b = net.activate(inputs)
                p.vx = _vel_from_sigmoid(vx, speed=0.9)
                p.vz = _vel_from_sigmoid(vz, speed=0.9)
                p.r, p.g, p.b = r, g, b

                p.x = p.x + scale * p.vx * dt
                p.z = p.z + scale * p.vz * dt

            # Keep quads near bounds by mild wrapping.
            for p in corners:
                if p.x < -1.0:
                    p.x += 2.0
                elif p.x > 1.0:
                    p.x -= 2.0
                if p.z < -1.0:
                    p.z += 2.0
                elif p.z > 1.0:
                    p.z -= 2.0

    def draw(self, surface, rect) -> None:
        import pygame

        x0, y0, w, h = rect
        cx = x0 + w // 2
        cy = y0 + h // 2
        s = min(w, h) * 0.45

        for corners in self.quads:
            pts = [(int(cx + p.x * s), int(cy + p.z * s)) for p in corners]
            # Average corner color.
            r = sum(p.r for p in corners) / 4.0
            g = sum(p.g for p in corners) / 4.0
            b = sum(p.b for p in corners) / 4.0
            color = (_to_rgb01(r), _to_rgb01(g), _to_rgb01(b))
            pygame.draw.polygon(surface, color, pts, 1)


def make_system(system_name: str, seed: Optional[int] = None) -> BaseSystem:
    name = system_name.lower().strip()
    if name == "generic":
        return GenericSystem(seed=seed)
    if name == "trail":
        return TrailSystem(seed=seed)
    if name == "beam":
        return BeamSystem(seed=seed)
    if name == "plane":
        return PlaneSystem(seed=seed)
    raise ValueError(f"Unknown system: {system_name!r}")
