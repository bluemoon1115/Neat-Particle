import math
import pygame
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional, Sequence

#======== math functions ===========

def _clamp(v, lo, high) -> float:
    return max(lo, min(high, v))

def _dist3(x, y, z) -> float:
    return math.sqrt(x*x + y*y + z*z)

def _dist2(x, y) -> float:
    return math.sqrt(x*x + y*y)

def _unit_sphere_point(rng: random.Random) -> Tuple[float, float, float]:
    # return a point that is inside a unit sphere
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        if x*x + y*y + z*z <= 1.0:
            return x, y, z
        
def _update_pos(x, scale, vx, dt) -> float:
    return x + scale * vx *dt

# ====== functions for converting node values =========
def _to_rgb(v: float) -> int:
    # with sigmoid values in [0, 1]
    return int(_clamp(v, 0.0, 1.0) * 255)

def velocity_from_sigmoid(v, speed) -> float:
    # map [0, 1] to [-1, 1] and then scale
    return (v * 2.0 - 1.0) * speed

#======= particle and trail dot dataclass ===========
@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    ttl: float = 2.0
    age: float = 0.0

@dataclass
class TrailDot:
    x: float
    y: float
    z: float
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    ttl: float = 2.0
    age: float = 0.0

# ====== For each systems =======
class BaseSystem:
    # define the basics of systmes for better development of new particle systems
    name = "base"

    def reset(self, rng: random.Random) -> None:
        raise NotImplementedError
    
    def update(self, net, dt, scale) -> None:
        raise NotImplementedError
    
    def draw(self, surface, rect) -> None:
        raise NotImplementedError
    
class GenericSystem(BaseSystem):
    name = "generic"

    def __init__(self, num_particles=80, bounds=1.0, seed: Optional[int] = None):
        self.num_particles = num_particles
        self.bounds = bounds
        self.rng = random.Random(seed)
        self.particles: List[Particle] = []
        self.reset(self.rng)

    def reset(self, rng: random.Random) -> None:
        # reset this whole particle system
        self.particles = []
        for _ in range(self.num_particles):
            ux, uy, uz = _unit_sphere_point(self.rng)
            self.particles.append(Particle(x=ux*0.2, y=uy*0.2, z=uz*0.2, ttl=rng.uniform(1.0, 3.0)))

    def _respawn(self, p: Particle) -> None:
        # initialize the variables of a particle for respawning a particle
        ux, uy, uz = _unit_sphere_point(self.rng)
        p.x, p.y, p.z = ux*0.2, uy*0.2, uz*0.2
        p.vx = p.vy = p.vz = 0.0
        p.r = p.g = p.b = 1.0
        p.age = 0.0
        p.ttl = self.rng.uniform(1.0, 3.0)

    def step_particle(self, net, p: Particle, dt: float, scale: float) -> None:
        # updating the particle
        SPEED = 1.0
        dc = _dist3(p.x, p.y, p.z)
        inputs = (p.x, p.y, p.z, dc, 1.0) # 1.0 as bias
        vx, vy, vz, r, g, b = net.activate(inputs)
        p.vx = velocity_from_sigmoid(vx, speed=SPEED)
        p.vy = velocity_from_sigmoid(vy, speed=SPEED)
        p.vz = velocity_from_sigmoid(vz, speed=SPEED)
        p.r, p.g, p.b = r, g, b

        p.x = _update_pos(p.x, scale, p.vx, dt)
        p.y = _update_pos(p.y, scale, p.vy, dt)
        p.z = _update_pos(p.z, scale, p.vz, dt)

        p.age += dt
        # terminate the particle and respawn it  when exceeding its age
        if p.age >= p.ttl:
            self._respawn(p)
            return
        
        # particles respawn when getting out of the bound
        b = self.bounds
        if (p.x < -b) | (p.x > b) | (p.y < -b) | (p.y > b) | (p.z < -b) | (p.z > b):
            self._respawn(p)
        
    def update(self, net, dt, scale) -> None:
        # use the step function to update each particles
        for p in self.particles:
            self.step_particle(net, p, dt, scale)

    def draw(self, surface, rect) -> None:
        x0, y0, w, h = rect
        cx = x0 + w // 2
        cy = y0 + h // 2
        s = min(w, h) * 0.45  # keeps the particles drawn within 90% of the frame (45% as radius)
        for p in self.particles:
            px = int(cx + p.x * s)
            py = int(cy + p.y * s)
            color = (_to_rgb(p.r), _to_rgb(p.g), _to_rgb(p.b))
            pygame.draw.circle(surface, color, (px, py), 2)

class TrailSystem(GenericSystem):
    name = "trail"

    def __init__(self, num_particles=45, trail_len=20, seed: Optional[int] = None):
        self.trail_len = trail_len
        self.trails = List[List[TrailDot]] = []
        super().__init__(num_particles=num_particles, seed=seed)

    def reset(self, rng: random.Random) -> None:
        super().reset(rng)
        self.trails = [[] for _ in range(self.num_particles)]

    def update(self, net, dt, scale) -> None:
        for idx, p in enumerate(self.particles):
            self.step_particle(net, p, dt, scale)
            td = TrailDot(x=p.x, y=p.y, z=p.z, r=p.r, g=p.g, b=p.b, ttl=0.7)
            tr = self.trails[idx]
            tr.append(td)
            if len(tr) > self.trail_len:
                tr.pop(0)  # remove the oldest of the trail

        # aging for trail dots
        for tr in self.trails:
            for dot in tr:
                dot.age += dt
            tr[:] = [d for d in tr if d.age < d.ttl]

    def draw(self, surface, rect) -> None:
        x0, y0, w, h = rect
        cx = x0 + w // 2
        cy = y0 + h // 2
        s = min(w, h) * 0.45
        
        for tr in self.trails:
            for dot in tr:
                # calculate the normalized lifetime of the particle
                t = max(0.0, min(1.0, 1.0 - dot.age / dot.ttl))
                px = int(cx + dot.x * s)
                py = int(cy + dot.y * s)
                color = (int(_to_rgb(dot.r) * t), int(_to_rgb(dot.g) * t), int(_to_rgb(dot.b)) * t)
                pygame.draw.circle(surface, color, (px, py), 1)

        super().draw(surface, rect)

class BeamSystem(BaseSystem):
    