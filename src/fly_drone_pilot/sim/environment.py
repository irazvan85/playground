"""Procedural corridor environment: side walls, floor, ceiling, and randomly
placed cylindrical pillars, all textured with a checkerboard pattern so a
moving/rotating eye sees changing luminance (the raw signal real
photoreceptors send into the lamina) rather than a hand-computed flow field.
Direction selectivity emerges downstream, in the connectome-derived T4/T5
wiring (see connectome/graph.py), from these raw luminance samples -- this
module has no knowledge of "motion" at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Environment:
    half_width: float = 2.0
    ceiling: float = 3.0
    corridor_length: float = 200.0
    n_pillars: int = 18
    pillar_radius: float = 0.15
    texture_scale: float = 0.5
    seed: int = 0
    pillars: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        xs = rng.uniform(2.0, self.corridor_length - 2.0, self.n_pillars)
        ys = rng.uniform(-self.half_width * 0.85, self.half_width * 0.85, self.n_pillars)
        self.pillars = np.stack([xs, ys], axis=1)

    def _axis_plane_hit(self, origin: np.ndarray, direction: np.ndarray, axis: int, value: float):
        if abs(direction[axis]) < 1e-9:
            return None
        t = (value - origin[axis]) / direction[axis]
        return t if t > 1e-6 else None

    def _cylinder_hit(self, origin: np.ndarray, direction: np.ndarray, cx: float, cy: float, r: float):
        ox, oy = origin[0] - cx, origin[1] - cy
        dx, dy = direction[0], direction[1]
        a = dx * dx + dy * dy
        if a < 1e-12:
            return None
        b = 2 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - r * r
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
        for t in (t1, t2):
            if t > 1e-6:
                return t
        return None

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_range: float = 20.0):
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        hits = []

        for value in (self.half_width, -self.half_width):
            t = self._axis_plane_hit(origin, direction, 1, value)
            if t is not None:
                hits.append(t)
        for value in (0.0, self.ceiling):
            t = self._axis_plane_hit(origin, direction, 2, value)
            if t is not None:
                hits.append(t)
        for cx, cy in self.pillars:
            t = self._cylinder_hit(origin, direction, cx, cy, self.pillar_radius)
            if t is not None:
                hits.append(t)

        in_range = [t for t in hits if t <= max_range]
        if not in_range:
            return max_range, 0.5  # open background, neutral gray

        t_hit = min(in_range)
        hit_point = origin + t_hit * direction
        return t_hit, self._texture(hit_point)

    def _texture(self, point: np.ndarray) -> float:
        x, y, z = point
        pattern = (np.floor(x / self.texture_scale) + np.floor(z / self.texture_scale)) % 2
        return 0.9 if pattern == 0 else 0.1

    def clearance(self, position: np.ndarray) -> float:
        """Distance from `position` to the nearest wall or pillar surface (for collision checks)."""
        x, y, z = position
        d = min(self.half_width - y, y + self.half_width, self.ceiling - z, z)
        for cx, cy in self.pillars:
            d = min(d, np.hypot(x - cx, y - cy) - self.pillar_radius)
        return float(d)
