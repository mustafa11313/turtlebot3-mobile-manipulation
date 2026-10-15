"""Randomized scenario generation (pure Python, unit-testable)."""

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from .layouts import Layout
from .sdf_gen import OBJECT_TYPES

DIFFICULTIES = ('easy', 'distractors', 'occluded', 'dynamic', 'hard')


@dataclass
class PlacedObject:
    model_name: str
    object_type: str
    x: float
    y: float
    yaw: float
    is_target: bool


@dataclass
class DynamicObstacle:
    model_name: str
    waypoints: List[Tuple[float, float]]
    speed: float


@dataclass
class Scenario:
    layout: str
    seed: int
    difficulty: str
    target_type: str
    robot_start: Tuple[float, float, float]
    objects: List[PlacedObject] = field(default_factory=list)
    obstacles: List[DynamicObstacle] = field(default_factory=list)

    @property
    def target(self) -> PlacedObject:
        return next(o for o in self.objects if o.is_target)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> 'Scenario':
        d = json.loads(text)
        d['objects'] = [PlacedObject(**o) for o in d['objects']]
        d['obstacles'] = [DynamicObstacle(**o) for o in d['obstacles']]
        d['robot_start'] = tuple(d['robot_start'])
        return Scenario(**d)


def _sample_point(layout: Layout, rng: random.Random, taken: List[Tuple[float, float]],
                  min_sep: float, clearance: float, near_furniture: bool = False,
                  max_tries: int = 2000) -> Optional[Tuple[float, float]]:
    sx, sy, _ = layout.robot_start
    for _ in range(max_tries):
        if near_furniture and layout.furniture:
            # Partially hidden: right next to a piece of furniture, so it is
            # only visible from some viewpoints.
            f = rng.choice(layout.furniture)
            side = rng.choice((-1, 1))
            along = rng.uniform(-0.4, 0.4)
            c, s = math.cos(f.yaw), math.sin(f.yaw)
            if rng.random() < 0.5:
                lx, ly = side * (f.sx / 2 + clearance + 0.02), along * f.sy
            else:
                lx, ly = along * f.sx, side * (f.sy / 2 + clearance + 0.02)
            x, y = f.x + c * lx - s * ly, f.y + s * lx + c * ly
            if layout.distance_to_walls(x, y) < clearance or \
               any(g.footprint_contains(x, y, clearance) for g in layout.furniture):
                continue
        else:
            x0, y0, x1, y1 = rng.choice(layout.spawn_regions)
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if not layout.is_free(x, y, clearance):
                continue
        if math.hypot(x - sx, y - sy) < 1.0:
            continue
        if any(math.hypot(x - tx, y - ty) < min_sep for tx, ty in taken):
            continue
        return (x, y)
    return None


def generate(layout: Layout, seed: int, difficulty: str = 'easy',
             target_type: str = 'red_cube') -> Scenario:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f'difficulty must be one of {DIFFICULTIES}')
    rng = random.Random(seed)
    sc = Scenario(layout=layout.name, seed=seed, difficulty=difficulty,
                  target_type=target_type, robot_start=layout.robot_start)
    taken: List[Tuple[float, float]] = []
    dx, dy, _ = layout.dropoff_pose
    taken.append((dx, dy))  # keep the drop-off zone clear

    occluded = difficulty in ('occluded', 'hard')
    p = _sample_point(layout, rng, taken, 1.0, 0.30, near_furniture=occluded)
    if p is None:
        p = _sample_point(layout, rng, taken, 1.0, 0.30)
    taken.append(p)
    sc.objects.append(PlacedObject('target_object', target_type, p[0], p[1],
                                   rng.uniform(-math.pi, math.pi), True))

    if difficulty in ('distractors', 'hard'):
        n = rng.randint(2, 4)
        distractor_types = [t for t in OBJECT_TYPES if t != target_type]
        for i in range(n):
            q = _sample_point(layout, rng, taken, 0.6, 0.30)
            if q is None:
                continue
            taken.append(q)
            sc.objects.append(PlacedObject(f'distractor_{i}', rng.choice(distractor_types),
                                           q[0], q[1], rng.uniform(-math.pi, math.pi), False))

    if difficulty in ('dynamic', 'hard'):
        for i in range(rng.randint(1, 2)):
            a = _sample_point(layout, rng, taken, 0.8, 0.45)
            b = _sample_point(layout, rng, taken + ([a] if a else []), 0.8, 0.45)
            if a is None or b is None or math.hypot(a[0] - b[0], a[1] - b[1]) < 1.5:
                continue
            sc.obstacles.append(DynamicObstacle(f'dynamic_obstacle_{i}', [a, b],
                                                rng.uniform(0.10, 0.25)))
    return sc
