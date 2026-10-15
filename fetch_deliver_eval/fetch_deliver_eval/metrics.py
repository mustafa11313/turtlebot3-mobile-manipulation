"""Ground-truth metrics (pure Python, unit-tested).

Everything here is computed from simulator ground truth, never from what the
robot believes: collisions from the true robot footprint vs. the true
geometry, detections scored against the true object positions, placement
error from the true released object pose.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from fetch_deliver_gazebo.layouts import Layout

FOOTPRINT = ((0.19, 0.16), (0.19, -0.16), (-0.22, -0.16), (-0.22, 0.16))


def _seg_dist(px, py, x1, y1, x2, y2) -> float:
    vx, vy = x2 - x1, y2 - y1
    l2 = vx * vx + vy * vy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / l2))
    return math.hypot(px - (x1 + t * vx), py - (y1 + t * vy))


def footprint_points(pose: Sequence[float], step: float = 0.03) -> List[Tuple[float, float]]:
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for i in range(len(FOOTPRINT)):
        (ax, ay), (bx, by) = FOOTPRINT[i], FOOTPRINT[(i + 1) % len(FOOTPRINT)]
        n = max(1, int(math.hypot(bx - ax, by - ay) / step))
        for k in range(n):
            lx, ly = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
            pts.append((x + c * lx - s * ly, y + s * lx + c * ly))
    return pts


def clearance(layout: Layout, pose: Sequence[float],
              boxes: Sequence[Tuple[float, float, float, float, float]] = ()) -> float:
    """Smallest distance from the robot footprint to walls, furniture and extra boxes
    (x, y, yaw, sx, sy). Negative = penetration."""
    best = float('inf')
    rects = [(f.x, f.y, f.yaw, f.sx, f.sy) for f in layout.furniture] + list(boxes)
    for px, py in footprint_points(pose):
        for x1, y1, x2, y2 in layout.walls:
            best = min(best, _seg_dist(px, py, x1, y1, x2, y2) - layout.wall_thickness / 2)
        for bx, by, byaw, sx, sy in rects:
            c, s = math.cos(-byaw), math.sin(-byaw)
            lx, ly = c * (px - bx) - s * (py - by), s * (px - bx) + c * (py - by)
            dx, dy = abs(lx) - sx / 2, abs(ly) - sy / 2
            if dx <= 0 and dy <= 0:
                d = max(dx, dy)
            else:
                d = math.hypot(max(dx, 0.0), max(dy, 0.0))
            best = min(best, d)
    return best


class CollisionCounter:
    """Counts contact episodes (clearance < threshold), with hysteresis."""

    def __init__(self, threshold: float = 0.005, release: float = 0.03):
        self.threshold, self.release = threshold, release
        self.in_contact = False
        self.count = 0
        self.min_clearance = float('inf')

    def update(self, c: float) -> None:
        self.min_clearance = min(self.min_clearance, c)
        if not self.in_contact and c < self.threshold:
            self.in_contact = True
            self.count += 1
        elif self.in_contact and c > self.release:
            self.in_contact = False


def segments_intersect(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def line_of_sight(layout: Layout, a, b, min_height: float = 0.15) -> bool:
    """True if no wall or furniture taller than ``min_height`` blocks a->b."""
    for x1, y1, x2, y2 in layout.walls:
        if segments_intersect(a, b, (x1, y1), (x2, y2)):
            return False
    for f in layout.furniture:
        if f.sz < min_height:
            continue
        c, s = math.cos(f.yaw), math.sin(f.yaw)
        corners = [(f.x + c * dx - s * dy, f.y + s * dx + c * dy)
                   for dx, dy in ((f.sx / 2, f.sy / 2), (f.sx / 2, -f.sy / 2),
                                  (-f.sx / 2, -f.sy / 2), (-f.sx / 2, f.sy / 2))]
        for i in range(4):
            if segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
                return False
    return True


def in_camera_view(robot_pose, obj_xy, hfov: float = 1.0856, min_range: float = 0.15,
                   max_range: float = 2.5, cam_offset: float = 0.08) -> bool:
    x, y, yaw = robot_pose
    cx, cy = x + cam_offset * math.cos(yaw), y + cam_offset * math.sin(yaw)
    dx, dy = obj_xy[0] - cx, obj_xy[1] - cy
    r = math.hypot(dx, dy)
    if not min_range <= r <= max_range:
        return False
    bearing = math.atan2(dy, dx) - yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    return abs(bearing) <= hfov / 2 * 0.9


@dataclass
class DetectionScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gate: float = 0.25
    per_label: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def precision(self) -> Optional[float]:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> Optional[float]:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    def score_frame(self, detections: Sequence[Tuple[str, float, float]],
                    visible: Sequence[Tuple[str, float, float]]) -> None:
        """detections/visible: (label, x, y) in the world frame. Greedy matching by
        distance; a match needs the same color, and the same shape if the detector
        reported one. Unmatched visible objects are misses."""
        unmatched = list(visible)
        for label, x, y in detections:
            color, _, shape = label.partition('_')
            best, best_d = None, self.gate
            for v in unmatched:
                vcolor, _, vshape = v[0].partition('_')
                if vcolor != color or (shape not in ('unknown', '') and shape != vshape):
                    continue
                d = math.hypot(v[1] - x, v[2] - y)
                if d < best_d:
                    best, best_d = v, d
            if best is None:
                self.fp += 1
            else:
                self.tp += 1
                unmatched.remove(best)
        self.fn += len(unmatched)


def placement_error(released_xy, dropoff_xy) -> float:
    return math.hypot(released_xy[0] - dropoff_xy[0], released_xy[1] - dropoff_xy[1])
