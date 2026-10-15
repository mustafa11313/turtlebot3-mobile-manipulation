"""Planar pose helpers for the mission (pure Python)."""

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Pose2D = Tuple[float, float, float]


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def standoff_candidates(target: Sequence[float], robot: Sequence[float], distance: float,
                        offsets_deg: Iterable[float] = (0, 40, -40, 80, -80, 135, -135, 180)
                        ) -> List[Pose2D]:
    """Poses at ``distance`` from ``target`` facing it, preferring the robot's side."""
    base = math.atan2(robot[1] - target[1], robot[0] - target[0])
    out = []
    for off in offsets_deg:
        a = base + math.radians(off)
        x = target[0] + distance * math.cos(a)
        y = target[1] + distance * math.sin(a)
        out.append((x, y, wrap(a + math.pi)))
    return out


class CostmapView:
    """Read-only helper around a nav_msgs/OccupancyGrid-like costmap."""

    def __init__(self, data: np.ndarray, resolution: float, origin: Sequence[float]):
        self.data, self.res, self.origin = data, resolution, origin

    def cost(self, x: float, y: float) -> int:
        i = int((y - self.origin[1]) / self.res)
        j = int((x - self.origin[0]) / self.res)
        if 0 <= i < self.data.shape[0] and 0 <= j < self.data.shape[1]:
            return int(self.data[i, j])
        return 255

    def is_free(self, x: float, y: float, threshold: int = 60, unknown_ok: bool = False) -> bool:
        c = self.cost(x, y)
        if c < 0:
            return unknown_ok
        return c < threshold

    def footprint_free(self, pose: Pose2D, half_len: float = 0.2, half_wid: float = 0.16,
                       threshold: int = 99) -> bool:
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        for lx in np.linspace(-half_len, half_len, 5):
            for ly in np.linspace(-half_wid, half_wid, 4):
                if not self.is_free(x + c * lx - s * ly, y + s * lx + c * ly, threshold):
                    return False
        return True


def pick_standoff(target, robot, distance, costmap: Optional[CostmapView],
                  exclude: Sequence[Pose2D] = ()) -> Optional[Pose2D]:
    for cand in standoff_candidates(target, robot, distance):
        if any(math.hypot(cand[0] - e[0], cand[1] - e[1]) < 0.15 for e in exclude):
            continue
        if costmap is None or costmap.footprint_free(cand):
            return cand
    return None
