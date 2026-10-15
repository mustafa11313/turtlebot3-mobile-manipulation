"""Frontier detection on an occupancy grid (pure numpy, unit-tested).

A frontier cell is a free cell with at least one unknown 4-neighbour.
Frontier cells are grouped into 8-connected clusters; each cluster is scored
by size and distance so the explorer heads for large, nearby unexplored areas.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

FREE_MAX = 25        # occupancy values <= this are free
UNKNOWN = -1


@dataclass
class Frontier:
    cells: int
    centroid: Tuple[float, float]       # world coordinates
    goal: Tuple[float, float]           # reachable free cell closest to the centroid
    distance: float = 0.0
    score: float = 0.0


def frontier_mask(grid: np.ndarray) -> np.ndarray:
    free = (grid >= 0) & (grid <= FREE_MAX)
    unknown = grid == UNKNOWN
    nb = np.zeros_like(unknown)
    nb[1:, :] |= unknown[:-1, :]
    nb[:-1, :] |= unknown[1:, :]
    nb[:, 1:] |= unknown[:, :-1]
    nb[:, :-1] |= unknown[:, 1:]
    return free & nb


def _label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected component labelling without scipy."""
    labels = np.zeros(mask.shape, np.int32)
    n = 0
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if labels[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        labels[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not labels[yy, xx]:
                        labels[yy, xx] = n
                        stack.append((yy, xx))
    return labels, n


def clearance_mask(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    """Free cells at least ``radius_cells`` away from any occupied cell."""
    occ = grid > 50
    blocked = occ.copy()
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            blocked |= np.roll(np.roll(occ, dy, 0), dx, 1)
    return (grid >= 0) & (grid <= FREE_MAX) & ~blocked


def _split_by_sector(xs: np.ndarray, ys: np.ndarray, center: Tuple[float, float],
                     sectors: int = 8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a (possibly ring-shaped) cluster into angular sectors around ``center``."""
    ang = np.arctan2(ys - center[1], xs - center[0])
    idx = ((ang + math.pi) / (2 * math.pi) * sectors).astype(int) % sectors
    return [(xs[idx == k], ys[idx == k]) for k in range(sectors) if np.any(idx == k)]


def find_frontiers(grid: np.ndarray, resolution: float, origin: Sequence[float],
                   robot_xy: Sequence[float], min_cells: int = 8,
                   robot_radius: float = 0.25, blacklist: Sequence[Tuple[float, float]] = (),
                   blacklist_radius: float = 0.5, split_cells: int = 60,
                   min_goal_distance: float = 0.5) -> List[Frontier]:
    mask = frontier_mask(grid)
    labels, n = _label(mask)
    safe = clearance_mask(grid, max(1, int(math.ceil(robot_radius / resolution))))
    safe_ys, safe_xs = np.nonzero(safe)
    if len(safe_xs) == 0:
        return []
    sgx = origin[0] + (safe_xs + 0.5) * resolution
    sgy = origin[1] + (safe_ys + 0.5) * resolution
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(labels == k)
        if len(xs) < min_cells:
            continue
        wx = origin[0] + (xs + 0.5) * resolution
        wy = origin[1] + (ys + 0.5) * resolution
        parts = _split_by_sector(wx, wy, tuple(robot_xy)) if len(xs) > split_cells else [(wx, wy)]
        for px, py in parts:
            if len(px) < min_cells:
                continue
            # aim at the frontier cell closest to the robot (but not on top of it)
            d = np.hypot(px - robot_xy[0], py - robot_xy[1])
            far = d >= min_goal_distance
            if not np.any(far):
                continue
            i0 = int(np.argmin(np.where(far, d, np.inf)))
            ax, ay = float(px[i0]), float(py[i0])
            cx, cy = float(px.mean()), float(py.mean())
            if any(math.hypot(ax - bx, ay - by) < blacklist_radius for bx, by in blacklist):
                continue
            # goal: the safe free cell closest to that frontier point
            i = int(np.argmin((sgx - ax) ** 2 + (sgy - ay) ** 2))
            if math.hypot(sgx[i] - ax, sgy[i] - ay) > 1.0:
                continue
            goal = (float(sgx[i]), float(sgy[i]))
            dist = math.hypot(goal[0] - robot_xy[0], goal[1] - robot_xy[1])
            if dist < 0.25:
                continue
            f = Frontier(len(px), (cx, cy), goal, dist)
            f.target = (ax, ay)
            out.append(f)
    for f in out:
        f.score = f.cells * resolution - 1.0 * f.distance   # metres of frontier vs metres to travel
    return sorted(out, key=lambda f: -f.score)


def best_frontier(frontiers: List[Frontier], min_distance: float = 0.4) -> Optional[Frontier]:
    cands = [f for f in frontiers if f.distance >= min_distance]
    return cands[0] if cands else (frontiers[0] if frontiers else None)
