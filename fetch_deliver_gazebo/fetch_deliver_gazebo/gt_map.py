"""Ground-truth occupancy grid of a layout, in the map frame (origin = robot start).

Used to score SLAM maps (occupied-cell IoU / precision / recall) and as a
fallback map for quick tests.
"""

import math
from typing import Tuple

import numpy as np

from .layouts import Layout


def rasterize(layout: Layout, resolution: float = 0.05, margin: float = 0.5,
              obstacle_min_height: float = 0.12) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Return (grid, origin_xy) with 100 = occupied, 0 = free, in the *map* frame.

    Only geometry that intersects the lidar plane (~0.13 m) is marked occupied.
    """
    sx, sy, syaw = layout.robot_start
    c, s = math.cos(-syaw), math.sin(-syaw)

    def to_map(x, y):
        return (c * (x - sx) - s * (y - sy), s * (x - sx) + c * (y - sy))

    pts = [to_map(x, y) for w in layout.walls for x, y in ((w[0], w[1]), (w[2], w[3]))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ox, oy = min(xs) - margin, min(ys) - margin
    w = int(math.ceil((max(xs) + margin - ox) / resolution))
    h = int(math.ceil((max(ys) + margin - oy) / resolution))
    grid = np.zeros((h, w), np.int8)
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    cx = ox + (jj + 0.5) * resolution
    cy = oy + (ii + 0.5) * resolution
    for x1, y1, x2, y2 in layout.walls:
        (ax, ay), (bx, by) = to_map(x1, y1), to_map(x2, y2)
        vx, vy = bx - ax, by - ay
        l2 = vx * vx + vy * vy
        t = np.clip(((cx - ax) * vx + (cy - ay) * vy) / l2, 0, 1)
        d = np.hypot(cx - (ax + t * vx), cy - (ay + t * vy))
        grid[d <= layout.wall_thickness / 2 + resolution / 2] = 100
    for f in layout.furniture:
        if f.sz < obstacle_min_height:
            continue
        fx, fy = to_map(f.x, f.y)
        yaw = f.yaw - syaw
        cc, ss = math.cos(-yaw), math.sin(-yaw)
        lx = cc * (cx - fx) - ss * (cy - fy)
        ly = ss * (cx - fx) + cc * (cy - fy)
        grid[(np.abs(lx) <= f.sx / 2) & (np.abs(ly) <= f.sy / 2)] = 100
    return grid, (ox, oy)


def surface_cells(grid: np.ndarray) -> np.ndarray:
    """Keep only occupied cells a lidar can see (with a free 4-neighbour)."""
    occ = grid > 50
    free = grid == 0
    nb = np.zeros_like(free)
    nb[1:, :] |= free[:-1, :]
    nb[:-1, :] |= free[1:, :]
    nb[:, 1:] |= free[:, :-1]
    nb[:, :-1] |= free[:, 1:]
    out = np.zeros_like(grid)
    out[occ & nb] = 100
    return out


def occupied_iou(slam: np.ndarray, slam_origin, slam_res: float, gt: np.ndarray, gt_origin,
                 gt_res: float, tolerance: float = 0.10) -> dict:
    """Compare occupied cells with a distance tolerance (both grids in the map frame).

    precision: SLAM obstacles that are real; recall: visible real surfaces that were mapped.
    """
    def occ_points(grid, origin, res):
        ii, jj = np.nonzero(grid > 50)
        return np.stack([origin[0] + (jj + 0.5) * res, origin[1] + (ii + 0.5) * res], axis=1)

    a = occ_points(slam, slam_origin, slam_res)
    b = occ_points(surface_cells(gt), gt_origin, gt_res)
    if len(a) == 0 or len(b) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def frac_within(p, q):
        # fraction of points in p with a neighbour in q within tolerance (grid hashing)
        cell = tolerance
        buckets = {}
        for x, y in q:
            buckets.setdefault((int(x // cell), int(y // cell)), []).append((x, y))
        hit = 0
        for x, y in p:
            kx, ky = int(x // cell), int(y // cell)
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for qx, qy in buckets.get((kx + dx, ky + dy), ()):
                        if (qx - x) ** 2 + (qy - y) ** 2 <= tolerance ** 2:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            hit += found
        return hit / len(p)

    precision = frac_within(a, b)
    recall = frac_within(b, a)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def save_map(grid: np.ndarray, origin, resolution: float, path_no_ext: str) -> None:
    import os
    img = np.full(grid.shape, 254, np.uint8)
    img[grid > 50] = 0
    img[grid < 0] = 205
    img = img[::-1]   # PGM rows go top-down, map rows bottom-up
    with open(path_no_ext + '.pgm', 'wb') as f:
        f.write(f'P5\n{img.shape[1]} {img.shape[0]}\n255\n'.encode())
        f.write(img.tobytes())
    with open(path_no_ext + '.yaml', 'w') as f:
        f.write(f'image: {os.path.basename(path_no_ext)}.pgm\nmode: trinary\nresolution: {resolution}\n'
                f'origin: [{origin[0]:.3f}, {origin[1]:.3f}, 0.0]\nnegate: 0\n'
                f'occupied_thresh: 0.65\nfree_thresh: 0.25\n')
