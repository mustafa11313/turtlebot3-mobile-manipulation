"""Load the shared environment layouts (config/layouts.yaml)."""

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import yaml


def default_layouts_file() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('fetch_deliver_gazebo'),
                            'config', 'layouts.yaml')
    except Exception:  # running from the source tree (tests, generator)
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, '..', 'config', 'layouts.yaml')


@dataclass
class Furniture:
    name: str
    x: float
    y: float
    yaw: float
    sx: float
    sy: float
    sz: float
    color: str = 'Gazebo/Wood'

    def footprint_contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        dx, dy = px - self.x, py - self.y
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return abs(lx) <= self.sx / 2 + margin and abs(ly) <= self.sy / 2 + margin


@dataclass
class Layout:
    name: str
    robot_start: Tuple[float, float, float]
    walls: List[Tuple[float, float, float, float]]
    furniture: List[Furniture]
    spawn_regions: List[Tuple[float, float, float, float]]
    dropoff_pose: Tuple[float, float, float]
    marker_id: int
    marker_pose: Tuple[float, float, float, float]
    search_waypoints: List[Tuple[float, float, float]]
    wall_thickness: float = 0.10
    wall_height: float = 0.60
    extra: dict = field(default_factory=dict)

    def distance_to_walls(self, px: float, py: float) -> float:
        best = float('inf')
        for x1, y1, x2, y2 in self.walls:
            vx, vy = x2 - x1, y2 - y1
            seg2 = vx * vx + vy * vy
            t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / seg2))
            cx, cy = x1 + t * vx, y1 + t * vy
            best = min(best, math.hypot(px - cx, py - cy) - self.wall_thickness / 2)
        return best

    def is_free(self, px: float, py: float, clearance: float = 0.35) -> bool:
        if self.distance_to_walls(px, py) < clearance:
            return False
        return not any(f.footprint_contains(px, py, clearance) for f in self.furniture)


def load_layouts(path: Optional[str] = None) -> dict:
    with open(path or default_layouts_file(), 'r') as f:
        data = yaml.safe_load(f)
    defaults = data.get('defaults', {})
    layouts = {}
    for name, d in data['layouts'].items():
        layouts[name] = Layout(
            name=name,
            robot_start=tuple(d['robot_start']),
            walls=[tuple(w) for w in d['walls']],
            furniture=[Furniture(f['name'], *f['pose'], *f['size'], f.get('color', 'Gazebo/Wood'))
                       for f in d.get('furniture', [])],
            spawn_regions=[tuple(r) for r in d['spawn_regions']],
            dropoff_pose=tuple(d['dropoff']['pose']),
            marker_id=int(d['dropoff']['marker_id']),
            marker_pose=tuple(d['dropoff']['marker_pose']),
            search_waypoints=[tuple(w) for w in d['search_waypoints']],
            wall_thickness=defaults.get('wall_thickness', 0.10),
            wall_height=defaults.get('wall_height', 0.60),
        )
    return layouts


def load_layout(name: str, path: Optional[str] = None) -> Layout:
    layouts = load_layouts(path)
    if name not in layouts:
        raise KeyError(f"unknown layout '{name}', available: {sorted(layouts)}")
    return layouts[name]


def world_to_map(layout: Layout, xyyaw: Sequence[float]) -> Tuple[float, float, float]:
    """Convert a world pose to the map frame of a map built from robot_start.

    slam_toolbox starts the map at the robot's initial pose, so the map frame is
    the world frame shifted/rotated by robot_start.
    """
    sx, sy, syaw = layout.robot_start
    x, y = xyyaw[0] - sx, xyyaw[1] - sy
    c, s = math.cos(-syaw), math.sin(-syaw)
    yaw = (xyyaw[2] if len(xyyaw) > 2 else 0.0) - syaw
    return (c * x - s * y, s * x + c * y, math.atan2(math.sin(yaw), math.cos(yaw)))


def map_to_world(layout: Layout, xyyaw: Sequence[float]) -> Tuple[float, float, float]:
    sx, sy, syaw = layout.robot_start
    c, s = math.cos(syaw), math.sin(syaw)
    x, y = xyyaw[0], xyyaw[1]
    yaw = (xyyaw[2] if len(xyyaw) > 2 else 0.0) + syaw
    return (c * x - s * y + sx, s * x + c * y + sy, math.atan2(math.sin(yaw), math.cos(yaw)))
