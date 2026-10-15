"""Multi-object tracker in a fixed frame (map/odom) with label voting.

Objects are static, so a track is a running average of its observed position.
Labels are voted: far-away detections only give a color ("red_unknown"),
close-up ones also give a shape, and the shape votes override unknowns.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    z: float
    color: str
    hits: int = 1
    last_seen: float = 0.0
    first_seen: float = 0.0
    shape_votes: Counter = field(default_factory=Counter)
    min_range: float = float('inf')
    rejected: bool = False   # e.g. a pick on this track failed as "not the target"

    @property
    def shape(self) -> str:
        known = {k: v for k, v in self.shape_votes.items() if k != 'unknown'}
        if not known:
            return 'unknown'
        shape, votes = max(known.items(), key=lambda kv: kv[1])
        return shape if votes >= 2 else 'unknown'

    @property
    def label(self) -> str:
        return f'{self.color}_{self.shape}'

    def matches_target(self, target: str) -> bool:
        tcolor, _, tshape = target.partition('_')
        return not self.rejected and self.color == tcolor and self.shape in (tshape, 'unknown')

    def is_verified(self, target: str) -> bool:
        return self.matches_target(target) and self.shape == target.partition('_')[2]


class ObjectTracker:

    def __init__(self, gate: float = 0.30, min_hits: int = 3, alpha: float = 0.3):
        self.gate = gate
        self.min_hits = min_hits
        self.alpha = alpha
        self.tracks: Dict[int, Track] = {}
        self._next = 1

    def update(self, observations: List[Tuple[str, str, float, float, float, float]],
               stamp: float) -> List[Track]:
        """observations: (color, shape, x, y, z, range). Returns the updated tracks."""
        updated = []
        for color, shape, x, y, z, rng in observations:
            best: Optional[Track] = None
            best_d = self.gate
            for t in self.tracks.values():
                if t.color != color:
                    continue
                d = math.hypot(t.x - x, t.y - y)
                # Range-dependent gate: ground-plane estimates get worse with distance.
                if d < max(best_d, 0.1 * rng) and (best is None or d < best_d):
                    best, best_d = t, d
            if best is None:
                best = Track(self._next, x, y, z, color, last_seen=stamp, first_seen=stamp)
                self.tracks[self._next] = best
                self._next += 1
            else:
                # Closer observations are more accurate: weight them more.
                a = self.alpha if rng >= best.min_range else max(self.alpha, 0.6)
                best.x += a * (x - best.x)
                best.y += a * (y - best.y)
                best.z += a * (z - best.z)
                best.hits += 1
                best.last_seen = stamp
            best.min_range = min(best.min_range, rng)
            best.shape_votes[shape] += 2 if rng < 0.8 else 1
            updated.append(best)
        return updated

    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]

    def best_target(self, target: str, robot_xy: Tuple[float, float]) -> Optional[Track]:
        cands = [t for t in self.confirmed() if t.matches_target(target)]
        if not cands:
            return None
        # Prefer verified tracks, then the closest one.
        return min(cands, key=lambda t: (not t.is_verified(target),
                                         math.hypot(t.x - robot_xy[0], t.y - robot_xy[1])))

    def reject(self, track_id: int) -> None:
        if track_id in self.tracks:
            self.tracks[track_id].rejected = True

    def remove_near(self, x: float, y: float, radius: float = 0.3) -> None:
        for tid in [t.track_id for t in self.tracks.values()
                    if math.hypot(t.x - x, t.y - y) < radius]:
            del self.tracks[tid]
