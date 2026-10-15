"""Closed-form kinematics of the OpenMANIPULATOR-X mounted on a TurtleBot3 Waffle Pi.

All positions are expressed in ``base_footprint``. The arm is a yaw joint
followed by a planar 3-link chain (shoulder, elbow, wrist pitch), so a grasp
is fully described by the end-effector position and its pitch ``phi``
(``phi > 0`` points the gripper downwards). Being 4-DOF, the arm cannot reach
arbitrary orientations; the solver therefore searches over the pitch to find
the most top-down grasp that is reachable.

Geometry comes from the URDF (turtlebot3_manipulation_description):
  base_footprint -> base_link        z +0.010
  base_link      -> link1            (-0.092, 0, 0.091)
  link1 -> joint1 (yaw)              ( 0.012, 0, 0.017)
  joint1 -> joint2 (pitch)           ( 0,     0, 0.0595)
  joint2 -> joint3 (pitch)           ( 0.024, 0, 0.128)
  joint3 -> joint4 (pitch)           ( 0.124, 0, 0)
  joint4 -> end_effector_link        ( 0.126, 0, 0)
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

J1_XYZ = (-0.080, 0.0, 0.118)          # joint1 axis origin in base_footprint
SHOULDER_Z = J1_XYZ[2] + 0.0595         # joint2 height
L1_VEC = (0.024, 0.128)                 # joint2 -> joint3 (x, z) at zero angle
L1 = math.hypot(*L1_VEC)
ALPHA1 = math.atan2(L1_VEC[1], L1_VEC[0])
L2 = 0.124                              # joint3 -> joint4
L3 = 0.126                              # joint4 -> end effector

LIMITS = (
    (-math.pi * 0.9, math.pi * 0.9),
    (-math.pi * 0.57, math.pi * 0.5),
    (-math.pi * 0.3, math.pi * 0.44),
    (-math.pi * 0.57, math.pi * 0.65),
)
JOINT_NAMES = ('joint1', 'joint2', 'joint3', 'joint4')

GRIPPER_OPEN = 0.019
GRIPPER_CLOSED = -0.010


@dataclass
class ArmSolution:
    joints: Tuple[float, float, float, float]
    pitch: float

    def as_list(self) -> List[float]:
        return list(self.joints)


def _rot_plane(angle: float, a: float, c: float) -> Tuple[float, float]:
    """Rotate an (x, z) vector by a rotation of ``angle`` about +y."""
    return (a * math.cos(angle) + c * math.sin(angle), -a * math.sin(angle) + c * math.cos(angle))


def forward(q: Sequence[float]) -> Tuple[Tuple[float, float, float], float, float]:
    """Return (ee_xyz, pitch, yaw) of end_effector_link in base_footprint."""
    q1, q2, q3, q4 = q
    r1, z1 = _rot_plane(q2, *L1_VEC)
    r2, z2 = _rot_plane(q2 + q3, L2, 0.0)
    r3, z3 = _rot_plane(q2 + q3 + q4, L3, 0.0)
    r = r1 + r2 + r3
    z = SHOULDER_Z + z1 + z2 + z3
    x = J1_XYZ[0] + r * math.cos(q1)
    y = J1_XYZ[1] + r * math.sin(q1)
    return (x, y, z), q2 + q3 + q4, q1


def joint_positions(q: Sequence[float]) -> List[Tuple[float, float, float]]:
    """Positions of joint2, joint3, joint4 and the end effector (for clearance checks)."""
    q1, q2, q3, q4 = q
    pts = []
    r, z = 0.0, SHOULDER_Z
    pts.append((r, z))
    dr, dz = _rot_plane(q2, *L1_VEC)
    r, z = r + dr, z + dz
    pts.append((r, z))
    dr, dz = _rot_plane(q2 + q3, L2, 0.0)
    r, z = r + dr, z + dz
    pts.append((r, z))
    dr, dz = _rot_plane(q2 + q3 + q4, L3, 0.0)
    r, z = r + dr, z + dz
    pts.append((r, z))
    return [(J1_XYZ[0] + pr * math.cos(q1), J1_XYZ[1] + pr * math.sin(q1), pz) for pr, pz in pts]


def within_limits(q: Sequence[float], margin: float = 0.02) -> bool:
    return all(lo + margin <= v <= hi - margin for v, (lo, hi) in zip(q, LIMITS))


def inverse(xyz: Sequence[float], pitch: float, elbow_up: bool = True) -> Optional[ArmSolution]:
    """Analytic IK for an end-effector position and pitch. None if unreachable."""
    x, y, z = xyz
    dx, dy = x - J1_XYZ[0], y - J1_XYZ[1]
    q1 = math.atan2(dy, dx)
    r = math.hypot(dx, dy)
    # wrist (joint4) position in the arm plane, relative to joint2
    wr = r - L3 * math.cos(pitch)
    wz = z + L3 * math.sin(pitch) - SHOULDER_Z
    d2 = wr * wr + wz * wz
    c = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1.0:
        return None
    best = None
    for sign in ((-1.0, 1.0) if elbow_up else (1.0, -1.0)):
        gamma = sign * math.acos(c)          # planar angle of L2 relative to L1
        theta1 = math.atan2(wz, wr) - math.atan2(L2 * math.sin(gamma), L1 + L2 * math.cos(gamma))
        q2 = ALPHA1 - theta1
        q3 = -gamma - ALPHA1
        q4 = pitch - q2 - q3
        q = tuple(math.atan2(math.sin(v), math.cos(v)) for v in (q1, q2, q3, q4))
        if within_limits(q):
            best = ArmSolution(q, pitch)
            break
    return best


def min_clearance_ok(q: Sequence[float], floor_z: float = 0.0, base_front_x: float = 0.12,
                     base_top_z: float = 0.135) -> bool:
    """Coarse self/floor clearance: wrist above the floor and not inside the base body."""
    pts = joint_positions(q)
    j3, j4 = pts[1], pts[2]
    for px, py, pz in (j3, j4):
        if pz < floor_z + 0.03:
            return False
        if px < base_front_x and abs(py) < 0.15 and pz < base_top_z + 0.03:
            return False
    return True


def solve_grasp(xyz: Sequence[float], pitches: Optional[Iterable[float]] = None,
                floor_z: float = 0.0) -> Optional[ArmSolution]:
    """Find the most top-down reachable grasp for an object center ``xyz``."""
    if pitches is None:
        pitches = [math.radians(a) for a in range(85, -1, -5)]
    for phi in pitches:
        sol = inverse(xyz, phi)
        if sol is not None and min_clearance_ok(sol.joints, floor_z):
            return sol
    return None


def approach_point(xyz: Sequence[float], pitch: float, yaw: float, back: float,
                   up: float = 0.0) -> Tuple[float, float, float]:
    """A point ``back`` metres behind ``xyz`` along the approach direction, plus ``up``."""
    x, y, z = xyz
    return (x - back * math.cos(pitch) * math.cos(yaw),
            y - back * math.cos(pitch) * math.sin(yaw),
            z + back * math.sin(pitch) + up)


def reachable_band(z: float, pitch_deg: Iterable[int] = range(85, -1, -5)) -> Tuple[float, float]:
    """Range of forward distances x (y = 0) at which an object at height z is graspable."""
    xs = [x / 1000.0 for x in range(0, 500, 2)]
    ok = [x for x in xs if solve_grasp((x, 0.0, z), [math.radians(a) for a in pitch_deg])]
    return (min(ok), max(ok)) if ok else (float('nan'), float('nan'))


def yaw_to_quaternion_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)
