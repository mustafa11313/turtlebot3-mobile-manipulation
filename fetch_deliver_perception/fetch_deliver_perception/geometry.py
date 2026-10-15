"""Small geometry helpers (no ROS dependency, unit-tested)."""

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix from quaternion (x, y, z, w)."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def transform_to_matrix(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(rotation)
    T[:3, 3] = translation
    return T


def pixel_to_plane(u: float, v: float, K: np.ndarray, T_target_cam: np.ndarray,
                   plane_z: float) -> Optional[np.ndarray]:
    """Intersect the camera ray through pixel (u, v) with the plane z = plane_z.

    ``T_target_cam`` maps camera-optical-frame points into the target frame
    (e.g. base_footprint, whose z = 0 is the floor). Returns the 3D point in the
    target frame, or None if the ray does not hit the plane in front of the camera.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    R, o = T_target_cam[:3, :3], T_target_cam[:3, 3]
    d = R @ d_cam
    if abs(d[2]) < 1e-9:
        return None
    s = (plane_z - o[2]) / d[2]
    if s <= 0:
        return None
    return o + s * d


def project_point(p_cam: Sequence[float], K: np.ndarray) -> Optional[Tuple[float, float]]:
    x, y, z = p_cam
    if z <= 1e-6:
        return None
    return (K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2])


def yaw_from_quat(q: Sequence[float]) -> float:
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
