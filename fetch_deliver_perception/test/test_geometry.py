import math

import numpy as np

from fetch_deliver_perception.geometry import (matrix_to_quat, pixel_to_plane, project_point,
                                               quat_to_matrix, transform_to_matrix)


def optical_to_base(pitch=0.4, height=0.116, x=0.081):
    """Camera looking forward (+x of base) and pitched down by ``pitch``."""
    # optical frame: z forward, x right, y down
    R_link_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    c, s = math.cos(pitch), math.sin(pitch)
    R_pitch = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T = np.eye(4)
    T[:3, :3] = R_pitch @ R_link_opt
    T[:3, 3] = (x, 0.0, height)
    return T


K = np.array([[554.0, 0, 320.5], [0, 554.0, 240.5], [0, 0, 1]])


def test_quaternion_roundtrip():
    q = (0.1, -0.3, 0.2, 0.9)
    n = math.sqrt(sum(v * v for v in q))
    q = tuple(v / n for v in q)
    q2 = matrix_to_quat(quat_to_matrix(q))
    assert min(np.allclose(q, q2, atol=1e-9), True) or np.allclose(q, [-v for v in q2])


def test_pixel_to_plane_inverts_projection():
    T = optical_to_base()
    T_inv = np.linalg.inv(T)
    for p in [(0.2, 0.0, 0.0125), (1.5, 0.4, 0.0125), (3.0, -1.0, 0.0125)]:
        pc = T_inv[:3, :3] @ np.array(p) + T_inv[:3, 3]
        uv = project_point(pc, K)
        assert uv is not None
        est = pixel_to_plane(uv[0], uv[1], K, T, 0.0125)
        assert np.allclose(est, p, atol=1e-6)


def test_ray_above_horizon_misses_floor():
    T = optical_to_base(pitch=0.0)
    assert pixel_to_plane(320.5, 100.0, K, T, 0.0) is None


def test_transform_matrix_identity():
    T = transform_to_matrix((1, 2, 3), (0, 0, 0, 1))
    assert np.allclose(T[:3, 3], (1, 2, 3)) and np.allclose(T[:3, :3], np.eye(3))
