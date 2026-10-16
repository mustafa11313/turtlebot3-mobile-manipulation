import math

import numpy as np

from fetch_deliver_perception.geometry import (ground_contact_center, matrix_to_quat, pixel_to_plane,
                                               project_point, quat_to_matrix, transform_to_matrix)


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


def _cube_bbox(center, size, T):
    """Pixel bbox of an axis-aligned cube resting on the floor."""
    T_inv = np.linalg.inv(T)
    pts = []
    for dx in (-size / 2, size / 2):
        for dy in (-size / 2, size / 2):
            for z in (0.0, size):
                p = np.array([center[0] + dx, center[1] + dy, z])
                pts.append(project_point(T_inv[:3, :3] @ p + T_inv[:3, 3], K))
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    return min(us), min(vs), max(us) - min(us), max(vs) - min(vs)


def test_ground_contact_center_is_unbiased_at_range():
    T = optical_to_base(pitch=0.30, height=0.101, x=0.079)
    s = 0.025
    for c in [(0.3, 0.0), (1.0, 0.2), (2.0, -0.3), (3.0, 0.5)]:
        x, y, w, h = _cube_bbox(c, s, T)
        est = ground_contact_center(x + w / 2, y + h, K, T, s / 2, s / 2)
        # small residual from the cube's corners; well under the 15 % centroid bias
        assert math.hypot(est[0] - c[0], est[1] - c[1]) < 0.02 * math.hypot(*c) + 0.005


def test_center_from_height_matches_true_range():
    import numpy as np
    from fetch_deliver_perception.geometry import center_from_height
    K = np.array([[554.0, 0, 320], [0, 554.0, 240], [0, 0, 1]])
    # camera looking along +x of the target frame, 0.1 m up (optical z forward, y down)
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    T[:3, 3] = (0.0, 0.0, 0.10)
    x, h, r = 1.5, 0.10, 0.015
    top = 240 + 554.0 * (0.10 - 0.17) / (x - r)
    bottom = 240 + 554.0 * (0.10 - 0.07) / (x - r)
    est = center_from_height(320, top, bottom, K, T, h, r, 0.12)
    assert abs(est[0] - x) < 0.01 and abs(est[1]) < 1e-6 and est[2] == 0.12
