"""ROS helpers shared by the perception nodes."""

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time

from .geometry import transform_to_matrix


def lookup_matrix(tf_buffer, target: str, source: str, stamp=None, timeout: float = 0.05):
    """4x4 transform mapping points in ``source`` to ``target`` (None if unavailable)."""
    for t in ((stamp,) if stamp is not None else ()) + (Time(),):
        try:
            tf = tf_buffer.lookup_transform(target, source, t, timeout=Duration(seconds=timeout))
            tr, q = tf.transform.translation, tf.transform.rotation
            return transform_to_matrix((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))
        except Exception:  # noqa: BLE001  (try the latest transform next)
            continue
    return None


def apply(T: np.ndarray, p) -> np.ndarray:
    return T[:3, :3] @ np.asarray(p, float) + T[:3, 3]
