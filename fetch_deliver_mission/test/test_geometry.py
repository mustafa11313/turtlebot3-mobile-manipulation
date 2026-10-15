import math

import numpy as np

from fetch_deliver_mission.geometry import CostmapView, pick_standoff, standoff_candidates


def test_first_candidate_faces_target_from_robot_side():
    c = standoff_candidates((1.0, 0.0), (0.0, 0.0), 0.5)[0]
    assert math.isclose(c[0], 0.5, abs_tol=1e-9) and math.isclose(c[1], 0.0, abs_tol=1e-9)
    assert math.isclose(abs(c[2]), 0.0, abs_tol=1e-9)


def test_pick_standoff_avoids_obstacles():
    data = np.zeros((100, 100), np.int16)
    data[:, :50] = 100   # everything with x < 2.5 m is occupied
    cm = CostmapView(data, 0.05, (0.0, 0.0))
    pose = pick_standoff((3.0, 2.5), (0.0, 2.5), 0.5, cm)
    assert pose is not None and pose[0] > 2.7
