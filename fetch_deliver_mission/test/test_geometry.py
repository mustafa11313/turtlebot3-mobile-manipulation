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


def test_pick_standoff_checks_the_grasp_pose():
    # A thin wall 0.25 m south of the target: the approach pose straight from the
    # robot (south, 0.6 m) is beyond the wall and free, but servoing in to 0.2 m
    # would put the footprint on the wall, so another side must be chosen.
    data = np.zeros((100, 100), np.int16)
    data[44:46, 30:70] = 100              # wall at y = 2.2..2.3 m, x = 1.5..3.5 m
    cm = CostmapView(data, 0.05, (0.0, 0.0))
    target, robot = (2.5, 2.5), (2.5, 0.5)
    naive = pick_standoff(target, robot, 0.6, cm)
    assert naive is not None and naive[1] < 2.2
    pose = pick_standoff(target, robot, 0.6, cm, grasp_distance=0.2)
    assert pose is not None and pose[1] > 2.3
