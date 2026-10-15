import math

from fetch_deliver_eval.metrics import (CollisionCounter, DetectionScore, clearance, in_camera_view,
                                        line_of_sight)
from fetch_deliver_gazebo.layouts import Furniture, Layout


def box_layout():
    return Layout(name='t', robot_start=(0, 0, 0),
                  walls=[(-2, -2, 2, -2), (2, -2, 2, 2), (2, 2, -2, 2), (-2, 2, -2, -2)],
                  furniture=[Furniture('table', 1.0, 0.0, 0.0, 0.4, 0.4, 0.7)],
                  spawn_regions=[], dropoff_pose=(0, 0, 0), marker_id=7,
                  marker_pose=(0, 0, 0, 0), search_waypoints=[])


def test_clearance_and_collision_counter():
    lay = box_layout()
    assert clearance(lay, (0.0, 0.0, 0.0)) > 0.5
    assert clearance(lay, (0.62, 0.0, 0.0)) < 0.0       # nose inside the table
    cc = CollisionCounter()
    for x in (0.0, 0.3, 0.62, 0.62, 0.3, 0.62):
        cc.update(clearance(lay, (x, 0.0, 0.0)))
    assert cc.count == 2


def test_line_of_sight_blocked_by_tall_furniture():
    lay = box_layout()
    assert not line_of_sight(lay, (0.0, 0.0), (1.8, 0.0))
    assert line_of_sight(lay, (0.0, 0.0), (0.0, 1.5))


def test_camera_view():
    assert in_camera_view((0, 0, 0), (1.0, 0.1))
    assert not in_camera_view((0, 0, 0), (-1.0, 0.0))
    assert not in_camera_view((0, 0, math.pi / 2), (1.0, 0.0))


def test_detection_scoring():
    s = DetectionScore()
    s.score_frame([('red_cube', 1.0, 1.0), ('red_unknown', 2.0, 2.05), ('blue_cube', 5, 5)],
                  [('red_cube', 1.05, 1.0), ('red_cylinder', 2.0, 2.0), ('green_cube', 3, 3)])
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)
    assert math.isclose(s.precision, 2 / 3) and math.isclose(s.recall, 2 / 3)
