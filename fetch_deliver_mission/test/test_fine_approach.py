import math

import pytest

behaviours = pytest.importorskip('fetch_deliver_mission.behaviours')


@pytest.mark.parametrize('start', [(0.0, 0.0, 0.0), (0.1, -0.08, 0.3), (-0.05, 0.1, -0.4)])
def test_fine_approach_converges(start):
    point, reach = (0.4, 0.0), 0.19
    x, y, th = start
    dt = 0.1
    for _ in range(600):
        v, w, done = behaviours.fine_approach_command((x, y, th), point, reach)
        if done:
            break
        th += w * dt
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
    assert done
    xb = math.cos(th) * (point[0] - x) + math.sin(th) * (point[1] - y)
    yb = -math.sin(th) * (point[0] - x) + math.cos(th) * (point[1] - y)
    assert abs(math.hypot(xb, yb) - reach) < 0.02
    assert abs(math.atan2(yb, xb)) < 0.04
