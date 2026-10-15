import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fetch_deliver_gazebo.layouts import load_layouts, map_to_world, world_to_map  # noqa: E402
from fetch_deliver_gazebo.scenario import DIFFICULTIES, Scenario, generate  # noqa: E402
from fetch_deliver_gazebo.sdf_gen import world_sdf  # noqa: E402

LAYOUTS = load_layouts(os.path.join(os.path.dirname(HERE), 'config', 'layouts.yaml'))


@pytest.mark.parametrize('name', sorted(LAYOUTS))
@pytest.mark.parametrize('difficulty', DIFFICULTIES)
def test_scenarios_are_valid_and_deterministic(name, difficulty):
    layout = LAYOUTS[name]
    for seed in range(10):
        a = generate(layout, seed, difficulty)
        b = generate(layout, seed, difficulty)
        assert a.to_json() == b.to_json()
        assert sum(o.is_target for o in a.objects) == 1
        for o in a.objects:
            assert layout.distance_to_walls(o.x, o.y) >= 0.29
            assert not any(f.footprint_contains(o.x, o.y, 0.29) for f in layout.furniture)
        for i, o in enumerate(a.objects):
            for p in a.objects[i + 1:]:
                assert math.hypot(o.x - p.x, o.y - p.y) > 0.5
        if difficulty in ('distractors', 'hard'):
            assert len(a.objects) >= 2


def test_json_roundtrip():
    sc = generate(LAYOUTS['apartment'], 3, 'hard')
    assert Scenario.from_json(sc.to_json()).to_json() == sc.to_json()


@pytest.mark.parametrize('name', sorted(LAYOUTS))
def test_waypoints_and_dropoff_are_free(name):
    layout = LAYOUTS[name]
    for x, y, _ in layout.search_waypoints + [layout.robot_start]:
        assert layout.is_free(x, y, 0.25), (x, y)
    assert layout.is_free(*layout.dropoff_pose[:2], 0.2)


def test_map_world_roundtrip():
    layout = LAYOUTS['apartment']
    for pose in [(0.0, 0.0, 0.0), (2.0, -1.0, 1.0), (-4.0, 3.0, -2.5)]:
        back = map_to_world(layout, world_to_map(layout, pose))
        assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(pose, back))


@pytest.mark.parametrize('name', sorted(LAYOUTS))
def test_committed_worlds_are_up_to_date(name):
    path = os.path.join(os.path.dirname(HERE), 'worlds', f'{name}.world')
    with open(path) as f:
        assert f.read() == world_sdf(LAYOUTS[name]), \
            'worlds are stale: run scripts/generate_worlds.py'


@pytest.mark.parametrize('name', sorted(LAYOUTS))
def test_dynamic_obstacle_paths_stay_in_free_space(name):
    lay = LAYOUTS[name]
    for seed in range(10):
        sc = generate(lay, seed, 'hard')
        assert sc.obstacles
        for obs in sc.obstacles:
            (ax, ay), (bx, by) = obs.waypoints
            for i in range(21):
                t = i / 20
                x, y = ax + t * (bx - ax), ay + t * (by - ay)
                assert lay.is_free(x, y, 0.2)
                assert math.hypot(x - lay.robot_start[0], y - lay.robot_start[1]) > 0.8
