import math
import random

import pytest

from fetch_deliver_manipulation import kinematics as kin


def test_fk_ik_roundtrip():
    rng = random.Random(0)
    checked = 0
    for _ in range(2000):
        q = [rng.uniform(lo + 0.05, hi - 0.05) for lo, hi in kin.LIMITS]
        xyz, pitch, _ = kin.forward(q)
        sol = kin.inverse(xyz, pitch)
        if sol is None:
            continue
        xyz2, pitch2, _ = kin.forward(sol.joints)
        assert math.dist(xyz, xyz2) < 1e-9
        assert abs(math.remainder(pitch - pitch2, 2 * math.pi)) < 1e-9
        checked += 1
    assert checked > 500


def test_floor_band_covers_grasp_standoff():
    lo, hi = kin.reachable_band(0.0125)
    assert lo < 0.17 and hi > 0.23


@pytest.mark.parametrize('y', [-0.05, 0.0, 0.05])
def test_solve_grasp_prefers_top_down(y):
    sol = kin.solve_grasp((0.20, y, 0.0125))
    assert sol is not None
    assert sol.pitch > math.radians(30)
    xyz, _, _ = kin.forward(sol.joints)
    assert math.dist(xyz, (0.20, y, 0.0125)) < 1e-9


def test_unreachable_returns_none():
    assert kin.solve_grasp((1.0, 0.0, 0.0)) is None


def test_approach_point_is_behind_and_above():
    p = kin.approach_point((0.2, 0.0, 0.0), math.radians(60), 0.0, 0.05)
    assert p[0] < 0.2 and p[2] > 0.0
