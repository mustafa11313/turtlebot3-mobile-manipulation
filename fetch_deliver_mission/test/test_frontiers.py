import numpy as np

from fetch_deliver_mission.frontiers import best_frontier, find_frontiers, frontier_mask


def make_grid():
    g = np.full((60, 60), -1, np.int8)
    g[10:50, 10:30] = 0          # explored free room
    g[10:50, 10] = 100           # walls
    g[10, 10:30] = 100
    g[49, 10:30] = 100
    return g


def test_frontier_mask_on_boundary():
    m = frontier_mask(make_grid())
    assert m[30, 29]                 # free cell next to unknown on the right
    assert not m[30, 20]             # interior


def test_find_frontiers_returns_reachable_goal():
    g = make_grid()
    fr = find_frontiers(g, 0.05, (0.0, 0.0), (1.0, 1.5), min_cells=5)
    assert fr, 'expected a frontier on the open side'
    f = best_frontier(fr)
    gx, gy = f.goal
    assert 0.5 < gx < 1.5 and 0.5 < gy < 2.5
    i, j = int(gy / 0.05), int(gx / 0.05)
    assert g[i, j] == 0


def test_blacklist_removes_frontier():
    g = make_grid()
    fr = find_frontiers(g, 0.05, (0.0, 0.0), (1.0, 1.5), min_cells=5)
    fr2 = find_frontiers(g, 0.05, (0.0, 0.0), (1.0, 1.5), min_cells=5,
                         blacklist=[fr[0].target])
    assert len(fr2) < len(fr)


def test_fully_explored_has_no_frontiers():
    g = np.zeros((20, 20), np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    assert find_frontiers(g, 0.05, (0, 0), (0.5, 0.5)) == []
