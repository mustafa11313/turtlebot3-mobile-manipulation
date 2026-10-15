import cv2
import numpy as np

from fetch_deliver_perception.color_detector import ColorShapeDetector, label_matches
from fetch_deliver_perception.tracker import ObjectTracker


def synthetic_image():
    img = np.full((480, 640, 3), 160, np.uint8)
    cv2.rectangle(img, (100, 300), (140, 340), (0, 0, 220), -1)           # red square (cube)
    cv2.ellipse(img, (400, 300), (20, 7), 0, 0, 360, (0, 0, 220), -1)     # tall red cylinder
    cv2.rectangle(img, (380, 300), (420, 370), (0, 0, 220), -1)
    cv2.ellipse(img, (400, 370), (20, 7), 0, 0, 360, (0, 0, 220), -1)
    cv2.rectangle(img, (250, 200), (262, 212), (220, 40, 10), -1)         # small blue blob
    return img


def test_detects_colors_and_shapes():
    dets = ColorShapeDetector().detect(synthetic_image())
    labels = sorted(d.label for d in dets)
    assert 'red_cube' in labels
    assert 'red_cylinder' in labels
    assert any(d.color == 'blue' for d in dets)


def test_label_matching():
    assert label_matches('red_cube', 'red_cube')
    assert label_matches('red_unknown', 'red_cube')
    assert not label_matches('red_cylinder', 'red_cube')
    assert not label_matches('blue_cube', 'red_cube')


def test_tracker_votes_and_targets():
    tr = ObjectTracker(min_hits=3)
    for i in range(3):
        tr.update([('red', 'unknown', 1.0 + 0.01 * i, 2.0, 0.0125, 2.5)], float(i))
    t = tr.best_target('red_cube', (0.0, 0.0))
    assert t is not None and not t.is_verified('red_cube')
    for i in range(3):
        tr.update([('red', 'cylinder', 1.0, 2.0, 0.0125, 0.4)], 10.0 + i)
    assert tr.best_target('red_cube', (0.0, 0.0)) is None  # it was a cylinder after all
    for i in range(3):
        tr.update([('red', 'cube', 3.0, 0.0, 0.0125, 0.4)], 20.0 + i)
    best = tr.best_target('red_cube', (0.0, 0.0))
    assert best is not None and best.is_verified('red_cube')
    tr.reject(best.track_id)
    assert tr.best_target('red_cube', (0.0, 0.0)) is None
