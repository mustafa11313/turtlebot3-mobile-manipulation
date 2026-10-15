"""Color and shape based detector for the small graspable objects.

The detector is deliberately classical (HSV thresholds + contour analysis) so
it runs anywhere without a GPU; a YOLO backend with the same output format is
in ``yolo_detector.py``.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# HSV ranges (OpenCV: H in [0, 180)). Red wraps around 0.
DEFAULT_COLORS: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    'red': [((0, 120, 60), (8, 255, 255)), ((172, 120, 60), (180, 255, 255))],
    'green': [((45, 120, 50), (80, 255, 255))],
    'blue': [((105, 150, 50), (125, 255, 255))],
    'yellow': [((22, 120, 80), (35, 255, 255))],
}


@dataclass
class Detection:
    label: str           # "<color>_<shape>", shape may be "unknown" when too small
    color: str
    shape: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    center: Tuple[float, float]       # pixel centroid
    area: float

    @property
    def width(self) -> int:
        return self.bbox[2]


def classify_shape(contour: np.ndarray, min_size_px: int) -> Tuple[str, float]:
    """Return (shape, confidence): "cube", "cylinder" or "unknown" (too small to tell).

    Seen from the robot's camera, a cube's silhouette is roughly as tall as it
    is wide (aspect 0.8-1.4 depending on yaw and viewing angle), while the
    distractor cylinder stands twice as tall as its diameter (aspect >= 1.7).
    """
    x, y, w, h = cv2.boundingRect(contour)
    if max(w, h) < min_size_px:
        return 'unknown', 0.5
    area = cv2.contourArea(contour)
    solidity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1e-6)
    if solidity < 0.8:
        return 'unknown', 0.3
    aspect = h / float(max(w, 1))
    if aspect < 1.45:
        return 'cube', float(np.clip(1.0 - abs(aspect - 1.0), 0.6, 1.0))
    if aspect > 1.65:
        return 'cylinder', float(np.clip(0.5 + (aspect - 1.65), 0.6, 1.0))
    return 'unknown', 0.4


class ColorShapeDetector:

    def __init__(self, colors: Optional[Dict] = None, min_area: float = 12.0,
                 min_shape_px: int = 14, max_aspect: float = 3.2):
        self.colors = colors or DEFAULT_COLORS
        self.min_area = min_area
        self.min_shape_px = min_shape_px
        self.max_aspect = max_aspect
        self.kernel = np.ones((3, 3), np.uint8)

    def mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self.colors[color]:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (3, 3), 0), cv2.COLOR_BGR2HSV)
        out: List[Detection] = []
        for color in self.colors:
            contours, _ = cv2.findContours(self.mask(hsv, color), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if max(w, h) / max(1, min(w, h)) > self.max_aspect:
                    continue
                m = cv2.moments(c)
                cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
                shape, shape_conf = classify_shape(c, self.min_shape_px)
                fill = area / float(w * h)
                conf = float(np.clip(0.5 * shape_conf + 0.5 * fill, 0.0, 1.0))
                out.append(Detection(f'{color}_{shape}', color, shape, conf,
                                     (x, y, w, h), (cx, cy), area))
        return out


def label_matches(det_label: str, target: str) -> bool:
    """A detection is a target candidate if the color matches and the shape is not contradicting."""
    color, _, shape = det_label.partition('_')
    tcolor, _, tshape = target.partition('_')
    return color == tcolor and shape in (tshape, 'unknown')


def draw(bgr: np.ndarray, detections: Sequence[Detection], target: str) -> np.ndarray:
    img = bgr.copy()
    for d in detections:
        x, y, w, h = d.bbox
        col = (0, 255, 0) if label_matches(d.label, target) else (0, 165, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)
        cv2.putText(img, f'{d.label} {d.confidence:.2f}', (x, max(10, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return img
