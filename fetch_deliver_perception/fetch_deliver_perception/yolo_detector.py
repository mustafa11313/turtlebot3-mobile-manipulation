"""Optional YOLO backend (Ultralytics). Same output type as ColorShapeDetector.

Train a model on simulator-generated data with tools/generate_yolo_dataset.py
and tools/train_yolo.py; class names must be "<color>_<shape>" (e.g. red_cube).
"""

from typing import List

import numpy as np

from .color_detector import Detection


class YoloDetector:

    def __init__(self, weights: str, confidence: float = 0.5):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError('backend:=yolo needs `pip install ultralytics`') from e
        if not weights:
            raise RuntimeError('backend:=yolo needs yolo_weights:=/path/to/best.pt')
        self.model = YOLO(weights)
        self.confidence = confidence

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        res = self.model.predict(bgr, conf=self.confidence, verbose=False)[0]
        out = []
        for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            label = res.names[int(cls)]
            color, _, shape = label.partition('_')
            out.append(Detection(label, color, shape or 'unknown', float(conf),
                                 (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                                 ((x1 + x2) / 2, (y1 + y2) / 2), float((x2 - x1) * (y2 - y1))))
        return out
