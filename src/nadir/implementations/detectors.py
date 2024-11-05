import logging
from typing import List

import numpy as np
from ultralytics import YOLO

from ..interfaces import IObjectDetector
from ..models import BoundingBox, Detection


class YOLODetector(IObjectDetector):
    def __init__(self, model_path: str):
        self.logger = logging.getLogger(__name__)
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            self.logger.error(f"YOLO 모델 로드 실패: {str(e)}")
            raise

    def detect_and_track(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
    ) -> List[Detection]:
        try:
            results = self.model.track(frame, persist=True)
            detections = []

            if results[0].boxes.id is not None:
                for box, track_id, cls, conf in zip(
                    results[0].boxes.xywh.cpu(),
                    results[0].boxes.id.int().cpu().tolist(),
                    results[0].boxes.cls.int().cpu().tolist(),
                    results[0].boxes.conf.cpu().tolist(),
                ):
                    if conf < confidence_threshold:
                        continue

                    x, y, w, h = box.tolist()
                    class_name = self.model.names[cls]

                    detection = Detection(
                        track_id=track_id,
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=BoundingBox(
                            x_center=float(x),
                            y_center=float(y),
                            width=float(w),
                            height=float(h),
                        ),
                        geo_location=None,  # Will be set by processor
                    )
                    detections.append(detection)

            return detections
        except Exception as e:
            self.logger.error(f"객체 감지 실패: {str(e)}")
            raise
