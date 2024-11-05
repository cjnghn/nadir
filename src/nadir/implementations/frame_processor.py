# src/nadir/implementations/frame_processor.py
import logging
from typing import Dict, List, Optional
import numpy as np

from ..interfaces import IFrameProcessor, IObjectDetector, IGeoCalculator
from ..models import DroneFrame, VideoMetadata, Detection, BoundingBox, GeoPoint


class DefaultFrameProcessor(IFrameProcessor):
    def __init__(
        self,
        detector: IObjectDetector,
        geo_calculator: IGeoCalculator,
        confidence_threshold: float,
    ):
        self.detector = detector
        self.geo_calculator = geo_calculator
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_data: DroneFrame,
        metadata: VideoMetadata,
        tracks: Dict,
    ) -> Optional[Dict]:
        try:
            detections = self.detector.detect_and_track(
                frame, self.confidence_threshold
            )
            result_detections = []

            for detection in detections:
                geo_location_dict = self.geo_calculator.calculate_geo_location(
                    pixel_x=detection.bbox.x_center,
                    pixel_y=detection.bbox.y_center,
                    frame_width=metadata.width,
                    frame_height=metadata.height,
                    drone_lat=frame_data.drone_state.position.latitude,
                    drone_lon=frame_data.drone_state.position.longitude,
                    drone_alt_feet=frame_data.drone_state.altitude_feet,
                    heading_degrees=frame_data.drone_state.heading_degrees,
                )

                # 새로운 Detection 객체 생성
                detection_with_geo = Detection(
                    track_id=detection.track_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    geo_location=GeoPoint(**geo_location_dict),
                )
                result_detections.append(detection_with_geo)

                self._update_tracks(tracks, detection_with_geo, frame_data)

            return self._format_frame_result(frame_data, result_detections)

        except Exception as e:
            self.logger.error(f"프레임 처리 실패: {str(e)}")
            return None

    def _update_tracks(
        self, tracks: Dict, detection: Detection, frame_data: DroneFrame
    ) -> None:
        """트랙 정보 업데이트"""
        if detection.track_id not in tracks:
            tracks[detection.track_id] = {
                "track_id": detection.track_id,
                "class": detection.class_name,
                "frame_history": [],
                "trajectory": {"pixel_coordinates": [], "geo_coordinates": []},
            }

        # Pydantic 모델의 model_dump() 사용
        tracks[detection.track_id]["frame_history"].append(
            {
                "frame_number": frame_data.number,
                "bbox": detection.bbox.model_dump(),
                "geo_location": detection.geo_location.model_dump()
                if detection.geo_location
                else None,
            }
        )

        tracks[detection.track_id]["trajectory"]["pixel_coordinates"].append(
            [detection.bbox.x_center, detection.bbox.y_center]
        )
        if detection.geo_location:
            tracks[detection.track_id]["trajectory"]["geo_coordinates"].append(
                [detection.geo_location.latitude, detection.geo_location.longitude]
            )

    def _format_frame_result(
        self, frame_data: DroneFrame, detections: List[Detection]
    ) -> Dict:
        """프레임 결과를 딕셔너리로 포맷"""
        return {
            "frame_number": frame_data.number,
            "timestamp": frame_data.timestamp.model_dump(),
            "drone_data": {
                "position": frame_data.drone_state.position.model_dump(),
                "altitude_feet": frame_data.drone_state.altitude_feet,
                "compass_heading": frame_data.drone_state.heading_degrees,
            },
            "detections": [
                {
                    "track_id": d.track_id,
                    "class": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox.model_dump(),
                    "geo_location": d.geo_location.model_dump()
                    if d.geo_location
                    else None,
                }
                for d in detections
            ],
        }
