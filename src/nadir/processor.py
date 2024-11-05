# src/nadir/processor.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from ultralytics import YOLO

from .geo_calculator import GeographicCalculator
from .log_mapper import DroneLogMapper
from .models import (
    BoundingBox,
    Detection,
    DroneFrame,
    GeoPoint,
    Timestamp,
    VideoMetadata,
    VideoSegment,
)
from .trajectory_analyzer import TrajectoryAnalyzer, TrajectoryMetrics


class DroneVideoProcessor:
    """드론 비디오를 객체 감지 및 지리적 매핑으로 처리합니다."""

    def __init__(
        self,
        model_path: str,
        camera_fov_degrees: float,
        confidence_threshold: float = 0.5,
        dangerous_psm_threshold: float = 1.5,
        min_intersection_angle: float = 15.0,
    ):
        """
        비디오 프로세서를 초기화합니다.

        Args:
            model_path: YOLO 모델 경로
            camera_fov_degrees: 카메라 시야각 (도)
            confidence_threshold: 최소 객체 감지 신뢰도
            dangerous_psm_threshold: 위험한 상호작용 PSM 임계값 (초)
            min_intersection_angle: 최소 교차각 (도)
        """
        self.model = YOLO(model_path)
        self.calculator = GeographicCalculator(camera_fov_degrees)
        self.confidence_threshold = confidence_threshold
        self.trajectory_analyzer = TrajectoryAnalyzer(
            dangerous_psm_threshold=dangerous_psm_threshold,
            min_intersection_angle=min_intersection_angle,
        )
        self.logger = logging.getLogger(__name__)

    def process_videos(self, video_paths: List[str], log_path: str) -> Dict[str, Dict]:
        """동기화된 드론 데이터와 함께 여러 비디오를 처리합니다."""
        try:
            mapper = DroneLogMapper(log_path)
            mapped_segments = mapper.map_videos_to_segments(video_paths)

            results = {}
            for video_path, segment in mapped_segments.items():
                results[Path(video_path).name] = self._process_single_video(
                    video_path, segment
                )

            return results
        except Exception as e:
            self.logger.error(f"비디오 처리 실패: {str(e)}")
            raise

    def _extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """비디오 파일에서 메타데이터를 추출합니다."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오 열기 실패: {video_path}")

            # 비디오 속성 가져오기
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 지속 시간 계산
            duration_seconds = frame_count / fps if fps > 0 else 0

            # 생성 시간 가져오기
            creation_time = int(
                Path(video_path).stat().st_mtime * 1000
            )  # 밀리초로 변환

            cap.release()

            return VideoMetadata(
                total_frames=frame_count,
                duration_seconds=duration_seconds,
                timestamp=Timestamp(
                    epoch_ms=creation_time,
                    relative_ms=0,
                    datetime_utc=Path(video_path).stat().st_mtime,
                ),
                fps=fps,
                width=width,
                height=height,
            )
        except Exception as e:
            self.logger.error(f"{video_path}에서 메타데이터 추출 실패: {str(e)}")
            raise

    def _process_single_video(self, video_path: str, segment: "VideoSegment") -> Dict:
        """단일 비디오를 객체 감지 및 추적로 처리합니다."""
        try:
            metadata = self._extract_video_metadata(video_path)
            tracks = {}
            processed_frames = []

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오 열기 실패: {video_path}")

            for frame_data in segment.frames:
                processed_frame = self._process_frame(cap, frame_data, metadata, tracks)
                if processed_frame:
                    processed_frames.append(processed_frame)

            cap.release()

            # 궤적 분석
            trajectory_metrics = self.trajectory_analyzer.analyze_trajectories(
                tracks, metadata.fps
            )

            return {
                "metadata": self._format_metadata_dict(metadata),
                "frames": processed_frames,
                "tracks": list(tracks.values()),
                "trajectory_analysis": self.trajectory_analyzer.format_metrics_dict(
                    trajectory_metrics
                ),
                "summary": self._generate_summary(tracks, trajectory_metrics),
            }
        except Exception as e:
            self.logger.error(f"{video_path} 처리 실패: {str(e)}")
            raise

    def _process_frame(
        self,
        cap: cv2.VideoCapture,
        frame_data: "DroneFrame",
        metadata: VideoMetadata,
        tracks: Dict,
    ) -> Optional[Dict]:
        """단일 프레임을 객체 감지 및 추적로 처리합니다."""
        success, frame = cap.read()
        if not success:
            return None

        # YOLO 감지 및 추적 실행
        results = self.model.track(frame, persist=True, show=True)

        detections = []
        if results[0].boxes.id is not None:
            for box, track_id, cls, conf in zip(
                results[0].boxes.xywh.cpu(),
                results[0].boxes.id.int().cpu().tolist(),
                results[0].boxes.cls.int().cpu().tolist(),
                results[0].boxes.conf.cpu().tolist(),
            ):
                if conf < self.confidence_threshold:
                    continue

                x, y, w, h = box.tolist()
                class_name = self.model.names[cls]

                # 지리적 위치 계산
                geo_location = self.calculator.calculate_geo_location(
                    pixel_x=x,
                    pixel_y=y,
                    frame_width=metadata.width,
                    frame_height=metadata.height,
                    drone_lat=frame_data.drone_state.position.latitude,
                    drone_lon=frame_data.drone_state.position.longitude,
                    drone_alt_feet=frame_data.drone_state.altitude_feet,
                    heading_degrees=frame_data.drone_state.heading_degrees,
                )

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
                    geo_location=GeoPoint(
                        latitude=geo_location["latitude"],
                        longitude=geo_location["longitude"],
                    ),
                )
                detections.append(self._format_detection_dict(detection))

                # 추적 정보 업데이트
                if track_id not in tracks:
                    tracks[track_id] = {
                        "track_id": track_id,
                        "class": class_name,
                        "frame_history": [],
                        "trajectory": {"pixel_coordinates": [], "geo_coordinates": []},
                    }

                # 추적 기록 업데이트
                tracks[track_id]["frame_history"].append(
                    {
                        "frame_number": frame_data.number,
                        "bbox": self._format_bbox_dict(detection.bbox),
                        "geo_location": {
                            "latitude": geo_location["latitude"],
                            "longitude": geo_location["longitude"],
                        },
                    }
                )

                # 궤적 업데이트
                tracks[track_id]["trajectory"]["pixel_coordinates"].append([x, y])
                tracks[track_id]["trajectory"]["geo_coordinates"].append(
                    [geo_location["latitude"], geo_location["longitude"]]
                )

        return {
            "frame_number": frame_data.number,
            "timestamp": self._format_timestamp_dict(frame_data.timestamp),
            "drone_data": {
                "latitude": frame_data.drone_state.position.latitude,
                "longitude": frame_data.drone_state.position.longitude,
                "altitude_feet": frame_data.drone_state.altitude_feet,
                "compass_heading": frame_data.drone_state.heading_degrees,
            },
            "detections": detections,
        }

    def _generate_summary(
        self, tracks: Dict, trajectory_metrics: Dict[int, TrajectoryMetrics]
    ) -> Dict:
        """요약 통계를 포함하여 궤적 분석을 생성합니다."""
        # 기본 통계
        class_counts = {}
        total_confidence = 0.0
        detection_count = 0

        for track in tracks.values():
            class_name = track["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            for frame in track["frame_history"]:
                if "confidence" in frame:
                    total_confidence += frame["confidence"]
                    detection_count += 1

        # 위험한 상호작용 카운트
        dangerous_interactions = sum(
            1
            for metrics in trajectory_metrics.values()
            for intersection in metrics.intersections
            if intersection.is_dangerous
        )

        return {
            "total_unique_objects": len(tracks),
            "object_classes": class_counts,
            "average_detection_confidence": (
                total_confidence / detection_count if detection_count > 0 else 0
            ),
            "trajectory_statistics": {
                "total_intersections": sum(
                    len(metrics.intersections)
                    for metrics in trajectory_metrics.values()
                ),
                "dangerous_interactions": dangerous_interactions,
            },
        }

    @staticmethod
    def _format_metadata_dict(metadata: VideoMetadata) -> Dict:
        """VideoMetadata를 딕셔너리로 포맷합니다."""
        return {
            "total_frames": metadata.total_frames,
            "duration_seconds": metadata.duration_seconds,
            "timestamp": {
                "milliseconds": metadata.timestamp.epoch_ms,
                "datetime_utc": metadata.timestamp.datetime_utc,
            },
            "fps": metadata.fps,
            "resolution": {"width": metadata.width, "height": metadata.height},
        }

    @staticmethod
    def _format_detection_dict(detection: Detection) -> Dict:
        """Detection을 딕셔너리로 포맷합니다."""
        return {
            "track_id": detection.track_id,
            "class": detection.class_name,
            "confidence": detection.confidence,
            "bbox": {
                "x_center": detection.bbox.x_center,
                "y_center": detection.bbox.y_center,
                "width": detection.bbox.width,
                "height": detection.bbox.height,
            },
            "geo_location": {
                "latitude": detection.geo_location.latitude,
                "longitude": detection.geo_location.longitude,
            },
        }

    @staticmethod
    def _format_bbox_dict(bbox: BoundingBox) -> Dict:
        """BoundingBox을 딕셔너리로 포맷합니다."""
        return {
            "x_center": bbox.x_center,
            "y_center": bbox.y_center,
            "width": bbox.width,
            "height": bbox.height,
        }

    @staticmethod
    def _format_geo_point_dict(point: GeoPoint) -> Dict:
        """GeoPoint을 딕셔너리로 포맷합니다."""
        return {"latitude": point.latitude, "longitude": point.longitude}

    @staticmethod
    def _format_timestamp_dict(timestamp: Timestamp) -> Dict:
        """Timestamp을 딕셔너리로 포맷합니다."""
        return {
            "relative_ms": timestamp.relative_ms,
            "milliseconds": timestamp.epoch_ms,
            "datetime_utc": timestamp.datetime_utc,
        }

    def save_results(self, results: Dict, output_path: str) -> None:
        """처리 결과를 JSON 파일로 저장합니다."""
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            self.logger.info(f"결과가 {output_path}에 저장되었습니다.")
        except Exception as e:
            self.logger.error(f"결과 저장 실패: {str(e)}")
            raise
