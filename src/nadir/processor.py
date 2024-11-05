import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
from pydantic import BaseModel

from .interfaces import (
    IFrameProcessor,
    IGeoCalculator,
    ILogMapper,
    IObjectDetector,
    IResultWriter,
    ITrajectoryAnalyzer,
)
from .models import (
    Timestamp,
    VideoMetadata,
    VideoSegment,
    TrajectoryMetrics,
)


class ObjectCount(BaseModel):
    """객체 클래스별 카운트"""

    total_unique_objects: int
    object_classes: Dict[str, int]
    average_detection_confidence: float


class TrajectoryStatistics(BaseModel):
    """궤적 분석 통계"""

    total_intersections: int
    dangerous_interactions: int


class VideoSummary(BaseModel):
    """비디오 처리 결과 요약"""

    total_unique_objects: int
    object_classes: Dict[str, int]
    average_detection_confidence: float
    trajectory_statistics: TrajectoryStatistics


class VideoResult(BaseModel):
    """단일 비디오 처리 결과"""

    metadata: Dict
    frames: List[Dict]
    tracks: List[Dict]
    trajectory_analysis: Dict[str, Dict]
    summary: VideoSummary


class DroneVideoProcessor:
    def __init__(
        self,
        detector: IObjectDetector,
        geo_calculator: IGeoCalculator,
        trajectory_analyzer: ITrajectoryAnalyzer,
        log_mapper: ILogMapper,
        frame_processor: IFrameProcessor,
        result_writer: IResultWriter,
    ):
        self.detector = detector
        self.geo_calculator = geo_calculator
        self.trajectory_analyzer = trajectory_analyzer
        self.log_mapper = log_mapper
        self.frame_processor = frame_processor
        self.result_writer = result_writer
        self.logger = logging.getLogger(__name__)

    def process_videos(self, video_paths: List[str]) -> Dict[str, Dict]:
        try:
            mapped_segments = self.log_mapper.map_videos_to_segments(video_paths)
            results = {}

            for video_path, segment in mapped_segments.items():
                result = self._process_single_video(video_path, segment)
                results[Path(video_path).name] = result.model_dump()

            return results
        except Exception as e:
            self.logger.error(f"비디오 처리 실패: {str(e)}")
            raise

    def _process_single_video(
        self, video_path: str, segment: VideoSegment
    ) -> VideoResult:
        try:
            metadata = self._extract_video_metadata(video_path)
            tracks = {}
            processed_frames = []

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오 열기 실패: {video_path}")

            for frame_data in segment.frames:
                success, frame = cap.read()
                if not success:
                    break

                processed_frame = self.frame_processor.process_frame(
                    frame, frame_data, metadata, tracks
                )
                if processed_frame:
                    processed_frames.append(processed_frame)

            cap.release()

            # 궤적 분석
            trajectory_metrics = self.trajectory_analyzer.analyze_trajectories(
                tracks, metadata.fps
            )

            summary = self._generate_summary(tracks, trajectory_metrics)

            return VideoResult(
                metadata=metadata.model_dump(),
                frames=processed_frames,
                tracks=list(tracks.values()),
                trajectory_analysis={
                    str(track_id): metrics.model_dump()
                    for track_id, metrics in trajectory_metrics.items()
                },
                summary=summary,
            )

        except Exception as e:
            self.logger.error(f"{video_path} 처리 실패: {str(e)}")
            raise

    def _extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """비디오 파일에서 메타데이터 추출"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오 열기 실패: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = frame_count / fps if fps > 0 else 0
            creation_time = int(Path(video_path).stat().st_mtime * 1000)

            # Unix timestamp를 ISO 형식으로 변환
            dt = datetime.fromtimestamp(creation_time / 1000.0, tz=timezone.utc)
            iso_time = dt.isoformat()

            cap.release()

            return VideoMetadata(
                total_frames=frame_count,
                duration_seconds=duration_seconds,
                timestamp=Timestamp(
                    epoch_ms=creation_time, relative_ms=0, datetime_utc=iso_time
                ),
                fps=fps,
                width=width,
                height=height,
            )

        except Exception as e:
            self.logger.error(f"{video_path}에서 메타데이터 추출 실패: {str(e)}")
            raise

    def _generate_summary(
        self, tracks: Dict, trajectory_metrics: Dict[int, TrajectoryMetrics]
    ) -> VideoSummary:
        """요약 통계 생성"""
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

        dangerous_interactions = sum(
            1
            for metrics in trajectory_metrics.values()
            for intersection in metrics.intersections
            if intersection.is_dangerous
        )

        return VideoSummary(
            total_unique_objects=len(tracks),
            object_classes=class_counts,
            average_detection_confidence=(
                total_confidence / detection_count if detection_count > 0 else 0
            ),
            trajectory_statistics=TrajectoryStatistics(
                total_intersections=sum(
                    len(metrics.intersections)
                    for metrics in trajectory_metrics.values()
                ),
                dangerous_interactions=dangerous_interactions,
            ),
        )

    def save_results(self, results: Dict, output_path: str) -> None:
        """처리 결과 저장"""
        self.result_writer.save_results(results, output_path)
