# src/nadir/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import (
    Detection,
    DroneFrame,
    TrajectoryMetrics,
    VideoMetadata,
    VideoSegment,
)


class IObjectDetector(ABC):
    @abstractmethod
    def detect_and_track(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> List[Detection]:
        """프레임에서 객체를 감지하고 추적"""
        pass


class ITrajectoryAnalyzer(ABC):
    @abstractmethod
    def analyze_trajectories(
        self, tracks: Dict[int, Dict], fps: float
    ) -> Dict[int, TrajectoryMetrics]:
        """객체 궤적을 분석하고 메트릭 계산"""
        pass


class IGeoCalculator(ABC):
    @abstractmethod
    def calculate_geo_location(
        self,
        pixel_x: float,
        pixel_y: float,
        frame_width: int,
        frame_height: int,
        drone_lat: float,
        drone_lon: float,
        drone_alt_feet: float,
        heading_degrees: float,
    ) -> Dict[str, float]:
        """픽셀 좌표를 지리 좌표로 변환"""
        pass


class ILogMapper(ABC):
    @abstractmethod
    def map_videos_to_segments(self, video_paths: List[str]) -> Dict[str, VideoSegment]:
        """비디오를 로그 세그먼트에 매핑"""
        pass


class IResultWriter(ABC):
    @abstractmethod
    def save_results(self, results: Dict, output_path: str) -> None:
        """처리 결과를 저장"""
        pass


class IFrameProcessor(ABC):
    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        frame_data: DroneFrame,
        metadata: VideoMetadata,
        tracks: Dict,
    ) -> Optional[Dict]:
        """단일 프레임 처리"""
        pass
