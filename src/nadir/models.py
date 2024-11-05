# src/nadir/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class PixelCoordinate:
    """이미지 픽셀 좌표"""

    x: float
    y: float


@dataclass
class ImageDimensions:
    """픽셀 단위 이미지 크기"""

    width: int
    height: int


@dataclass
class GeoPoint:
    """지리 좌표 점"""

    latitude: float
    longitude: float


@dataclass
class DroneState:
    """드론 위치 및 방향 상태"""

    position: GeoPoint
    altitude_feet: float
    heading_degrees: float


@dataclass
class Timestamp:
    """비디오 프레임 타임스탬프 정보"""

    epoch_ms: int
    relative_ms: int
    datetime_utc: str


@dataclass
class BoundingBox:
    """객체 감지 경계 상자"""

    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class Detection:
    """단일 객체 감지 결과"""

    track_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    geo_location: GeoPoint


@dataclass
class VideoMetadata:
    """비디오 파일 메타데이터"""

    total_frames: int
    duration_seconds: float
    timestamp: Timestamp
    fps: float
    width: int
    height: int


@dataclass
class DroneFrame:
    """드론 상태와 함께한 단일 프레임 데이터"""

    number: int
    timestamp: Timestamp
    drone_state: DroneState
    detections: List[Detection] = field(default_factory=list)


@dataclass
class VideoSegment:
    """동기화된 드론 데이터를 포함한 비디오 세그먼트"""

    start_time: float
    end_time: float
    frames: List[DroneFrame] = field(default_factory=list)
    video_path: Optional[str] = None
    fps: float = 30.0
    log_data: Optional[pd.DataFrame] = None

    @property
    def duration_ms(self) -> float:
        return self.end_time - self.start_time
