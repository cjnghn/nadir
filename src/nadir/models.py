# src/nadir/models.py
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class PixelCoordinate(BaseModel):
    """이미지 내 픽셀 좌표"""

    x: float
    y: float


class ImageDimensions(BaseModel):
    """이미지 크기 정보"""

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class GeoPoint(BaseModel):
    """지리적 좌표점"""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class DroneState(BaseModel):
    """드론의 상태 정보"""

    position: GeoPoint
    altitude_feet: float = Field(ge=0)
    heading_degrees: float = Field(ge=0, le=360)


class Timestamp(BaseModel):
    """시간 정보"""

    epoch_ms: int = Field(ge=0)
    relative_ms: int = Field(ge=0)
    datetime_utc: str

    @field_validator("datetime_utc")
    def validate_datetime(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError("유효한 ISO 형식이어야 합니다")


class BoundingBox(BaseModel):
    """객체 감지 경계 상자"""

    x_center: float
    y_center: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """객체 감지 결과"""

    track_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    geo_location: Optional[GeoPoint] = None


class Intersection(BaseModel):
    """궤적 교차 정보"""

    track_id1: int
    track_id2: int
    frame_number: float
    intersection_point: Tuple[float, float]
    psm: float = Field(ge=0)
    is_dangerous: bool
    approaching_angle: float = Field(ge=0, le=180)


class TrajectoryMetrics(BaseModel):
    """궤적 분석 메트릭"""

    track_id: int
    total_distance: float = Field(ge=0)
    average_speed: float = Field(ge=0)
    max_speed: float = Field(ge=0)
    intersections: List[Intersection] = Field(default_factory=list)

    @property
    def dangerous_intersection_count(self) -> int:
        return sum(1 for i in self.intersections if i.is_dangerous)


class VideoMetadata(BaseModel):
    """비디오 파일 메타데이터"""

    total_frames: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    timestamp: Timestamp
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DroneFrame(BaseModel):
    """단일 비디오 프레임 정보"""

    number: int = Field(ge=0)
    timestamp: Timestamp
    drone_state: DroneState
    detections: List[Detection] = Field(default_factory=list)


class VideoSegment(BaseModel):
    """비디오 세그먼트 정보"""

    start_time: float
    end_time: float
    frames: List["DroneFrame"] = Field(default_factory=list)
    video_path: Optional[str] = None
    fps: float = Field(gt=0, default=30.0)

    @model_validator(mode="after")
    def validate_times(self) -> "VideoSegment":
        """시작 시간이 종료 시간보다 작은지 검증"""
        if self.end_time < self.start_time:
            raise ValueError("종료 시간은 시작 시간보다 커야 합니다")
        return self

    @property
    def duration_ms(self) -> float:
        return self.end_time - self.start_time
