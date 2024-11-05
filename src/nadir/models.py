# src/nadir/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


import pandas as pd


@dataclass
class PixelCoordinate:
    """Image pixel coordinate"""

    x: float
    y: float


@dataclass
class ImageDimensions:
    """Image dimensions in pixels"""

    width: int
    height: int


@dataclass
class GeoPoint:
    """Geographic coordinate point"""

    latitude: float
    longitude: float


@dataclass
class DroneState:
    """Drone position and orientation state"""

    position: GeoPoint
    altitude_feet: float
    heading_degrees: float


@dataclass
class Timestamp:
    """Video frame timestamp information"""

    epoch_ms: int
    relative_ms: int
    datetime_utc: str


@dataclass
class BoundingBox:
    """Object detection bounding box"""

    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class Detection:
    """Single object detection result"""

    track_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    geo_location: GeoPoint


@dataclass
class VideoMetadata:
    """Video file metadata"""

    total_frames: int
    duration_seconds: float
    timestamp: Timestamp
    fps: float
    width: int
    height: int


@dataclass
class DroneFrame:
    """Single frame data with drone state"""

    number: int
    timestamp: Timestamp
    drone_state: DroneState
    detections: List[Detection] = field(default_factory=list)


@dataclass
class VideoSegment:
    """Video segment with synchronized drone data"""

    start_time: float
    end_time: float
    frames: List[DroneFrame] = field(default_factory=list)
    video_path: Optional[str] = None
    fps: float = 30.0
    log_data: Optional[pd.DataFrame] = None

    @property
    def duration_ms(self) -> float:
        return self.end_time - self.start_time
